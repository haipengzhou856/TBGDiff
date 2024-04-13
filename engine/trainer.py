import torch
from tqdm import tqdm
from dataset.visha_dataset_video_clip_v2 import ViSha_Dataset
from torch.utils.data import DataLoader
import os
from utils.metrix import computeALL
from utils.utils import *
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch import nn
from accelerate.utils import set_seed


def training_func(cfg, accelerator, model, logger, writer, ckpt_path, result_path):
    set_seed(cfg.SEED)
    train_dataset = ViSha_Dataset(cfg, mode="train")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.DATASET.BATCH_SIZE,
                              num_workers=cfg.DATASET.NUM_WORKERS,
                              shuffle=True)

    test_dataset = ViSha_Dataset(cfg, mode="test")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             num_workers=cfg.DATASET.NUM_WORKERS)

    logger.info("-----------------Finish dataloader----------------")

    # hyps
    total_epoch = cfg.SOLVER.EPOCH
    lr = cfg.SOLVER.LR
    milestongs = cfg.SOLVER.MILESTONGS
    gamma = cfg.SOLVER.GAMMA
    momentum = cfg.SOLVER.MOMENTUM
    decay = cfg.SOLVER.DECAY
    store_epoch = cfg.OUTPUT.STORE_EPOCH
    device = accelerator.device
    resume_path = cfg.MODEL.RESUME_PATH

    if cfg.SOLVER.OPTIM == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr, momentum=momentum, weight_decay=decay)
    elif cfg.SOLVER.OPTIM == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=lr, weight_decay=decay)
    else:
        assert False, "CHOOSE THE OPTIMIZER!!!"

    if cfg.SOLVER.LR_SCHEDULE:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestongs, gamma=gamma, last_epoch=-1)
        # you need manually set the best lr strategy
    else:
        scheduler = None

    model.to(device)


    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_loader)

    logger.info("----------------Starting training------------------")
    logger.info(f"--------------Total  {total_epoch} Epochs--------------")

    accelerator.print(resume_path)
    if resume_path != "":  # if resume
        logger.info(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        path = os.path.basename(resume_path)
        starting_epoch = int(path.replace("ckpt_epoch", "")) + 1
    else:
        starting_epoch = 1

    overall_step = 0
    for epoch in range(starting_epoch, total_epoch + 1):
        for idx, batch in enumerate(tqdm(train_loader)):
            model.train()
            frames, labels,boundaries = batch["image"], batch["label"],batch["boundary"]
            labels = (labels > 0.5).to(torch.long)  # the label seems un-standard, apply this threshold
            boundaries = (boundaries > 0.5).to(torch.long)
            accelerator.wait_for_everyone()
            # batch_size, num_time, h, w = frames.shape[0], frames.shape[1], frames.shape[3], frames.shape[4]
            # the huggingface require 3 dims for binary segment, drop the C dim.

            ################# training detail ###################
            ################ model gives logits, add loss function here ###########
            pseudo_mask = None
            if cfg.OUTPUT.DATA_NAME == "STEDiff":
                loss, pred_mask,pseudo_mask,edge = model(frames, labels,boundaries)
            else:
                loss, pred_mask = model(frames, labels)
            accelerator.backward(loss)

            ################# end detail #######################
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            overall_step += 1
            writer.add_scalar("loss", loss, overall_step)
            writer.add_scalar("lr:", optimizer.state_dict()['param_groups'][0]['lr'], overall_step)

            accelerator.wait_for_everyone()
            ################### monitor #################
            if overall_step % 40 == 0:
                logger.info("Current step:{step}, loss:{loss}, epoch:{epoch}, lr:{lr}".format(
                    step=overall_step, loss=loss, epoch=epoch, lr=optimizer.state_dict()['param_groups'][0]['lr']))
                writer.add_image("pred_{ii}".format(ii=0), pred_mask[0, 0], overall_step)
                writer.add_image("gt_{ii}".format(ii=0), labels[0, 0], overall_step)
                writer.add_image("boundary_{ii}".format(ii=0), boundaries[0, 0], overall_step)
                writer.add_image("img_{ii}".format(ii=0), reverse_normalize(frames[0, 0]), overall_step)
                if pseudo_mask is not None:
                    writer.add_image("raw_{ii}".format(ii=0), pseudo_mask[0, 0], overall_step)
                    writer.add_image("raw_edge_{ii}".format(ii=0), edge[0, 0], overall_step)
            ######################### store and eval #####################
        if epoch % store_epoch == 0:
        #if overall_step % 10 == 0:
            logger.info(
                "----------------Save ckpt_epoch{epoch}------------------".format(epoch=epoch))
            restore_path = os.path.join(ckpt_path, "ckpt_epoch{epoch}".format(epoch=epoch))
            # XXX/ckpt_apth/ckpt_epoch/--bin --optimizer
            accelerator.wait_for_everyone()
            accelerator.save_state(restore_path)
            if epoch > 0:
                logger.info(
                    "----------Starting Testing, now is step:{step} epoch:{epoch}-----------".format(
                        step=overall_step,
                        epoch=epoch))
                model.eval()
                for _, batch in enumerate(tqdm(test_loader)):
                    img, label, gt_path = batch["image"].to(device), batch["label"].to(device), batch["label_path"]
                    ori_h, ori_w = batch["h"][0], batch["w"][0]
                    # img = img.flatten(start_dim=0, end_dim=1).contiguous()

                    # this is for hugging face testing
                    with torch.no_grad():
                        _, prediction = model(img, is_train=False)
                        resized_pred = torch.nn.functional.interpolate(
                            prediction[0], size=(ori_h, ori_w), mode="bilinear", align_corners=False)
                        for pred, path in zip(resized_pred, gt_path):
                            pred = (pred > 0.5).to(float)
                            # pred = pred.to(float)
                            img = Tensor2PIL(pred)
                            video_name, frame_name = path[0].split("/")[-2:][0], path[0].split("/")[-2:][1]
                            folder_path = os.path.join(result_path, "ckpt_epoch{epoch}".format(epoch=epoch),
                                                       video_name)
                            accelerator.wait_for_everyone()
                            check_dir(folder_path)
                            cur_path = os.path.join(folder_path, frame_name)
                            img.save(cur_path)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    metrix = computeALL(gt_path="/home/haipeng/Code/Data/ViSha/test/labels",
                                        pred_path=os.path.join(result_path,
                                                               "ckpt_epoch{epoch}".format(epoch=epoch)))
                    logger.info(
                        "IoU:{IoU},F_beta:{F},MAE:{mae},BER:{BER},SBER:{SB},NBER:{NB}".format(IoU=metrix["IoU"],
                                                                                              F=metrix["Fmeasure"],
                                                                                              mae=metrix["MAE"],
                                                                                              BER=metrix["BER"],
                                                                                              SB=metrix["S-BER"],
                                                                                              NB=metrix["N-BER"]))
                    writer.add_scalar("IoU", metrix["IoU"], epoch)
                    writer.add_scalar("F_beta", metrix["Fmeasure"], epoch)
                    writer.add_scalar("MAE", metrix["MAE"], epoch)
                    writer.add_scalar("BER", metrix["BER"], epoch)
                    writer.add_scalar("SBER", metrix["S-BER"], epoch)
                    writer.add_scalar("NBER", metrix["N-BER"], epoch)