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


def testing_func(cfg, accelerator, model, logger, ckpt_path, result_path):
    set_seed(cfg.SEED)
    test_dataset = ViSha_Dataset(cfg, mode="test")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             num_workers=cfg.DATASET.NUM_WORKERS)

    logger.info("-----------------Finish Testing Dataloader----------------")

    # hyps
    device = accelerator.device

    model.to(device)

    model, test_loader = accelerator.prepare(
        model, test_loader)

    logger.info(f"Inference ckpt with: {ckpt_path}")
    accelerator.load_state(ckpt_path)
    ckpt_number = os.path.basename(ckpt_path)


    logger.info(
        "----------Starting Testing, the ckpt epoch is:{epoch}-----------".format(
            epoch=ckpt_number))
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
                img = Tensor2PIL(pred)
                video_name, frame_name = path[0].split("/")[-2:][0], path[0].split("/")[-2:][1]
                folder_path = os.path.join(result_path, ckpt_number,video_name)
                accelerator.wait_for_everyone()
                check_dir(folder_path)
                cur_path = os.path.join(folder_path, frame_name)
                img.save(cur_path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrix = computeALL(gt_path="/home/haipeng/Code/Data/ViSha/test/labels",
                            pred_path=os.path.join(result_path,ckpt_number))
        logger.info(
            "IoU:{IoU},F_beta:{F},MAE:{mae},BER:{BER},SBER:{SB},NBER:{NB}".format(IoU=metrix["IoU"],
                                                                                  F=metrix["Fmeasure"],
                                                                                  mae=metrix["MAE"],
                                                                                  BER=metrix["BER"],
                                                                                  SB=metrix["S-BER"],
                                                                                  NB=metrix["N-BER"]))

