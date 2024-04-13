import yaml
from utils.utils import *
import accelerate
from accelerate import DistributedDataParallelKwargs
import time
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed
from engine.trainer import training_func
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="configs/ViShaVideo_STEDiff.yml", type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:  # change the config here
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    # check out_dir
    time_stamp = time.strftime("%m%d%H%M", time.localtime())
    # add timestamp to distinguish
    # exp: /output/visda/exp09121200/tb_path ##
    out_dir = config.OUTPUT.HOME + config.OUTPUT.DATA_NAME + "/" + config.OUTPUT.MODEL_NAME + time_stamp
    tb_path = check_dir(out_dir + config.OUTPUT.TB)  # tensorboard
    ckpt_path = check_dir(out_dir + config.OUTPUT.CKPT)  # checkpoint
    log_path = check_dir(out_dir + config.OUTPUT.LOG)  # logging

    result_path = check_dir(out_dir + config.OUTPUT.RESULT)  # store the test results
    # copy_folder_without_images("/home/haipeng/Code/Data/ViSha/test/labels", result_path)
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir

    import torch.distributed as dist
    dist.init_process_group(backend="gloo")


    set_seed(config.SEED)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    writer = SummaryWriter(tb_path)
    # logger init
    logger = setup_logger(config.OUTPUT.DATA_NAME,
                          log_path,
                          accelerator.process_index,
                          "log.txt")

    # init env
    logger.info("----------------------NEW RUN----------------------------")
    logger.info("----------------------Basic Setting----------------------------")
    logger.info("---work place in: {dir}-----".format(dir=out_dir))
    logger.info("Img_size: {}".format(config.DATASET.IMG_SIZE))
    logger.info("TIME_CLIP: {}".format(config.DATASET.TIME_CLIP))
    logger.info("BATCH_SIZE: {}".format(config.DATASET.BATCH_SIZE))
    logger.info("lr: {}".format(config.SOLVER.LR))
    logger.info("opim: {}".format(config.SOLVER.OPTIM))

    logger.info("----------------------Diffusion----------------------------")
    logger.info("timestep: {}".format(config.DIFFUSION.TIMESTEPS))
    logger.info("BitScale: {}".format(config.DIFFUSION.SCALE))
    logger.info("Scheduler: {}".format(config.DIFFUSION.SCHEDULER))
    logger.info("TimeDifference: {}".format(config.DIFFUSION.TD))
    logger.info(
        "--------------------USE {model_name}-----------------------".format(model_name=config.OUTPUT.MODEL_NAME))
    logger.info(
        "Using {num_gpu} GPU for training, {mix_pix} mix_precision used.".format(num_gpu=accelerator.num_processes,
                                                                                 mix_pix=accelerator.mixed_precision))
    model_name = config.OUTPUT.MODEL_NAME


    if "PEDiff" in model_name:
        from models.PEEDiff import PEEDiff, Segformer

        pretrain_model = Segformer()
        model = PEEDiff(PretrainedSegformer=pretrain_model,
                        bit_scale=config.DIFFUSION.SCALE,
                        timesteps=config.DIFFUSION.TIMESTEPS,
                        noise_schedule=config.DIFFUSION.SCHEDULER,
                        time_difference=config.DIFFUSION.TD,
                        num_frames=config.DATASET.TIME_CLIP)

    elif "Pix2Seq" in model_name:
        from models.Pix2Seq import Pix2Seq, Segformer

        pretrain_model = Segformer()
        model = Pix2Seq(PretrainedSegformer=pretrain_model,
                        bit_scale=config.DIFFUSION.SCALE,
                        timesteps=config.DIFFUSION.TIMESTEPS,
                        noise_schedule=config.DIFFUSION.SCHEDULER,
                        time_difference=config.DIFFUSION.TD,
                        num_frames=config.DATASET.TIME_CLIP)

    elif "STEDiff" in model_name:
        from models.STEDiff import STAN, Segformer

        pretrain_model = Segformer()
        model = STAN(PretrainedSegformer=pretrain_model,
                     bit_scale=config.DIFFUSION.SCALE,
                     timesteps=config.DIFFUSION.TIMESTEPS,
                     noise_schedule=config.DIFFUSION.SCHEDULER,
                     time_difference=config.DIFFUSION.TD,
                     num_frames=config.DATASET.TIME_CLIP)

    else:
        assert "NO MODEL IMPLEMENTED"


    def cal_param(model):
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue

        print(f'Total params: {Total_params / 1e6}M')
        print(f'Trainable params: {Trainable_params / 1e6}M')
        print(f'Non-trainable params: {NonTrainable_params / 1e6}M')
    cal_param(model)

    model = model.to(device=accelerator.device)
    training_func(config, accelerator, model, logger, writer, ckpt_path, result_path)

    logger.info("----------------------END Training RUN----------------------------")

