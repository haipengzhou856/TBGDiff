import yaml
from utils.utils import *
import accelerate
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from engine.tester import testing_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ViShaVideo_STEDiff.yml", type=str,
                        help='Path to the configuration file')
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:  # change the config here
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    # check out_dir
    time_stamp = "INFERENCE"
    out_dir = config.OUTPUT.HOME + config.OUTPUT.DATA_NAME + "/" + config.OUTPUT.MODEL_NAME + time_stamp
    tb_path = check_dir(out_dir + config.OUTPUT.TB)  # tensorboard
    log_path = check_dir(out_dir + config.OUTPUT.LOG)  # logging

    result_path = check_dir(out_dir + config.OUTPUT.RESULT)  # store the test results
    # copy_folder_without_images("/home/haipeng/Code/Data/ViSha/test/labels", result_path)
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir

    # My GPUs not support the NCCL
    # Use gloo if you meet stuck
    import torch.distributed as dist
    dist.init_process_group(backend="gloo")

    set_seed(config.SEED)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=2)
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
    logger.info(
        "--------------------USE {model_name}-----------------------".format(model_name=config.OUTPUT.MODEL_NAME))

    model_name = config.OUTPUT.MODEL_NAME

    from engine.trainer import training_func

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

    model = model.to(device=accelerator.device)
    weight_path = config.MODEL.CKPT_PATH
    testing_func(config, accelerator, model, logger, weight_path, result_path)

    logger.info("----------------------END RUN----------------------------")

