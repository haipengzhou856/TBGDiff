## Timeline and Boundary Guided Diffusion Network for Video Shadow Detection

#### News: [2024/07]

This work has been accepted by ACM MM 2024. Thanks for the co-authos.

## Quick View

This is an official repo for TBGDiff.  We use Diffusion for video shadow detection. 

 **I don't like this incremental shit work, even it is accepted.** [see why](##Others)

But you can freely stolen the codes, I guess it maybe helpful.

Visit our [project page](https://haipengzhou856.github.io/paper_page/TBGDiff/TBGDiff.html) to find more details. Thanks for your attention : )

## Preparation

### dataset

Please see [ViSha](https://erasernut.github.io/ViSha.html) to download it, and modify the data dir_path in `configs/xxx.yml`. 

### packages

My basic environment: `Python=3.8, CUDA=12.2, ` ~~Pytorch=1.11.0~~ . `medpy` is not suggested to install since it does not support latest `numpy`. I have modified the **matrix.py** borrowed from [scotch&soda (cvpr23)](https://github.com/lihaoliu-cambridge/scotch-and-soda/blob/main/model/scotch_and_soda.py) . Check the `environment.yaml` to install other required packages. 



**News: ** Please update the `pytorch ` **with higher version (1.12.1 or higher)** for using the resume training. I found that the older version will produce some bugs (see [here](https://stackoverflow.com/questions/73095460/assertionerror-if-capturable-false-state-steps-should-not-be-cuda-tensors)) for the optimizer. And using lower version of `accelerate` (see next section)

### Pretrained backbone

To reproduce and train our method, you should download the pre-trained `Segformer` model for initialization. See [huggingface-segformer](https://huggingface.co/nvidia/segformer-b3-finetuned-ade-512-512/tree/main)  to download the corresponding files.

In the `model/PEEDiff.py, STEDiff.py, Pix2Seq.py` , please modify the path. 

```python
def Segformer():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/nvidia/segformer-b3-finetuned-ade-512-512",
        ignore_mismatched_sizes=True,
        num_labels=1)
    return model

def SegformerWithMask():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/nvidia/segformer-b1-finetuned-ade-512-512",
        ignore_mismatched_sizes=True,
        num_labels=1,
        num_channels=4)
    return model
```

BTW, the `engine/tester.py & trainer.py` should be modified as well:

```python
if accelerator.is_main_process:
                    metrix = computeALL(gt_path="/home/haipeng/Code/Data/ViSha/test/labels",
           pred_path=os.path.join(result_path,"ckpt_epoch{epoch}".format(epoch=epoch)))
```

Please change the `gt_path` for your platform.

## Off-the-shelf Results

If you guys do not want to reproduce ours and other methods, directly download the visual results on [Google Drive](https://drive.google.com/file/d/1ek2AIsEXXNvozA1cllfWIYYNzoFtupXl/view?usp=drive_link) for a convenience.

Most of them are provided by  [scotch&soda (cvpr23)](https://github.com/lihaoliu-cambridge/scotch-and-soda/tree/main). We add some latest works.



## Experiments

### Set up

#### Using **[accelerate](https://huggingface.co/docs/accelerate/index)** 🤗 to launch distributed training

Hate redundant scripts of DDP in PyTorch?   **ALL YOU NEED IS ONLY ONE LINE SNIPPT !!!**

Please see the details of the **[accelerate documents](https://huggingface.co/docs/accelerate/index)** to conduct GPUs settings.

Recommend to use lower version of it, or your checkpoint file is saved as `model.safetensor`  rather than `pytorch_model.bin` .   It may lead to some `missing key errors` when loading the weights.

Recommended version is:

```
pip install accelerate==0.20.3
```

#### FOLLOW ME:

1. Customize the configurations on your machine 

```
accelerate config
```

	or you can use the same settings as mine `accelerate_cfg.yaml`.

2.  Train/Test the model with one command line (see next sections)

3.  😎 Play games or listen music to wait the results. 😎 



### Training

**Run the fool snippet:**

```
accelerate launch --config_file accelerate_cfg.yaml --main_process_port 29050 main.py --config "configs/ViShaVideo_STEEDiff.yml"
```

And it will produce following outputs

```
--output
----model_name
------exp_name
--------ckpt_path       # store the ckpt
--------log_path        # store the logs
--------pred_result     # store the predictions
--------tb_path         # store the tensorboard information
```



### Inference

Actually we testing every epoch during training stage, and if you just want to infer our **well-trained** model, please download the [**ckpt of TBGDiff on Google Drive**](https://drive.google.com/file/d/1ELEOTQOXDfQ2n5WNy2AKMzQNUiMl6V4M/view?usp=drive_link)  and unzip it in the project (see the `MODEL.CKPT_PATH` in the config `yml` file) , then run the snippet:

```
accelerate launch --config_file accelerate_cfg.yaml --main_process_port 29050 inference.py --config "configs/ViShaVideo_STEDiff.yml"
```

It also produces following outputs

```
--output
----model_name
------exp_name
--------ckpt_path       # store the ckpt
--------log_path        # store the logs
--------pred_result     # store the predictions
--------tb_path         # store the tensorboard information
```

## Other Ablation Studies

Very long time ago, I have finished some basic experiments in other machines. I only find out the PEEDiff (i.e., PEE+DSA, without SBBA) and the ckpt is also provided in [here](https://drive.google.com/file/d/1dV0FqFROHXLm9rHy3VemglKjZvbAlosg/view?usp=drive_link). Other ablation studied models are not compatible since I've changed some module structures leading to wrong weight matching. I'm lazy to figure out 🤪. You may manually to test the effectiveness.



The performance of PEEDiff is also competitive. We have `IOU:0.645, F_beta:0.780, MAE:0.0286, BER:10.39, SBER:19.37, NBER:1.40`

For testing the PEEDiff, download the [**ckpt on Google Drive**](https://drive.google.com/file/d/1dV0FqFROHXLm9rHy3VemglKjZvbAlosg/view?usp=drive_link)  and unzip it, then run the command

```
accelerate launch --config_file accelerate_cfg.yaml --main_process_port 29050 inference.py --config "configs/ViShaVideo_PEEDiff.yml"
```



## TODO

* I still think this proposed approach is generalized for all the VOS tasks (the baseline and final model really works well), and I will deploy it to other Video Object Segmentation dataset. Wait for my technical report : )

  

## Note

* We conduct all the experiments with 4GPUs to get stable results, but when we transfer to single or two GPUs the performance gets decrease. This is may caused by  the **batch-size**.  We find larger batch size performs better,  24G may not enough. You may adjust it in customized ways. 
* Some communication proxies may be have problems (e.g., the dataloader may stuck), check the `NCCL  ` is available for your machine, and my proxy is under `gloo`.



## Others

* I think this work is a PIECE OF SHIT in my standard. I **ABSOLUTELY HATE,DISGUST and DESPISE A+B+C** ***SHIT*** to speculate and trick a conference or journal  (though the designed module is useful and make sense indeed). I **EXTREMELY HATE** such an academic taste. I finish this work just for my job. 
* Welcome to refer, stole and modify my code. If you are distressed and tired (just like me, no patience to read the documents lol 🤣) on modifying Highly Integrated Architecture like [MMSeg](https://github.com/open-mmlab/mmsegmentation), [Detectron2](https://github.com/facebookresearch/detectron2), [Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning) even [HuggingFace Trainer](https://huggingface.co/docs/transformers/v4.14.1/en/main_classes/trainer) ,  try my workflow. Maybe my proposed network is quite complex, just remove and modify them with your own network and design. I think my pipeline of the code production is enough clear, tidy to read and graceful. I will also arrange and open-source the code of workflow for specializing in common `cls&seg&det` task, wait for my masterpiece  :p
* Again, you can freely follow the code, but not suggest the taste. I do it just for responsibility bro :(  **Expand the knowledge boundary is research, Build the Lego Bricks is toy player and shit.**



## Acknowledgement

* Thanks for  Hongqiu, [Tian Ye](https://owen718.github.io/), and [Zhaohu](https://ge-xing.github.io/)  for their great idea and discussion! 
* Thanks for our great [TEAM](https://sites.google.com/site/indexlzhu/team)! My pleasure and lucky to join this group! 
* Thanks for the open-source researchers! 

## Citation

If this paper and code are useful in your research, please consider citing:
