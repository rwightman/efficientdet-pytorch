# EfficientDet (PyTorch)

A PyTorch implementation of EfficientDet.

It is based on the
* official Tensorflow implementation by [Mingxing Tan and the Google Brain team](https://github.com/google/automl)
* paper by Mingxing Tan, Ruoming Pang, Quoc V. Le [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) 

There are other PyTorch implementations. Either their approach didn't fit my aim to correctly reproduce the Tensorflow models (but with a PyTorch feel and flexibility) or they cannot come close to replicating MS COCO training from scratch.

Aside from the default model configs, there is a lot of flexibility to facilitate experiments and rapid improvements here -- some options based on the official Tensorflow impl, some of my own:
* BiFPN connections and combination mode are fully configurable and not baked into the model code
* BiFPN and head modules can be switched between depthwise separable or standard convolutions
* Activations, batch norm layers are switchable via arguments (soon config)
* Any backbone in my `timm` model collection that supports feature extraction (`features_only` arg) can be used as a bacbkone.

## Updates

### 2022-01-06
* New `efficientnetv2_ds` weights 50.1 mAP @ 1024x0124, using AGC clipping and `timm`'s `efficientnetv2_rw_s` backbone. Memory use comparable to D3, speed faster than D4. Smaller than optimal training batch size so can probably do better... 

### 2021-11-30
* Update `efficientnetv2_dt` weights to a new set, 46.1 mAP @ 768x768, 47.0 mAP @ 896x896 using AGC clipping.
* Add AGC (Adaptive Gradient Clipping support via `timm`). Idea from (`High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171)
* `timm` minimum version bumped up to 0.4.12

### 2021-11-16
* Add EfficientNetV2 backbone experiment `efficientnetv2_dt` based on `timm`'s `efficientnetv2_rw_t` (tiny) model. 45.8 mAP @ 768x768.
* Updated TF EfficientDet-Lite model defs incl weights ported from official impl (https://github.com/google/automl)
* For Lite models, updated feature resizing code in FPN to be based on feat size instead of reduction ratios, needed to support image size that aren't divisible by 128.
* Minor tweaks, bug fixes

### 2021-07-28
* Add training example to README provided by Chris Hughes for training w/ custom dataset & Lightning training code
  * [Medium blog post](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f)
  * [Python notebook](https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7)

### 2021-04-30
* Add EfficientDet AdvProp-AA weights for D0-D5 from TF impl. Model names `tf_efficientdet_d?_ap`
  * See https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md

### 2021-02-18
* Add some new model weights with bilinear interpolation for upsample and downsample in FPN.
  * 40.9 mAP - `efficientdet_q1`  (replace prev model at 40.6)
  * 43.2 mAP -`cspresdet50`
  * 45.2 mAP - `cspdarkdet53m`

### 2020-12-07
* Training w/ fully jit scripted model + bench (`--torchscript`) is possible with inclusion of ModelEmaV2 from `timm` and previous torchscript compat additions. Big speed gains for CPU bound training.
* Add weights for alternate FPN layouts. QuadFPN experiments (`efficientdet_q0/q1/q2`) and CSPResDeXt + PAN (`cspresdext50pan`). See updated table below. Special thanks to [Artus](https://twitter.com/artuskg) for providing resources for training the Q2 model.
* Heads can have a different activation from FPN via config
* FPN resample (interpolation) can be specified via config and include any F.interpolation method or `max`/`avg` pool
* Default focal loss changed back to `new_focal`, use `--legacy-focal` arg to use the original. Legacy uses less memory, but has more numerical stability issues.
* custom augmentation transform and collate fn can be passed to loader factory
* `timm` >= 0.3.2 required, NOTE double check any custom defined model config for breaking change 
* PyTorch >= 1.6 now required

### 2020-11-12
* add experimental PAN and Quad FPN configs to the existing EfficientDet BiFPN w/ two test model configs
* switch untrained experimental model configs to use torchscript compat bn head layout by default

### 2020-11-09
* set model config to read-only after creation to reduce likelyhood of misuse
* no accessing model or bench .config attr in forward() call chain (for torcscript compat)
* numerous smaller changes that allow jit scripting of the model or train/predict bench

### 2020-10-30
Merged a few months of accumulated fixes and additions.
* Proper fine-tuning compatible model init (w/ changeable # classes and proper init, demoed in train.py)
* A new dataset interface with dataset support (via parser classes) for COCO, VOC 2007/2012, and OpenImages V5/Challenge2019
* New focal loss def w/ label smoothing available as an option, support for jit of loss fn for (potential) speedup
* Improved a few hot spots that squeek out a couple % of throughput gains, higher GPU utilization
* Pascal / OpenImages evaluators based on Tensorflow Models Evaluator framework (usable for other datasets as well)
* Support for native PyTorch DDP, SyncBN, and AMP in PyTorch >= 1.6. Still defaults to APEX if installed.
* Non-square input image sizes are allowed for the model (the anchor layout). Specified by image_size tuple in model config. Currently still restricted to `size % 128 = 0` on each dim.
* Allow anchor target generation to be done in either dataloader process' via collate or in model as in past. Can help balance compute.
* Filter out unused target cls/box from dataset annotations in fixed size batch tensors before passing to target assigner. Seems to speed convergence.
* Letterbox aware Random Erasing augmentation added.
* A (very slow) SoftNMS impl added for inference/validation use. It can be manually enabled right now, can add arg if demand.
* Tested with PyTorch 1.7
* Add ResDet50 model weights, 41.6 mAP.

A few things on priority list I haven't tackled yet:
* Mosaic augmentation
* bbox IOU loss (tried a bit but so far not a great result, need time to debug/improve)

**NOTE** There are some breaking changes:
* Predict and Train benches now output XYXY boxes, NOT XYWH as before. This was done to support other datasets as XYWH is COCO's evaluator requirement.
* The TF Models Evaluator operates on YXYX boxes like the models. Conversion from XYXY is currently done by default. Why don't I just keep everything YXYX? Because PyTorch GPU NMS operates in XYXY.
* You must update your version of `timm` to the latest (>=0.3), as some APIs for helpers changed a bit.

Training sanity checks were done on VOC and OI
  * 80.0 @ 50 mAP finetune on voc0712 with no attempt to tune params (roughly as per command below)
  * 18.0 mAP @ 50 for OI Challenge2019 after couple days of training (only 6 epochs, eek!). It's much bigger, and takes a LOONG time, many classes are quite challenging.


## Models

The table below contains models with pretrained weights. There are quite a number of other models that I have defined in [model configurations](effdet/config/model_config.py) that use various `timm` backbones.

| Variant                | mAP (val2017) | mAP (test-dev2017) | mAP (TF official val2017) | mAP (TF official test-dev2017) | Params (M) | Img Size |
|------------------------|:-------------:| :---: | :---: | :---: |:----------:|:--------:|
| tf_efficientdet_lite0  |     27.1      | TBD | 26.4 | N/A |    3.24    |   320    |
| tf_efficientdet_lite1  |     32.2      | TBD | 31.5 | N/A |    4.25    |   384    |
| efficientdet_d0        |     33.6      | TBD | N/A | N/A |    3.88    |   512    |
| tf_efficientdet_d0     |     34.2      | TBD | 34.3 | 34.6 |    3.88    |   512    |
| tf_efficientdet_d0_ap  |     34.8      | TBD | 35.2 | 35.3 |    3.88    |   512    |
| efficientdet_q0        |     35.7      | TBD | N/A | N/A |    4.13    |   512    |
| tf_efficientdet_lite2  |     35.9      | TBD | 35.1 | N/A |    5.25    |   448    |
| efficientdet_d1        |     39.4      | 39.5 | N/A | N/A |    6.62    |   640    |
| tf_efficientdet_lite3  |     39.6      | TBD | 38.8 | N/A |    8.35    |   512    |
| tf_efficientdet_d1     |     40.1      | TBD | 40.2 | 40.5 |    6.63    |   640    |
| tf_efficientdet_d1_ap  |     40.8      | TBD | 40.9 | 40.8 |    6.63    |   640    |
| efficientdet_q1        |     40.9      | TBD | N/A | N/A |    6.98    |   640    |
| cspresdext50pan        |     41.2      | TBD | N/A | N/A |    22.2    |   640    |
| resdet50               |     41.6      | TBD | N/A | N/A |    27.6    |   640    |
| efficientdet_q2        |     43.1      | TBD | N/A | N/A |    8.81    |   768    |
| cspresdet50            |     43.2      | TBD | N/A | N/A |    24.3    |   768    |
| tf_efficientdet_d2     |     43.4      | TBD | 42.5 | 43 |    8.10    |   768    |
| tf_efficientdet_lite3x |     43.6      | TBD | 42.6 | N/A |    9.28    |   640    |
| tf_efficientdet_lite4  |     44.2      | TBD | 43.2 | N/A |    15.1    |   640    |
| tf_efficientdet_d2_ap  |     44.2      | TBD | 44.3 | 44.3 |    8.10    |   768    |
| cspdarkdet53m          |     45.2      | TBD | N/A | N/A |    35.6    |   768    |
| efficientdetv2_dt      |     46.1      | TBD | N/A | N/A |    13.4    |   768    |
| tf_efficientdet_d3     |     47.1      | TBD | 47.2 | 47.5 |    12.0    |   896    |
| tf_efficientdet_d3_ap  |     47.7      | TBD | 48.0 | 47.7 |    12.0    |   896    |
| tf_efficientdet_d4     |     49.2      | TBD | 49.3 | 49.7 |    20.7    |   1024   |
| efficientdetv2_ds      |     50.1      | TBD | N/A | N/A |    26.6    |   1024   |
| tf_efficientdet_d4_ap  |     50.2      | TBD | 50.4 | 50.4 |    20.7    |   1024   |
| tf_efficientdet_d5     |     51.2      | TBD | 51.2 | 51.5 |    33.7    |   1280   |
| tf_efficientdet_d6     |     52.0      | TBD | 52.1 | 52.6 |    51.9    |   1280   |
| tf_efficientdet_d5_ap  |     52.1      | TBD | 52.2 | 52.5 |    33.7    |   1280   |
| tf_efficientdet_d7     |     53.1      | 53.4 | 53.4 | 53.7 |    51.9    |   1536   |
| tf_efficientdet_d7x    |     54.3      | TBD | 54.4 | 55.1 |    77.1    |   1536   |


See [model configurations](effdet/config/model_config.py) for model checkpoint urls and differences.

_NOTE: Official scores for all modules now using soft-nms, but still using normal NMS here._

_NOTE: In training some experimental models, I've noticed some potential issues with the combination of synchronized BatchNorm (`--sync-bn`) and model EMA weight everaging (`--model-ema`) during distributed training. The result is either a model that fails to converge, or appears to converge (training loss) but the eval loss (running BN stats) is garbage. I haven't observed this with EfficientNets, but have with some backbones like CspResNeXt, VoVNet, etc. Disabling either EMA or sync bn seems to eliminate the problem and result in good models. I have not fully characterized this issue._

## Environment Setup

Tested in a Python 3.7 - 3.9 conda environment in Linux with:
* PyTorch 1.6 - 1.10
* PyTorch Image Models (timm) >= 0.4.12, `pip install timm` or local install from (https://github.com/rwightman/pytorch-image-models)
* Apex AMP master (as of 2020-08). I recommend using native PyTorch AMP and DDP now.

*NOTE* - There is a conflict/bug with Numpy 1.18+ and pycocotools 2.0, force install numpy <= 1.17.5 or ensure you install pycocotools >= 2.0.2

## Dataset Setup and Use

### COCO
MSCOCO 2017 validation data:
```
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

MSCOCO 2017 test-dev data:
```
wget http://images.cocodataset.org/zips/test2017.zip
unzip -q test2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip
```

#### COCO Evaluation

Run validation (val2017 by default) with D2 model: `python validate.py /localtion/of/mscoco/ --model tf_efficientdet_d2`


Run test-dev2017: `python validate.py /localtion/of/mscoco/ --model tf_efficientdet_d2 --split testdev`

#### COCO Training

`./distributed_train.sh 4 /mscoco --model tf_efficientdet_d0 -b 16 --amp  --lr .09 --warmup-epochs 5  --sync-bn --opt fusedmomentum --model-ema`

NOTE:
* Training script currently defaults to a model that does NOT have redundant conv + BN bias layers like the official models, set correct flag when validating.
* I've only trained with img mean (`--fill-color mean`) as the background for crop/scale/aspect fill, the official repo uses black pixel (0) (`--fill-color 0`). Both likely work fine.
* The official training code uses EMA weight averaging by default, it's not clear there is a point in doing this with the cosine LR schedule, I find the non-EMA weights end up better than EMA in the last 10-20% of training epochs 
* The default h-params is a very close to unstable (exploding loss), don't try using Nesterov momentum. Try to keep the batch size up, use sync-bn.


### Pascal VOC

2007, 2012, and combined 2007 + 2012 w/ labeled 2007 test for validation are supported.

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
find . -name '*.tar' -exec tar xf {} \;
```

There should be a `VOC2007` and `VOC2012` folder within `VOCdevkit`, dataset root for cmd line will be VOCdevkit.

Alternative download links, slower but up more often than ox.ac.uk:
```
http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
```

#### VOC Evaluation

Evaluate on VOC2012 validation set:
`python validate.py /data/VOCdevkit --model efficientdet_d0 --num-gpu 2 --dataset voc2007 --checkpoint mychekpoint.pth --num-classes 20`

#### VOC Training

Fine tune COCO pretrained weights to VOC 2007 + 2012:
`/distributed_train.sh 4 /data/VOCdevkit --model efficientdet_d0 --dataset voc0712 -b 16 --amp --lr .008 --sync-bn --opt fusedmomentum --warmup-epochs 3 --model-ema --model-ema-decay 0.9966 --epochs 150 --num-classes 20 --pretrained`

### OpenImages

Setting up OpenImages dataset is a commitment. I've tried to make it a bit easier wrt to the annotations, but grabbing the dataset is still going to take some time. It will take approx 560GB of storage space.

To download the image data, I prefer the CVDF packaging. The main OpenImages dataset page, annotations, dataset license info can be found at: https://storage.googleapis.com/openimages/web/index.html

#### CVDF Images Download

Follow the s3 download directions here: https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations

Each `train_<x>.tar.gz` should be extracted to `train/<x>` folder, where x is a hex digit from 0-F. `validation.tar.gz` can be extracted as flat files into `validation/`.

#### Annotations Download

Annotations can be downloaded separately from the OpenImages home page above. For convenience, I've packaged them all together with some additional 'info' csv files that contain ids and stats for all image files. My datasets rely on the `<set>-info.csv` files. Please see https://storage.googleapis.com/openimages/web/factsfigures.html for the License of these annotations. The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license.
```
wget https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1-anno/openimages-annotations.tar.bz2
wget https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1-anno/openimages-annotations-challenge-2019.tar.bz2
find . -name '*.tar.bz2' -exec tar xf {} \;
```

#### Layout

Once everything is downloaded and extracted the root of your openimages data folder should contain:
```
annotations/<csv anno for openimages v5/v6>
annotations/challenge-2019/<csv anno for challenge2019>
train/0/<all the image files starting with '0'>
.
.
.
train/f/<all the image files starting with 'f'>
validation/<all the image files in same folder>
```

#### OpenImages Training
Training with Challenge2019 annotations (500 classes):
`./distributed_train.sh 4 /data/openimages --model efficientdet_d0 --dataset openimages-challenge2019 -b 7 --amp --lr .042 --sync-bn --opt fusedmomentum --warmup-epochs 1 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.999966 --epochs 100 --remode pixel --reprob 0.15 --recount 4 --num-classes 500 --val-skip 2`

The 500 (Challenge2019) or 601 (V5/V6) class head for OI takes up a LOT more GPU memory vs COCO. You'll likely need to half batch sizes.

### Examples of Training / Fine-Tuning on Custom Datasets

The models here have been used with custom training routines and datasets with great results. There are lots of details to figure out so please don't file any 'I get crap results on my custom dataset issues'. If you can illustrate a reproducible problem on a public, non-proprietary, downloadable dataset, with public github fork of this repo including working dataset/parser implementations, I MAY have time to take a look.

Examples:
* Chris Hughes has put together a great example of training w/ `timm` EfficientNetV2 backbones and the latest versions of the EfficientDet models here
  * [Medium blog post](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f)
  * [Python notebook](https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7)
* Alex Shonenkov has a clear and concise Kaggle kernel which illustrates fine-tuning these models for detecting wheat heads: https://www.kaggle.com/shonenkov/training-efficientdet (NOTE: this is out of date wrt to latest versions here, many details have changed)

If you have a good example script or kernel training these models with a different dataset, feel free to notify me for inclusion here...

## Results

### My Training

#### EfficientDet-D0

Latest training run with .336 for D0 (on 4x 1080ti):
`./distributed_train.sh 4 /mscoco --model efficientdet_d0 -b 22 --amp --lr .12 --sync-bn --opt fusedmomentum --warmup-epochs 5 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.9999`

These hparams above resulted in a good model, a few points:
* the mAP peaked very early (epoch 200 of 300) and then appeared to overfit, so likely still room for improvement
* I enabled my experimental LR noise which tends to work well with EMA enabled
* the effective LR is a bit higher than official. Official is .08 for batch 64, this works out to .0872
* drop_path (aka survival_prob / drop_connect) rate of 0.1, which is higher than the suggested 0.0 for D0 in official, but lower than the 0.2 for the other models
* longer EMA period than default

VAL2017
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336251
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.521584
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356439
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.123988
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521695
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.287121
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.441450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467914
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197697
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.552515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689297
```

#### EfficientDet-D1 

Latest run with .394 mAP (on 4x 1080ti):
`./distributed_train.sh 4 /mscoco --model efficientdet_d1 -b 10 --amp --lr .06 --sync-bn --opt fusedmomentum --warmup-epochs 5 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.99995`

For this run I used some improved augmentations, still experimenting so not ready for release, should work well without them but will likely start overfitting a bit sooner and possibly end up a in the .385-.39 range.


### Ported Tensorflow weights

#### TEST-DEV2017

NOTE: I've only tried submitting D7 to dev server for sanity check so far

##### TF-EfficientDet-D7
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.726
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.577
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.569
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
 ```

#### VAL2017

##### TF-EfficientDet-D0
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341877
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525112
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.360218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.131366
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399686
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.537368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293137
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.447829
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.472954
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.195282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558127
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.695312
```

##### TF-EfficientDet-D1
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401070
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.590625
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.422998
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211116
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459650
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.577114
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507095
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308963
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.610450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731814
```

##### TF-EfficientDet-D2
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434042
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.627834
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.463488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237414
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.486118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.606151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.343016
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.538328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.571489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638884
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.746671
```

##### TF EfficientDet-D3

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471223
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.661550
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.505127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.301385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.518339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.626571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365186
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.582691
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.617252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.670761
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779611
```

##### TF-EfficientDet-D4
 ```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.491759
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.686005
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.527791
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325658
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.536508
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.635309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.373752
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.601733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.638343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.463057
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.685103
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.789180
```

##### TF-EfficientDet-D5
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.511767
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.704835
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.552920
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.355680
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650184
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.384516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.619196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.657445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.499319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.695617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788889
```

##### TF-EfficientDet-D6
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520200
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.713204
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.560973
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.361596
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567414
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.657173
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.387733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.629269
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.667495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.499002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.711909
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.802336
```

##### TF-EfficientDet-D7
 ```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.531256
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.724700
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.571787
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.368872
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573938
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668253
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.393620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.637601
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.676987
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.524850
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.717553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.806352
 ```

##### TF-EfficientDet-D7X

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.737
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.823
```

## TODO
- [x] Basic Training (object detection) reimplementation
- [ ] Mosaic Augmentation
- [ ] Rand/AutoAugment
- [ ] BBOX IoU loss (giou, diou, ciou, etc)
- [ ] Training (semantic segmentation) experiments
- [ ] Integration with Detectron2 / MMDetection codebases
- [ ] Addition and cleanup of EfficientNet based U-Net and DeepLab segmentation models that I've used in past projects
- [x] Addition and cleanup of OpenImages dataset/training support from a past project
- [ ] Exploration of instance segmentation possibilities...

If you are an organization is interested in sponsoring and any of this work, or prioritization of the possible future directions interests you, feel free to contact me (issue, LinkedIn, Twitter, hello at rwightman dot com). I will setup a github sponser if there is any interest.
