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
  * Currently this is includes to all models implemented by the EficientNet and MobileNetv3 classes (which also includes MNasNet, MobileNetV2, MixNet and more). More soon...


## Updates / Tasks
### 2020-09-03
* All models updated to latest checkpoints from TF original.
* Add experimental soft-nms code, must be manually enabled right now. It is REALLY slow, .1-.2 mAP increase.

### 2020-07-27
* Add updated TF ported weights for D3 model (better training) and model def and weights for new D7X model (54.3 val mAP)
* Fix Windows bug so it at least trains in non-distributed mode

### 2020-06-15
Add updated D7 weights from Tensorflow impl, 53.1 validation mAP here (53.4 in TF)

### 2020-06-14
New model results, I've trained a D1 model with some WIP augmentation enhancements (not commited), just squeaking by official weights.

EfficientDet-D1:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393798
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586831
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191880
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455586
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571316
```

Also, [Soyeb Nagori](https://github.com/soyebn) trained an EfficientDet-Lite0 config using this code and contributed the weights.
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.319861
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.500062
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.336777
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.111257
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.378062
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501938
```

Unlike the other tf_ prefixed models this is not ported from (as of yet unreleased) TF official model, but it used
TF ported weights from `timm` for the pretrained imagenet model as the backbone init, thus it uses SAME padding. 

### 2020-06-12

* Additional experimental model configs based on MobileNetV2, MobileNetV3, MixNet, EfficientNet-Lite. Requires
update to `timm==0.1.28` for string based activation factory.
* Redundant bias config handled more consistency, defaults to config unless overridden by arg

### 2020-06-04

Latest results in and training goal achieved. Slightly bested the TF model mAP results for D0 model.
This model uses:
* typical PyTorch symmetric padding (instead of TF compatible SAME)
* my PyTorch trained EfficientNet-B0 as the pretrained starting weights (from `timm`)
* BiFPN/Head layers without any redundant conv/BN bias layers (slightly fewer params 3877763 vs 3880067)

My latest D0 run:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336251
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.521584
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356439
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.123988
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521695
```

TF ported D0 weights:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335653
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516253
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.353884
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125278
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386957
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.528071
```

Pretrained weights added for this model `efficientdet_d0`  (Tensorflow port is `tf_efficientdet_d0`)

### 2020-05-27
* A D0 result in, started before last improvements: `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331`
* Another D0 and D1 running with the latest code.

### 2020-05-22 / 23
A bunch of changes:
* COCO eval per epoch for better selection of checkpoints while training, works with distributed
* optimizations to both train and inference that should see small throughput gains
* doing the above, attempted to torchscript the full training loss + anchor labeler but ran into problems so had to back out part way due messy hacks or weird AMP issues causing silent bad results. Hopefully in PyTorch 1.6 there will be less TS issues.
* updated results after clipping boxes, now pretty much exact match to official, even slightly better on a few models
* added model factory, pretrained download, cleanup model configs
* setup.py, pypi release

### 2020-05-04
Initial D1 training results in -- close but not quite there. Definitely in reach and better than any other non-official EfficientDet impl I've seen.

Biggest missing element is proper per-epoch mAP validation for better checkpoint selection (than loss based). I was resisting doing full COCO eval because it's so slow, but may throw that in for now...

D1: `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382`

Previous D0 result: `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.324`

### 2020-05-02
First decent MSCOCO training results (from scratch, w/ pretrained classification backbone weights as starting point). 32.4 mAP for D0. Working on improvements and D1 trials still running.

### 2020-04-15
Taking a pause on training, some high priority things came up. There are signs of life on the training branch, was working the basic augs before priority switch, loss fn appeared to be doing something sane with distributed training working, no proper eval yet, init not correct yet. I will get to it, with SOTA training config and good performance as the end goal (as with my EfficientNet work). 

### 2020-04-11
Cleanup post-processing. Less code and a five-fold throughput increase on the smaller models. D0 running > 130 img/s on a single 2080Ti, D1 > 130 img/s on dual 2080Ti up to D7 @ 8.5 img/s.

### 2020-04-10
Replace `generate_detections` with PyTorch impl using torchvision batched_nms. Significant performance increase with minor (+/-.001 mAP) score differences. Quite a bit faster than original TF impl on a GPU now.

### 2020-04-09
Initial code with working validation posted. Yes, it's a little slow, but I think faster than the official impl on a GPU if you leave AMP enabled. Post processing needs some love. 

### Core Tasks
- [x] Feature extraction from my EfficientNet implementations (https://github.com/rwightman/gen-efficientnet-pytorch or https://github.com/rwightman/pytorch-image-models)
- [x] Low level blocks / helpers (SeparableConv, create_pool2d (same padding), etc)
- [x] PyTorch implementation of BiFPN, BoxNet, ClassNet modules and related submodules
- [x] Port Tensorflow checkpoints to PyTorch -- initial D1 checkpoint converted, state_dict loaded, on to validation....
- [x] Basic MS COCO validation script
  - [x] Temporary (hacky) COCO dataset and transform 
  - [x] Port reference TF anchor and object detection code
  - [x] Verify model output sanity
  - [X] Integrate MSCOCO eval metric calcs
- [x] Some cleanup, testing
- [x] Submit to test-dev server, all good
- [x] pretrained URL based weight download
- [ ] Torch hub
- [x] Remove redundant bias layers that exist in the official impl and weights
- [ ] Add visualization support
- [x] Performance improvements, numpy TF detection code -> optimized PyTorch
- [ ] Verify/fix Torchscript and ONNX export compatibility
- [ ] Try PyTorch 1.6/1.7 w/ NHWC (channels last) order which matches TF impl

### Possible Future Tasks
- [x] Basic Training (object detection) reimplementation
- [ ] Advanced Training w/ Rand/AutoAugment, etc
- [ ] Training (semantic segmentation) experiments
- [ ] Integration with Detectron2 / MMDetection codebases
- [ ] Addition and cleanup of EfficientNet based U-Net and DeepLab segmentation models that I've used in past projects
- [ ] Addition and cleanup of OpenImages dataset/training support from a past project
- [ ] Exploration of instance segmentation possibilities...

If you are an organization is interested in sponsoring and any of this work, or prioritization of the possible future directions interests you, feel free to contact me (issue, LinkedIn, Twitter, hello at rwightman dot com). I will setup a github sponser if there is any interest.

## Models

| Variant | Download | mAP (val2017) | mAP (test-dev2017) | mAP (TF official val2017) | mAP (TF official test-dev2017) |
| --- | --- | :---: | :---: | :---: | :---: |
| lite0 | [tf_efficientdet_lite0.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite0-f5f303a9.pth) | 32.0 | TBD | N/A | N/A |
| D0 | [efficientdet_d0.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth) | 33.6 | TBD | 33.5 | 33.8 |
| D0 | [tf_efficientdet_d0.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-1851dfed.pth) | 34.2 | TBD | 34.3 | 34.6 |
| D1 | [efficientdet_d1.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth) | 39.4 | 39.5 | 39.1 | 39.6 |
| D1 | [tf_efficientdet_d1.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_40-a30f94af.pth) | 40.1 | TBD | 40.2 | 40.5 |
| D2 | [tf_efficientdet_d2.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_43-8107aa99.pth) | 43.4 | TBD | 42.5 | 43 |
| D3 | [tf_efficientdet_d3.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth) | 47.1 | TBD | 47.2 | 47.5 |
| D4 | [tf_efficientdet_d4.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_49-f56376d9.pth) | 49.2 | TBD | 49.3 | 49.7 |
| D5 | [tf_efficientdet_d5.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth) | 51.2 | TBD | 51.2 | 51.5 |
| D6 | [tf_efficientdet_d6.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6_52-4eda3773.pth) | 52.0 | TBD | 52.1 | 52.6 |
| D7 | [tf_efficientdet_d7.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth) | 53.1 | 53.4 | 53.4 | 53.7 |
| D7X | [tf_efficientdet_d7x.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7x-f390b87c.pth) | 54.3 | TBD | 54.4 | 55.1 |

_NOTE: Official scores for all modules now using soft-nms, but still using normal NMS here._

## Usage

### Environment Setup

Tested in a Python 3.7 or 3.8 conda environment in Linux with:
* PyTorch 1.4 or PyTorch 1.6 (I recommend avoiding PyTorch 1.5 due to some jit and argmax issues)
* PyTorch Image Models (timm) >= 0.1.28, `pip install timm` or local install from (https://github.com/rwightman/pytorch-image-models) 
* Apex AMP master (as of 2020-04)

*NOTE* - There is a conflict/bug with Numpy 1.18+ and pycocotools, force install numpy <= 1.17.5 or the coco eval will fail,
the validation script will still save the output JSON and that can be run through eval again later. 

### Dataset Setup

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

### Run COCO Evaluation

Run validation (val2017 by default) with D2 model: `python validation.py /localtion/of/mscoco/ --model tf_efficientdet_d2 --checkpoint tf_efficientdet_d2.pth`


Run test-dev2017: `python validation.py /localtion/of/mscoco/ --model tf_efficientdet_d2 --checkpoint tf_efficientdet_d2.pth --anno test-dev2017`

### Run Inference

TODO: Need an inference script

### Run Training

`./distributed_train.sh 2 /mscoco --model tf_efficientdet_d0 -b 16 --amp  --lr .04 --warmup-epochs 5  --sync-bn --opt fusedmomentum --fill-color mean --model-ema`

NOTE:
* Training script currently defaults to a model that does NOT have redundant conv + BN bias layers like the official models, set correct flag when validating.
* I've only trained with img mean (`--fill-color mean`) as the background for crop/scale/aspect fill, the official repo uses black pixel (0) (`--fill-color 0`). Both likely work fine.
* The official training code uses EMA weight averaging by default, it's not clear there is a point in doing this with the cosine LR schedule, I find the non-EMA weights end up better than EMA in the last 10-20% of training epochs 
* The default h-params is a very close to unstable (exploding loss), don't try using Nesterov momentum. Try to keep the batch size up, use sync-bn.

### Examples of Training / Fine-Tuning on Alternate Datasets

* Alex Shonenkov has a clear and concise Kaggle kernel which illustrates fine-tuning these models for detecting wheat heads: https://www.kaggle.com/shonenkov/training-efficientdet
* If you have a good example script or kernel training these models with a different dataset, feel free to notify me for inclusion here...

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