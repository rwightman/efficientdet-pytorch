# EfficientDet (PyTorch)

This is a work in progress PyTorch implementation of EfficientDet. It is based on the official Tensorflow implementation by [Mingxing Tan and the Google Brain team](https://github.com/google/automl).

I am aware there are some reasonable PyTorch implementations out there already. Their approach didn't fit well with my aim to replicate the Tensorflow models closely enough to allow weight ports while still maintaining a PyTorch feel and a high degree of flexibility for future additions. So, this is being done from scratch.

*NOTE* There is no code here at the moment aside from the skeleton model. Please do not create issues asking where the code/weights are. I just started getting into this last weekend (Sunday, March 21) after sitting at home for too long. The code will be pushed when it is working.

The initial objectives:
- [x] Feature extraction from my EfficientNet implementations (https://github.com/rwightman/gen-efficientnet-pytorch or https://github.com/rwightman/pytorch-image-models)
- [x] Low level blocks / helpers (SeparableConv, create_pool2d (same padding), etc)
- [x] PyTorch implementation of BiFPN, BoxNet, ClassNet modules and related submodules
- [x] Port Tensorflow checkpoints to PyTorch -- initial D1 checkpoint converted, state_dict loaded, on to validation....
- [x] Basic MS COCO validation script
  - [x] Temporary (hacky) COCO dataset and transform 
  - [x] Port reference TF anchor and object detection code
  - [x] Verify model output sanity (definitely signs of life with some valid predictions, has issues, likely alignment/padding/missed activation)
  - [ ] Integrate MSCOCO eval metric calcs
- [ ] Lots of cleanup, testing

Possible future directions:
- [ ] Training (object detection) reimplementation w/ Rand/AutoAugment, etc
- [ ] Training (semantic segmentation) experiments
- [ ] Integration with Detectron2 / MMDetection codebases
- [ ] Addition and cleanup of EfficientNet based U-Net and DeepLab segmentation models that I've used in past projects
- [ ] Addition and cleanup of OpenImages dataset/training support from a past project
- [ ] Exploration of instance segmentation possibilities...

Before the COVID-19 madness changed priorities I was looking into signing up for GitHub Sponsors. I've decided for now to focus on building. However, if you are an organization interested in sponsoring and any of this work, or prioritization of the possible future directions interests you, feel free to contact me (issue, LinkedIn, Twitter, hello at rwightman dot com)
