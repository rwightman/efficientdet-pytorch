#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import json
import time
import logging
import torch
import torch.nn.parallel
try:
    from apex import amp
    has_amp = True
except ImportError:
    has_amp = False


from collections import OrderedDict

from effdet import EfficientDet, get_efficientdet_config, DetBenchEval, load_checkpoint
from data import create_loader, CocoDetection
from utils import AverageMeter

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--anno', default='val2017',
                    help='mscoco annotation set (one of val2017, train2017, test-dev2017)')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
parser.add_argument('--no-redundant-bias', action='store_true', default=False,
                    help='remove redundant bias layers if True, need False for official TF weights')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='0', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    args.redundant_bias = not args.no_redundant_bias

    # create model
    config = get_efficientdet_config(args.model)
    config.redundant_bias = args.redundant_bias
    model = EfficientDet(config)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = DetBenchEval(model, config)
    bench = bench.cuda()
    if has_amp:
        print('Using AMP mixed precision.')
        bench = amp.initialize(bench, opt_level='O1')
    else:
        print('AMP not installed, running network in FP32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    if 'test' in args.anno:
        annotation_path = os.path.join(args.data, 'annotations', f'image_info_{args.anno}.json')
        image_dir = 'test2017'
    else:
        annotation_path = os.path.join(args.data, 'annotations', f'instances_{args.anno}.json')
        image_dir = args.anno
    dataset = CocoDetection(os.path.join(args.data, image_dir), annotation_path)

    loader = create_loader(
        dataset,
        input_size=config.image_size,
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        fill_color=args.fill_color,
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    img_ids = []
    results = []
    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            output = bench(input, target['scale'])
            output = output.cpu()
            sample_ids = target['img_id'].cpu()
            for index, sample in enumerate(output):
                image_id = int(sample_ids[index])
                for det in sample:
                    score = float(det[4])
                    if score < .001:  # stop when below this threshold, scores in descending order
                        break
                    coco_det = dict(
                        image_id=image_id,
                        bbox=det[0:4].tolist(),
                        score=score,
                        category_id=int(det[5]))
                    img_ids.append(image_id)
                    results.append(coco_det)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                     )
                )

    json.dump(results, open(args.results, 'w'), indent=4)
    if 'test' not in args.anno:
        coco_results = dataset.coco.loadRes(args.results)
        coco_eval = COCOeval(dataset.coco, coco_results, 'bbox')
        coco_eval.params.imgIds = img_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return results


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
