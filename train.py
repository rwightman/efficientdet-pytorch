#!/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
from datetime import datetime

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

from effdet import get_efficientdet_config, EfficientDet, DetBenchEval, DetBenchTrain

from data import CocoDetection, create_loader

from timm.data import FastCollateMixup, mixup_batch
from timm.models import resume_checkpoint, load_checkpoint
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import torch
import torch.nn as nn
import torchvision.utils

torch.backends.cudnn.benchmark = True


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='tf_efficientdet_d1', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--redundant-bias', action='store_true', default=False,
                    help='include redundant bias layers if True, need True to match official TF weights')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='0', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')

# Optimizer parameters
parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='loss', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "loss"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def _unwrap_bench(model):
    # FIXME push into CheckpointSaver or come up with cleaner support bench <-> model relationship
    if isinstance(model, ModelEma):  # unwrap ModelEma
        return _unwrap_bench(model.ema)
    elif hasattr(model, 'module'):  # unwrap DDP
        return _unwrap_bench(model.module)
    elif hasattr(model, 'model'):  # unwrap Bench -> model
        return _unwrap_bench(model.model)
    else:
        return model


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on 1 GPU.')

    torch.manual_seed(args.seed + args.rank)

    # create model
    config = get_efficientdet_config(args.model)
    config.redundant_bias = args.redundant_bias  # redundant conv + BN bias layers (True to match official models)
    model = EfficientDet(config)
    model = DetBenchTrain(model, config)

    # FIXME create model factory, pretrained zoo
    # model = create_model(
    #     args.model,
    #     pretrained=args.pretrained,
    #     num_classes=args.num_classes,
    #     drop_rate=args.drop,
    #     drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=args.drop_block,
    #     global_pool=args.gp,
    #     bn_tf=args.bn_tf,
    #     bn_momentum=args.bn_momentum,
    #     bn_eps=args.bn_eps,
    #     checkpoint_path=args.initial_checkpoint)

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()
    optimizer = create_optimizer(args, model)
    use_amp = False
    if has_apex and args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        use_amp = True
    if args.local_rank == 0:
        logging.info('NVIDIA APEX {}. AMP {}.'.format(
            'installed' if has_apex else 'not installed', 'on' if use_amp else 'off'))

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(_unwrap_bench(model), args.resume)
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')
            amp.load_state_dict(resume_state['amp'])
    del resume_state

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '')
            #resume=args.resume)  # FIXME bit of a mess with bench
        if args.resume:
            load_checkpoint(_unwrap_bench(model_ema), args.resume, use_ema=True)

    if args.distributed:
        if args.sync_bn:
            try:
                if has_apex:
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            model = DDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    train_anno_set = 'train2017'
    train_annotation_path = os.path.join(args.data, 'annotations', f'instances_{train_anno_set}.json')
    train_image_dir = train_anno_set
    dataset_train = CocoDetection(os.path.join(args.data, train_image_dir), train_annotation_path)

    # FIXME cutmix/mixup worth investigating?
    # collate_fn = None
    # if args.prefetcher and args.mixup > 0:
    #     collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)

    loader_train = create_loader(
        dataset_train,
        input_size=config.image_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        #re_prob=args.reprob,  # FIXME add back various augmentations
        #re_mode=args.remode,
        #re_count=args.recount,
        #re_split=args.resplit,
        #color_jitter=args.color_jitter,
        #auto_augment=args.aa,
        interpolation=args.train_interpolation,
        #mean=data_config['mean'],
        #std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        #collate_fn=collate_fn,
        pin_mem=args.pin_mem,
    )

    train_anno_set = 'val2017'
    train_annotation_path = os.path.join(args.data, 'annotations', f'instances_{train_anno_set}.json')
    train_image_dir = train_anno_set
    dataset_eval = CocoDetection(os.path.join(args.data, train_image_dir), train_annotation_path)

    loader_eval = create_loader(
        dataset_eval,
        input_size=config.image_size,
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        #mean=data_config['mean'],
        #std=data_config['std'],
        num_workers=args.workers,
        #distributed=args.distributed,
        pin_mem=args.pin_mem,
    )

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir, use_amp=use_amp, model_ema=model_ema)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, args)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(model_ema.ema, loader_eval, args, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    _unwrap_bench(model), optimizer, args,
                    epoch=epoch, model_ema=_unwrap_bench(model_ema), metric=save_metric, use_amp=use_amp)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, model, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None):

    if args.prefetcher and args.mixup > 0 and loader.mixup_enabled:
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            loader.mixup_enabled = False

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        loss = model(input, target['bbox'], target['cls'])
        loss, class_loss, box_loss = loss

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_grad)
        else:
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                _unwrap_bench(model), optimizer, args, epoch, model_ema=_unwrap_bench(model_ema),
                use_amp=use_amp, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            loss = model(input, target['bbox'], target['cls'])
            loss, class_loss, box_loss = loss

            # FIXME do mAP validation

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))

    metrics = OrderedDict([('loss', losses_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
