""" Object detection loader/collate

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data
from .transforms import *
from .random_erasing import RandomErasing
from timm.data.distributed_sampler import OrderedDistributedSampler


MAX_NUM_INSTANCES = 100


class DetectionFastCollate:

    def __init__(self, instance_keys=None, instance_shapes=None, instance_fill=-1, max_instances=MAX_NUM_INSTANCES):
        instance_keys = instance_keys or {'bbox', 'bbox_ignore', 'cls'}
        instance_shapes = instance_shapes or dict(
            bbox=(max_instances, 4), bbox_ignore=(max_instances, 4), cls=(max_instances,))
        self.instance_info = {k: dict(fill=instance_fill, shape=instance_shapes[k]) for k in instance_keys}
        self.max_instances = max_instances

    def __call__(self, batch):
        batch_size = len(batch)
        target = dict()

        def _get_target(k, v):
            if k in target:
                return target[k], k in self.instance_info
            is_instance = False
            fill_value = 0
            if k in self.instance_info:
                info = self.instance_info[k]
                is_instance = True
                fill_value = info['fill']
                shape = (batch_size,) + info['shape']
                dtype = torch.float32
            elif isinstance(v, (tuple, list)):
                # per batch elem sequence
                shape = (batch_size, len(v))
                dtype = torch.float32 if isinstance(v[0], (float, np.floating)) else torch.int32
            else:
                # per batch elem scalar
                shape = batch_size,
                dtype = torch.float32 if isinstance(v, (float, np.floating)) else torch.int64
            target_tensor = torch.full(shape, fill_value, dtype=dtype)
            target[k] = target_tensor
            return target_tensor, is_instance

        img_tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            img_tensor[i] += torch.from_numpy(batch[i][0])
            for tk, tv in batch[i][1].items():
                target_tensor, is_instance = _get_target(tk, tv)
                if is_instance:
                    num_elem = min(tv.shape[0], self.max_instances)
                    target_tensor[i, 0:num_elem] = torch.from_numpy(tv[0:num_elem])
                else:
                    target_tensor[i] = torch.tensor(tv, dtype=target_tensor.dtype)

        return img_tensor, target


class PrefetchLoader:

    def __init__(self,
            loader,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            ):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        if re_prob > 0.:
            self.random_erasing = RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input = next_input.float().sub_(self.mean).div_(self.std)
                next_target = {k: v.cuda(non_blocking=True) for k, v in next_target.items()}
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input, next_target)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_train=False,
        use_prefetcher=True,
        re_prob=0.,
        re_mode='pixel',
        re_count=1,
        interpolation='bilinear',
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        pin_mem=False,
):
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_train:
        transform = transforms_coco_train(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            fill_color=fill_color,
            mean=mean,
            std=std)
    else:
        transform = transforms_coco_eval(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            fill_color=fill_color,
            mean=mean,
            std=std)

    dataset.transform = transform

    sampler = None
    if distributed:
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_mem,
        collate_fn=DetectionFastCollate() if use_prefetcher else torch.utils.data.dataloader.default_collate,
    )
    if use_prefetcher:
        if is_train:
            loader = PrefetchLoader(loader, mean=mean, std=std, re_prob=re_prob, re_mode=re_mode, re_count=re_count)
        else:
            loader = PrefetchLoader(loader, mean=mean, std=std)

    return loader
