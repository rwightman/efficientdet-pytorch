import torch
import torch.distributed as dist
import abc
import json
from .distributed import synchronize, is_main_process, all_gather_container
from pycocotools.cocoeval import COCOeval


class Evaluator:

    def __init__(self):
        pass

    @abc.abstractmethod
    def add_predictions(self, output, target):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass


class COCOEvaluator(Evaluator):

    def __init__(self, coco_api, distributed=False):
        super().__init__()
        self.coco_api = coco_api
        self.distributed = distributed
        self.distributed_device = None
        self.img_ids = []
        self.predictions = []

    def reset(self):
        self.img_ids = []
        self.predictions = []

    def add_predictions(self, detections, target):
        if self.distributed:
            if self.distributed_device is None:
                # cache for use later to broadcast end metric
                self.distributed_device = detections.device
            synchronize()
            detections = all_gather_container(detections)
            #target = all_gather_container(target)
            sample_ids = all_gather_container(target['img_id'])
            if not is_main_process():
                return
        else:
            sample_ids = target['img_id']

        detections = detections.cpu()
        sample_ids = sample_ids.cpu()
        for index, sample in enumerate(detections):
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
                self.img_ids.append(image_id)
                self.predictions.append(coco_det)

    def evaluate(self):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            json.dump(self.predictions, open('./temp.json', 'w'), indent=4)
            results = self.coco_api.loadRes('./temp.json')
            coco_eval = COCOeval(self.coco_api, results, 'bbox')
            coco_eval.params.imgIds = self.img_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metric = coco_eval.stats[0]  # mAP 0.5-0.95
            if self.distributed:
                dist.broadcast(torch.tensor(metric, device=self.distributed_device), 0)
        else:
            metric = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(metric, 0)
            metric = metric.item()
        self.reset()
        return metric


class FastMapEvalluator(Evaluator):

    def __init__(self, distributed=False):
        super().__init__()
        self.distributed = distributed
        self.predictions = []

    def add_predictions(self, output, target):
        pass

    def evaluate(self):
        pass