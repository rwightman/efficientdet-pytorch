import os
import tqdm
import torch
try:
    from apex import amp
    has_amp = True
except ImportError:
    has_amp = False
from sotabencheval.object_detection import COCOEvaluator
from sotabencheval.utils import is_server, extract_archive
from effdet import create_model
from data import CocoDetection, create_loader

NUM_GPU = 1
BATCH_SIZE = (128 if has_amp else 64) * NUM_GPU
ANNO_SET = 'val2017'

if is_server():
    DATA_ROOT = './.data/vision/coco'
    image_dir_zip = os.path.join('./.data/vision/coco', f'{ANNO_SET}.zip')
    extract_archive(from_path=image_dir_zip, to_path='./.data/vision/coco')
else:
    # local settings
    DATA_ROOT = ''


def _bs(b=64):
    b *= NUM_GPU
    if has_amp:
        b *= 2
    return b


def _entry(model_name, paper_model_name, paper_arxiv_id, batch_size=BATCH_SIZE, model_desc=None):
    return dict(
        model_name=model_name,
        model_description=model_desc,
        paper_model_name=paper_model_name,
        paper_arxiv_id=paper_arxiv_id,
        batch_size=batch_size)

# NOTE For any original PyTorch models, I'll remove from this list when you add to sotabench to
# avoid overlap and confusion. Please contact me.
model_list = [
    ## Weights ported by myself from other frameworks
    _entry('tf_efficientdet_d0', 'EfficientDet-D0', '1911.09070', batch_size=_bs(112),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d1', 'EfficientDet-D1', '1911.09070', batch_size=_bs(72),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d2', 'EfficientDet-D2', '1911.09070', batch_size=_bs(48),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d3', 'EfficientDet-D3', '1911.09070', batch_size=_bs(32),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d4', 'EfficientDet-D4', '1911.09070', batch_size=_bs(16),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d5', 'EfficientDet-D5', '1911.09070', batch_size=_bs(12),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d6', 'EfficientDet-D6', '1911.09070', batch_size=_bs(8),
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d7', 'EfficientDet-D7', '1911.09070', batch_size=_bs(4),
           model_desc='Ported from official Google AI Tensorflow weights'),

    ## Weights trained by myself in PyTorch
    ## TODO
]


def eval_model(model_name, paper_model_name, paper_arxiv_id, batch_size=64, model_description=''):

    # create model
    bench = create_model(
        model_name,
        bench_task='predict',
        pretrained=True,
    )
    bench.eval()
    input_size = bench.config.image_size

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model_name, param_count))

    bench = bench.cuda()
    if has_amp:
        print('Using AMP mixed precision.')
        bench = amp.initialize(bench, opt_level='O1')
    else:
        print('AMP not installed, running network in FP32.')

    annotation_path = os.path.join(DATA_ROOT, 'annotations', f'instances_{ANNO_SET}.json')
    evaluator = COCOEvaluator(
        root=DATA_ROOT,
        model_name=paper_model_name,
        model_description=model_description,
        paper_arxiv_id=paper_arxiv_id)

    dataset = CocoDetection(os.path.join(DATA_ROOT, ANNO_SET), annotation_path)

    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=batch_size,
        use_prefetcher=True,
        fill_color='mean',
        num_workers=4,
        pin_mem=True)

    iterator = tqdm.tqdm(loader, desc="Evaluation", mininterval=5)
    evaluator.reset_time()

    with torch.no_grad():
        for i, (input, target) in enumerate(iterator):
            output = bench(input, target['img_scale'], target['img_size'])
            output = output.cpu()
            sample_ids = target['img_id'].cpu()
            results = []
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
                    results.append(coco_det)
            evaluator.add(results)

            if evaluator.cache_exists:
                break

    evaluator.save()


for m in model_list:
    eval_model(**m)
    torch.cuda.empty_cache()
