import mmcv
from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class WheatDataset(CocoDataset):

    CLASSES = [
        'wheat',
        ]