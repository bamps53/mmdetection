import mmcv
from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class WheatDataset(CocoDataset):

    CLASSES = [
        'wheat',
        ]