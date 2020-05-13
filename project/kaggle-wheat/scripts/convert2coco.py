import pandas as pd
from sklearn.model_selection import GroupKFold
import tqdm
import json
import numpy as np

"""
Convert to coco format.

'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'car'},
 ]
 """


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=NpEncoder)


def decode_bbox(bbox_str):
    return [float(v) for v in bbox_str.strip("[]").split(', ')]


def df2annotations(df):
    coco_annotations = {}
    coco_annotations['images'] = []
    coco_annotations['annotations'] = []
    coco_annotations['categories'] = [
        {"supercategory": "wheat", "id": 0, "name": "wheat"}]

    image_id = 1
    annot_id = 1
    ids = df.image_id.unique()
    for id_ in tqdm.tqdm(ids):
        coco_annotations['images'].append(
            {
                'file_name': f'{id_}.jpg',
                'height': 1024,
                'width': 1024,
                'id': image_id
            }
        )

        image_df = df.query('image_id==@id_')

        if 'bbox' in df.columns:
            for i, row in image_df.iterrows():
                bbox = decode_bbox(row.bbox)
                coco_annotations['annotations'].append(
                    {
                        'image_id': image_id,
                        'segmentation': [],
                        'iscrowd': 0,
                        'bbox': bbox,
                        'area': bbox[2] * bbox[3],
                        'category_id': 0,
                        'id': annot_id,
                    }
                )
                annot_id += 1

        image_id += 1
    return coco_annotations


def main():
    df = pd.read_csv('../data/train.csv')
    kf = GroupKFold(n_splits=5)
    for i, (_, val_idx) in enumerate(kf.split(df, groups=df.image_id)):
        df.loc[val_idx, 'fold'] = i

    for fold in range(5):
        trn_df = df.query('fold!=@fold')
        val_df = df.query('fold==@fold')

        trn_annotations = df2annotations(trn_df)
        val_annotations = df2annotations(val_df)

        save_json(trn_annotations,
                  f'../data/annotations/train_fold{fold}.json')
        save_json(val_annotations,
                  f'../data/annotations/valid_fold{fold}.json')

    test_df = pd.read_csv('../data/sample_submission.csv')
    test_annotations = df2annotations(test_df)
    save_json(test_annotations, '../data/annotations/test.json')


if __name__ == '__main__':
    main()
