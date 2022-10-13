# 내장
import os
import glob
import json
import random

# 서드파티
import imagesize


class SameDataset:
    def __init__(self):
        self._next_image_id = -1
        
    @property
    def next_image_id(self):
        self._next_image_id += 1
        return self._next_image_id


class Parser:
    def __init__(self,
                 same_dataset_indicator: SameDataset,
                 dataset_root_dir: str,
                 dataset_name: str,
                 dataset_url: str,
                 dataset_semantic_version: float = 1.0,
                 dataset_year: str = 'Unknown',
                 dataset_contributor: str = 'Unknown',
                 dataset_created: str = '0000/00/00',
                 dataset_license: str = 'Unknown',
                 dataset_license_url: str = 'Unknwon',
                 ) -> None:
        self.same_dataset_indicator = same_dataset_indicator
        self.dataset_root_dir = dataset_root_dir
        self.coco_json = {
            'info': {
                'description': f'{dataset_name}',
                'url': f'{dataset_url}',
                'version': f'{dataset_semantic_version}',
                'year': f'{dataset_year}',
                'contributor': f'{dataset_contributor}',
                'date_created': f'{dataset_created}'
            },
            'licenses': [{
                'url': f'{dataset_license_url}',
                'id': 1,
                'name': f'{dataset_license}'
            }],
            'categories': [],
            'images': [],
            'annotations': []
        }

    def dump(self):
        self.coco_json['categories'] = self.coco_json_categories
        self.coco_json['images'] = self.coco_json_images
        self.coco_json['annotations'] = self.coco_json_annotations
        with open(self.json_path, 'w') as f:
            json.dump(self.coco_json, f, indent=4)


class ClassificationParser(Parser):
    def __init__(self,
                 same_dataset_indicator: SameDataset,
                 dataset_root_dir: str,
                 output_dir: str,
                 output_json_name: str,
                 **kwargs) -> None:
        super().__init__(same_dataset_indicator, 
                         dataset_root_dir, 
                         **kwargs)
        self._output_dir = output_dir
        self.output_json_name = output_json_name
        if not '.json' in self.output_json_name:
            output_json_name = f'{output_json_name}.json'
        self.json_path = os.path.join(self._output_dir, output_json_name)


class ChildAdultCocoParser(ClassificationParser):
    def __init__(self, 
                 same_dataset_indicator: SameDataset,
                 dataset_root_dir: str, 
                 output_dir: str,
                 output_json_name: str,
                 adults_dir_from_root: str,
                 children_dir_from_root: str) -> None:
        super().__init__(same_dataset_indicator,
                         dataset_root_dir, output_dir, output_json_name,
                         dataset_name = 'ChildAdult',
                         dataset_url = 'https://www.kaggle.com/datasets/die9origephit/children-vs-adults-images?resource=download',
                         dataset_license = 'CC0: Public Domain',
                         dataset_license_url = 'https://creativecommons.org/publicdomain/zero/1.0/')
        self.adults_dir_from_root = adults_dir_from_root
        self.children_dir_from_root = children_dir_from_root
        self.adults_dir = os.path.join(self.dataset_root_dir, self.adults_dir_from_root)
        self.children_dir = os.path.join(self.dataset_root_dir, self.children_dir_from_root)
        self.coco_json_categories = [
            {
                'supercategory': 'person',
                'id': 0,
                'name': 'adult',
                'keypoints': [],
                'skeleton': [],
            },
            {
                'supercategory': 'person',
                'id': 1,
                'name': 'child',
                'keypoints': [],
                'skeleton': [],
            }
        ]
        
    def convert(self):
        self.coco_json_images = []
        self.coco_json_annotations = []
        dataset = {
            0: glob.glob(os.path.join(self.adults_dir, '*.jpg')),
            1: glob.glob(os.path.join(self.children_dir, '*.jpg')),
        }

        # 한번에 id 까지 작성해 넣지 않는 이유는 적절히 섞어 주기 위한 목적이다.
        image_annotation_pairs = []
        for category_id, path_li in dataset.items():
            for path in path_li:
                width, height = imagesize.get(path)
                image_annotation_pairs.append({
                    'image': {
                        'id': None,
                        'file_name': str(path),
                        'height': height,
                        'width': width,
                    },
                    'annotation': {
                        'segmentation': [[]],
                        'area': 0,
                        'iscrowd': 0,
                        'image_id': None,
                        'bbox': [],
                        'category_id': category_id,
                        'id': None,
                    }
                })
        
        random.shuffle(image_annotation_pairs)
        for image_annotation_pair in image_annotation_pairs:
            image = image_annotation_pair['image']
            annotation = image_annotation_pair['annotation']
            image_id = self.same_dataset_indicator.next_image_id
            # classification 문제에 대해서는 annotation_id와 image_id가 같다.
            image['id'] = image_id
            annotation['image_id'] = image_id
            annotation['id'] = image_id
            self.coco_json_images.append(image)
            self.coco_json_annotations.append(annotation)


if __name__ == '__main__':
    # 싸구려 코드이고 고칠 부분이 아주아주 많다.
    id_indexer = SameDataset()
    train_parser = ChildAdultCocoParser(
        id_indexer,
        dataset_root_dir = os.path.join('.', 'data', 'AdultChild'),
        output_dir=os.path.join('.', 'data', 'AdultChild'),
        output_json_name='train',
        adults_dir_from_root=os.path.join('train', 'adults'),
        children_dir_from_root=os.path.join('train', 'children'))
    test_parser = ChildAdultCocoParser(
        id_indexer,
        dataset_root_dir = os.path.join('.', 'data', 'AdultChild'),
        output_dir=os.path.join('.', 'data', 'AdultChild'),
        output_json_name='test',
        adults_dir_from_root=os.path.join('test', 'adults'),
        children_dir_from_root=os.path.join('test', 'children'))
    train_parser.convert()
    train_parser.dump()
    test_parser.convert()
    test_parser.dump()