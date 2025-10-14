import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset
import src.utils.transforms as T
import hydra


class COCODataset(Dataset):
    def __init__(self, root: str, annotation: str, numClass: int, scaling_thresholds: Dict[str, Tuple[float, float]] = None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.scaling_thresholds = scaling_thresholds
        self.transforms = T.Compose([
            T.ToTensor()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        image = Image.open(imgPath).convert('RGB')

        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])
        metadata = self.loadMetadata(imgID)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),}

        image, targets = self.transforms(image, targets)
        return image, metadata, targets


    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']
            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)


    def loadMetadata(self, imgID: int) -> np.ndarray:
        imgInfo = self.coco.imgs[imgID]
        metadata = imgInfo['meta']
        timestamp = imgInfo['date_captured']
        metadata['Hour'] = int(timestamp.split('T')[1].split(':')[0])
        metadata['Month'] = int(timestamp.split('T')[0].split('-')[1])
        metadata = self.scale_harborfront_metadata(metadata)
        metadata = torch.as_tensor(list(metadata.values()))
        return metadata

    def scale_harborfront_metadata(self, metadata: Dict[str, float]) -> Dict[str, float]:
        metadata = metadata.copy()
        for key, (min_val, max_val) in self.scaling_thresholds.items():
            metadata[key] = (metadata[key] - min_val) / (max_val - min_val)
            metadata[key] = (metadata[key] - 0.5) * 4 # shift the range to [-2, 2]
        return metadata

class YOLODataset(Dataset):
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, metadata, target = self.coco_dataset[idx]  # here image is a tensor (C, H, W) and target is a dict with 'boxes' and 'labels'
        boxes = target['boxes'] # (num_boxes, 4) in (x, y, w, h) format normalized
        labels = target['labels']
        yolo_labels = []

        for label, box in zip(labels, boxes):

            x, y, w, h = box.tolist()
            yolo_labels.append([
                int(label),   # class label
                x + w / 2,     # x_center (normalized or not depending on dataset)
                y + h / 2,     # y_center
                w,    # width
                h    # height
            ])

        if yolo_labels:
            target_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
        else:
            target_tensor = torch.empty((0, 5), dtype=torch.float32) # Empty tensor if no valid boxes

        return image, target_tensor, target

# MARK: - collate functions for dataloaders
def collate_fn_coco(batch: List[Tuple[Tensor, Tensor, dict]]) -> Tuple[Tensor, Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), torch.stack(batch[1]), batch[2]

def _process_target_dict(target: dict) -> Dict[str, Tensor]:
    """Convert target dictionary values to appropriate tensor types."""
    tensor_target = {}
    for k, v in target.items():
        if isinstance(v, np.ndarray):
            if k == 'labels':
                tensor_target[k] = torch.from_numpy(v).long()
            elif k == 'masks':
                tensor_target[k] = torch.from_numpy(v).bool()
            else:
                tensor_target[k] = torch.from_numpy(v).float()
        elif isinstance(v, Tensor):
            if k == 'labels':
                tensor_target[k] = v.long()
            elif k == 'masks':
                tensor_target[k] = v.bool()
            else:
                tensor_target[k] = v.float()
        else:
            if k == 'labels':
                tensor_target[k] = torch.tensor(v, dtype=torch.long)
            elif k == 'masks':
                tensor_target[k] = torch.tensor(v, dtype=torch.bool)
            else:
                tensor_target[k] = torch.tensor(v, dtype=torch.float32)
    return tensor_target


def collate_fn_yolo(batch: List[Tuple[Tensor, Tensor, dict]]) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
    images, targets, original_targets = zip(*batch)
    images_tensor = torch.stack([img.float() for img in images], dim=0)
    original_targets_list = [_process_target_dict(target) for target in original_targets]
    return images_tensor, targets, original_targets_list


def load_datasets(args):
    num_classes = args.numClass
    data_folder = args.dataDir
    train_file = args.trainAnnFile
    val_file = args.valAnnFile
    test_file = args.testAnnFile

    data_folder = os.path.join(args.currentDir, data_folder)
    train_file = os.path.join(args.currentDir, train_file)
    val_file = os.path.join(args.currentDir, val_file)
    test_file = os.path.join(args.currentDir, test_file)

    train_dataset = COCODataset(data_folder, train_file, num_classes, args.scaleMetadata)
    val_dataset = COCODataset(data_folder, val_file, num_classes, args.scaleMetadata)
    test_dataset = COCODataset(data_folder, test_file, num_classes, args.scaleMetadata)
    collate_fn = collate_fn_coco
    if args.model == 'yolo':
        train_dataset = YOLODataset(train_dataset)
        val_dataset = YOLODataset(val_dataset)
        test_dataset = YOLODataset(test_dataset)
        collate_fn = collate_fn_yolo

    return train_dataset, val_dataset, test_dataset, collate_fn

# TEST MAIN
@hydra.main(config_path='../../config', config_name='config', version_base="1.3")
def main(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader

    num_classes = 4
    data_folder = 'data'
    data_file = 'data/Test.json'
    data_folder = os.path.join(args.currentDir, data_folder)
    data_file = os.path.join(args.currentDir, data_file)
    dataset = COCODataset(data_folder, data_file, num_classes, args.scaleMetadata)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_coco)
    print(dataset.__len__())

    yolo_dataset = YOLODataset(dataset)


    
    for i in range(dataset.__len__()):
        image, meta, target = dataset.__getitem__(i)
        yolo_image, yolo_target, original_target = yolo_dataset.__getitem__(i)
        print(image.size(), meta.size(), target)
        print(yolo_image.size(), yolo_target.size())

        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))

        for i in range(len(target['boxes'])):
            img_w, img_h = image.size(2), image.size(1)
            print(target['boxes'][i], img_w, img_h)
            x, y, w, h = target['boxes'][i]
            x, y = x*img_w, y*img_h 
            w, h = w*img_w, h*img_h
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        for i in range(len(yolo_target)):
            x_center, y_center = yolo_target[i][1], yolo_target[i][2]
            yolo_w, yolo_h = yolo_target[i][3], yolo_target[i][4]
            yolo_img_w, yolo_img_h = yolo_image.size(2), yolo_image.size(1) 
            x_center, y_center = x_center*yolo_img_w, y_center*yolo_img_h
            yolo_w, yolo_h = yolo_w*yolo_img_w, yolo_h*yolo_img_h
            x, y = x_center, y_center
            rect = patches.Rectangle((x,y), yolo_w,yolo_h, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        plt.show()
    
if __name__ == '__main__':
    main()