from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import matplotlib.pyplot as plt
from src.datasets.dataset import load_datasets
import os
import hydra
import torch
import json


@hydra.main(config_path='config', config_name='config', version_base="1.3")   
def main(args):
    """
    Generate predictions for RTIOD challenge submission.
    Uses trained YOLO11x model at 640px resolution.
    """
    
    # Detect GPU for faster inference
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n🔍 Running predictions on: {device}")
    
    model = YOLO(args.modelCheckpoint)
    train, val, test, collate_fn = load_datasets(args)

    if args.submission.type == 'val':
        templatePath = args.submission.valTemplate
        dataset = val
    elif args.submission.type == 'test':
        templatePath = args.submission.testTemplate
        dataset = test

    with open(templatePath, 'r') as f:
        submission = json.load(f)
        img_ids = [int(key) for key in list(submission.keys())]

    for i in range(len(dataset)):
        print(f'\n\nProcessing image {i+1}/{len(dataset)}')
        imgPath = dataset.get_img_path(i)
        image = imgPath
        results = model.predict(source=image, imgsz=args.imgSize, conf=0.25, device=device)
        #print(results)
        result = results[0]
        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        labels = result.boxes.cls
        # convert labels to int and sum 1 to match the original labels
        labels = labels.int() + 1

        # convert everything to python lists
        boxes = boxes.cpu().numpy().tolist()
        conf = conf.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()

        # print(boxes)
        # print(conf)
        # print(labels)

        img_id_str = str(dataset.ids[i])
        if img_id_str not in submission:
            print(f"Warning: submission template missing id {img_id_str}; creating entry")
            continue
        submission[img_id_str]['boxes'] = boxes
        submission[img_id_str]['scores'] = conf
        submission[img_id_str]['labels'] = labels
    
    # save new json file
    os.makedirs('submissions', exist_ok=True)
    with open(f'submissions/predictions.json', 'w') as f:
        json.dump(submission, f, indent=2)

if __name__ == "__main__":
    main()
