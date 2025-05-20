import numpy as np
from sklearn.metrics import average_precision_score
import os,glob
def preprocess_gt(idx, textfile):
    """
    Preprocesses ground truth annotations for a specific class index from a text file.

    Args:
        idx (int): The index (class ID) to filter the ground truth annotations.
        textfile (str): The path to the text file containing ground truth annotations.

    Returns:
        list: A list of ground truth bounding boxes for the specified class index.
              Each bounding box is represented as a list of four floats: [x1, y1, x2, y2].
    """
    ground_truth = []
    with open(textfile,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        items = line.split(' ')
        if items[0]==str(idx):
            temp = [float(items[1]),float(items[2]),float(items[3]),float(items[4])]
            ground_truth.append(temp)
    return ground_truth

def preprocess_yolox(idx, textfile):
    """
    Preprocesses YOLOX predictions for a specific class index from a text file.

    Args:
        idx (int): The index (class ID) to filter the predictions.
        textfile (str): The path to the text file containing YOLOX predictions.

    Returns:
        list: A list of predictions for the specified class index.
              Each prediction is represented as a list of five floats: [x1, y1, x2, y2, confidence].
    """
    predictions = []
    with open(textfile,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        items = line.split(' ')
        #print("--",len(items))
        if len(items)>1:
            
            if items[0]==str(idx):
                temp = [float(items[1]),float(items[2]),float(items[3]),float(items[4]),float(items[5])]
                predictions.append(temp)
    return predictions

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): The first bounding box as a list of four floats [x1, y1, x2, y2].
        box2 (list): The second bounding box as a list of four floats [x1, y1, x2, y2].

    Returns:
        float: The IoU value between the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area



# Function to compute mAP
def compute_map(detections, annotations, iou_threshold=0.5):
    """
    Computes the mean Average Precision (mAP) given a list of detections and ground truth annotations.

    Args:
        detections (list of lists): A list where each inner list contains detections for an image.
                                    Each detection is a list of [x1, y1, x2, y2, confidence].
        annotations (list of lists): A list where each inner list contains ground truth annotations for an image.
                                     Each annotation is a list of [x1, y1, x2, y2].
        iou_threshold (float, optional): The IoU threshold for considering a detection as a true positive.
                                         Defaults to 0.5.

    Returns:
        float: The mean Average Precision (mAP) value.
    """
    aps = []
    for det, ann in zip(detections, annotations):
        if len(ann) == 0:
            continue  # Skip images with no annotations

        tp = 0
        fp = 0
        used = [False] * len(ann)

        for d in det:
            matched = False
            for idx, a in enumerate(ann):
                if used[idx]:
                    continue  # Skip already matched ground truth
                
                iou = compute_iou(d[:4], a)
                if iou >= iou_threshold:
                    tp += 1
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                fp += 1  # False positive if no match

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(ann) if len(ann) > 0 else 0
        aps.append(precision * recall)

    return np.mean(aps) if len(aps) > 0 else 0

# Calculate mAP
# Example usage
# Path to the prediction files
preds_path = 'YOLOX_outputs/yolox_s/vis_res/2025_05_20_19_30_36/'
# path to the GT files
gt_path = 'groundtruth/'
map = []
for i,idx in enumerate([1,2,5,6,7]):
    clasMap = []
    for id, d in enumerate(glob.glob(os.path.join(preds_path,"*.txt"))):
        print(d)        
        pred_path_file = d
        file_name = d.split('/')[-1].split('\\')[-1]
        print(file_name)
        gt_path_file = os.path.join(gt_path,file_name)
        if os.path.exists(gt_path_file):            
            predictions = preprocess_yolox(idx,pred_path_file)        
            ground_truth = preprocess_gt(idx, gt_path_file)  
            mAP = compute_map([predictions], [ground_truth])
            clasMap.append(mAP)

        if id ==1000:
            break
    clasMap = np.asarray(clasMap,dtype=np.float32)
    map.append(np.average(clasMap))
print(map)