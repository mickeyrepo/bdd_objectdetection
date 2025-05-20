import os
import json

def load_json(pathin):
    ''' Function to load json files and return the schema'''
    with open(pathin, 'r') as file:
        data = json.load(file)
    return data 

val = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
data = load_json(val)

categories = []
for i in range(len(data)):
     labels = data[i]['labels']
     for lab in labels:
        if 'box2d' in lab.keys():
             cat = lab['category']
             if cat not in categories:
                  categories.append(cat)

gt_files = 'groundtruth/'
if not os.path.exists(gt_files):
     os.makedirs(gt_files)

for imgs in data:
        labels = imgs["labels"]
        text = ''
        i=0
        name = imgs['name']
        for label in labels:
            if "box2d" in label.keys():
                bb = label['box2d']
                
                cls = categories.index(label['category'])
                if i==0:
                    temp = str(cls)+' '+str(bb['x1'])+' '+str(bb['y1'])+' '+str(bb['x2'])+' '+str(bb['y2'])
                    text+=temp
                    i+=1
                else:
                    temp = '\n'+str(cls)+' '+str(bb['x1'])+' '+str(bb['y1'])+' '+str(bb['x2'])+' '+str(bb['y2'])
                    text+=temp
        file_name = name.replace("jpg","txt")
        with open(gt_files+file_name,'w') as f:
            f.write(text)         