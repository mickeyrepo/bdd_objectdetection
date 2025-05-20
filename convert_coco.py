import json

tr_input = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
tr_output = 'train.json'
with open(tr_input, 'r') as file:
        data = json.load(file) 
ti = []
ta = [] 
tc = []



#Generate the class labels category
categories = []
for i in range(len(data)):
     labels = data[i]['labels']
     for lab in labels:
        if 'box2d' in lab.keys():
             cat = lab['category']
             # Add the category to the list if it's not already present
             if cat not in categories:
                  categories.append(cat)

for cat in categories:
    tempC = {}
    tempC['supercategory']= "none"    
    tempC['id']=categories.index(cat)
    tempC['name']=cat
    tc.append(tempC)
             
     
trainid = 0
# Process image data and annotations, taking every 10th image just to reduce the number or samples

for i in range(0,len(data),10):
    tempI = {}        
    tempI['id']=i        
    tempI['file_name']=data[i]['name']
    #Set a fixed width as per the YoloX requirements
    tempI['width']=640
    tempI['height']=640
    ti.append(tempI)

    
    labels  = data[i]['labels']
    #print(data[i]['name'])
    #print(len(labels))
    count = 0
    for j in range(0,len(labels)):
        lab = labels[j]
        tempA = {}
        if 'box2d' in lab.keys():
            tempA['iscrowd']=0
            tempA['ignore']=0
            tempA['image_id']=i  
            count+=1

            bb = lab['box2d']
            # Recalculate bounding box coordinates for the new image size (640x640)
            x = float(bb['x1'])*640/1280
            y = float(bb['y1'])*640/720
            w = abs(float(bb['x2'])-float(bb['x1']))*640/1280
            h = abs(float(bb['y2'])-float(bb['y1']))*640/720

            bbox = [x,y,w,h]              
            tempA['bbox'] = bbox

            tempA['area']=w*h
            tempA['segmentation']=[]
            tempA['category_id']= categories.index(lab['category'])
            tempA['id']=trainid
            
            #print(tempA['id'])
            trainid = trainid+1

            ta.append(tempA)    
# Structure the final output dictionary in COCO format    
t = {}
t['images']=ti
t['annotations']=ta
t['categories']=tc
jo = json.dumps(t, indent=4)
with open(tr_output, "w") as outfile:
    outfile.write(jo)
