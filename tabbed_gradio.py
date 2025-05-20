import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import json
from utils import check_dist,check_size,plot_dist,plot_box,class_stats
import os
import pandas as pd
import cv2

def load_json(pathin):
    ''' Function to load json files and return the schema'''
    with open(pathin, 'r') as file:
        data = json.load(file)
    return data 

def draw_boc(bb,img):
    '''
    Function to draw rectangle bounding box on an image.

    Input:
        bb (dict): A dictionary containing the bounding box coordinates ('x1', 'y1', 'x2', 'y2').
        img (numpy.ndarray): The image array.

    Output:
        numpy.ndarray: The image with the bounding box drawn on it.
    '''

    x1=int(bb['x1'])
    y1 =int(bb['y1'])
    x2 = int(bb['x2'])
    y2 = int(bb['y2'])
    im = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
    return im
    

def generate_plot(path_name,column_choice):
    '''
    Gradio function to generate an image with bounding boxes for a unique sample.

    Input:
        name (str): Name of the category (label).
        path_name (str): Path of the annotation JSON file (e.g., 'train.json' or 'val.json').
        column_choice (str): Radio button output indicating whether to show the image
                             with the minimum ('minArea') or maximum ('maxArea') bounding box area
                             for the selected label.

    Output:
        str: Filepath of the generated image with bounding boxes.
    '''

    data = load_json(path_name)
    plots = 'dist_plots/'
    if not os.path.exists(plots):
        os.makedirs(plots)
    if column_choice == "area":         
        sd,area=check_size(data)
        #value = [sd[key]['mean'] for key in sd.keys()]
        categories = area.keys() 
        value = [area[key] for key in categories]
        plot = plot_box(value,categories,"BBOX Area mean Distribution",plots+'area.png')        
    else:
        classes = check_dist(data)
        value = [classes[key] for key in classes.keys()]
        categories = classes.keys()
        plot = plot_dist(value,categories,"Class distribution Distribution",plots+'class.png')   
    return plot

def generate_image(name,path_name,column_choice):
    '''Gradio funtion to generate the unique image
        Input: Name of the category 
               path of the annotation json 
               radio button output
        Output: bbix impainted image'''
    data = load_json(path_name)
    class_unique = class_stats(data)
    root = '/app/bdd100k_images_100k/bdd100k/images/100k/'
    cat = path_name.split('/')[-1].split('_')[-1].split('.')[0]
    print("category",cat)
    img = cv2.imread(root+cat+'/'+class_unique[name][column_choice])
    for imgs in data:
        iname = imgs['name']
        if iname == class_unique[name][column_choice]:
            labels = imgs["labels"]
            for label in labels:
                if label['category']==name:
                    box = label['box2d']
                    anno = draw_boc(box,img)
    cv2.imwrite('dist_plots/temp.png',anno)


    return 'dist_plots/temp.png'
# Gradio tab interface for data statistics 
iface1 = gr.Interface(
        fn=generate_plot,
        inputs=[
        gr.Textbox(label="File Path"),
        gr.Radio(["class Histogram", "area distribution"], label="Select Column")
    ],
    outputs=gr.Image(type="filepath", label="Distribution")
)

# Gradio tab interface for finding unique samples
iface2 = gr.Interface(
        fn=generate_image,
        inputs=[
        gr.Textbox(label="Label Name"),
        gr.Textbox(label="Train or Val"),
        gr.Radio(["minArea", "maxArea"], label="Select Column")
    ],
    outputs=gr.Image(type="filepath", label="Distribution")
)

# Create a tabbed interface
tabbed = gr.TabbedInterface([iface1, iface2], ["class level", "Sample level"])

# Launch the tabbed interface on local host
tabbed.launch(server_name="0.0.0.0")