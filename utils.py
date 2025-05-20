import json
import numpy as np
import os
import imagehash
from PIL import Image
import matplotlib.pyplot as plt

def check_dist(data):
    '''
    Check the number of bounding boxes present per class.

    Args:
        data (list): A list of dictionaries, where each dictionary represents an image
                     and contains a 'labels' key with a list of label dictionaries.

    Returns:
        dict: A dictionary where keys are class categories and values are the frequency
              of bounding boxes for that class.
    '''
    class_freq = {}

    for imgs in data:
        labels = imgs["labels"]
        for label in labels:
            if "box2d" in label.keys():
                key = label["category"]
                if key not in class_freq.keys():
                    class_freq[key]=1
                else:
                    class_freq[key]=class_freq[key]+1                    
    return class_freq

def check_size(data):
    '''
    Check the Area of the bounding boxes per class and calculate mean and standard deviation.

    Args:
        data (list): A list of dictionaries, where each dictionary represents an image
                     and contains a 'labels' key with a list of label dictionaries.

    Returns:
        tuple: A tuple containing two dictionaries:
               - size_dist (dict): A dictionary where keys are class categories and values
                                   are dictionaries containing the mean and standard deviation
                                   of bounding box areas for that class.
               - area_list (dict): A dictionary where keys are class categories and values
                                   are lists of bounding box areas for that class.
    '''
    area_list = {}
    for imgs in data:
        labels = imgs["labels"]
        for label in labels:
            # Check if the label has a bounding box annotation
            if "box2d" in label.keys():
                key = label["category"]
                # Calculate the area of the bounding box
                area = abs((label["box2d"]["x2"]-label["box2d"]["x1"])*(label["box2d"]["y2"]-label["box2d"]["y1"]))
                if key not in area_list.keys():
                    area_list[key]=[]
                    area_list[key].append(area)
                else:
                    area_list[key].append(area)
    
    size_dist = {}
    # Calculate mean and standard deviation for each class's area
    for keys in area_list.keys():
        area = np.asarray(area_list[keys],dtype = np.float32)
        avg = np.mean(area)
        stdn = np.std(area)
        size_dist[keys] = {}
        size_dist[keys]['mean']=avg
        size_dist[keys]['stdn']=stdn           
                    
    return size_dist,area_list

def image2object(data):
    '''
    Check number of annotations in an image.

    Args:
        data (list): A list of dictionaries, where each dictionary represents an image
                     and contains a 'labels' key with a list of label dictionaries.

    Returns:
        numpy.ndarray: A numpy array where each element is the count of bounding box
                       annotations in the corresponding image.
    '''
    label_count= []
    for imgs in data:
        labels = imgs["labels"]
        count = 0
        for label in labels:
            if "box2d" in label.keys():
                count+=0        
        label_count.append(count)
    label_count = np.asarray(label,dtype = np.float32) 
    return label_count


def plot_dist(value,categories,distname,oploc):    
    '''
    Plot the distribution of class as a bar chart.

    Args:
        value (list): A list of values for each category.
        categories (list): A list of category names.
        distname (str): The title for the plot.
        oploc (str): The file path to save the plot.

    Returns:
        str: The file path where the plot was saved.
    '''

    plt.bar(categories, value, color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title(distname)     
    plt.savefig(oploc)
    plt.close()
    return oploc


def plot_box(value,categories,distname,oploc):    
    '''
    Box plot for area distribution.

    Args:
        value (list or array-like): The data to plot.
        categories (list): A list of category names for the box plots.
        distname (str): The title for the plot.
        oploc (str): The file path to save the plot.

    Returns:
        str: The file path where the plot was saved.
    '''
    plt.boxplot(value, labels=categories)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title(distname)     
    plt.savefig(oploc)
    plt.close()
    return oploc

def find_duplicate_images(directory):
    '''
    Find the unique images in a directory based on perceptual hash.

    Args:
        directory (str): The path to the directory containing images.

    Returns:
        list: A list of file paths to the unique images found in the directory.
    '''
    hashes = {}
    uniques = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    # Open the image and calculate perceptual hash
                    hash_value = imagehash.phash(img)
                    # If the hash is not already in the dictionary, it's a unique image
                    if hash_value not in hashes:                        
                        hashes[hash_value] = filepath
                        uniques.append(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return uniques



def class_stats(data):
    '''
    Class label area and aspect ratio distribution to isolate samples with largest and smallest area.

    Args:
        data (list): A list of dictionaries, where each dictionary represents an image
                     and contains a 'labels' key with a list of label dictionaries.

    Returns:
        dict: A dictionary where keys are class categories and values are dictionaries
              containing the filenames of images with the maximum and minimum bounding
              box areas for that class.
    '''
    cls_unique = {}
    class_unique = {}
    for imgs in data:
        labels = imgs["labels"]
        for label in labels:
            if "box2d" in label.keys():
                key = label["category"]
                area = abs((label["box2d"]["x2"]-label["box2d"]["x1"])*(label["box2d"]["y2"]-label["box2d"]["y1"]))
                ar = abs(label["box2d"]["x2"]-label["box2d"]["x1"])/abs(label["box2d"]["y2"]-label["box2d"]["y1"])
                if key not in  cls_unique.keys():
                    cls_unique[key] = []
                    templist = [area,ar,imgs['name']]
                    cls_unique[key].append(templist)
                else:
                    templist = [area,ar,imgs['name']]
                    cls_unique[key].append(templist)

    # Find images with max and min area for each class  
    for key in cls_unique.keys():  
        Marea = 0
        marea = float('inf')
        Mname = None
        mname = None
        for items in cls_unique[key]:            
            area = items[0]
            
            if area>Marea:
                Mname = items[2]
                Marea = area

            if area <=marea:
                mname = items[2]
                marea = area
                
        class_unique[key] = {}
        class_unique[key]['maxArea']=Mname  
        class_unique[key]['minArea']=mname
    return class_unique  
        




  


          



    
    



