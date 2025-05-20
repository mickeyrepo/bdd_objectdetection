import json

def get_list(data):
    files = []
    for img in data:
        files.append(img['name'])
    return files

def load_json(pathtrain):
    with open(pathtrain, 'r') as file:
        data = json.load(file)
    return data

def check_dup(tr,vl):
    '''
    Checks for duplicate filenames between two lists of filenames.

    Args:
        tr (list): The first list of filenames (e.g., train set).
        vl (list): The second list of filenames (e.g., validation set).

    Returns:
        list: A list of filenames that are present in both input lists.
    '''
    dups = []
    for files in tr:
        if files in vl:
            dups.append(files)
    return dups

def remove_dup(dups,data):
    '''
    Removes entries from the dataset based on a list of duplicate filenames.

    Args:
        dups (list): A list of filenames to be removed.
        data (list): The original dataset (list of dictionaries).

    Returns:
        list: A new list containing the entries from the original data
              whose filenames are not in the list of duplicates.
    '''
    rev = []
    for i in range(len(data)):
        if data[i]['name'] not in dups:
            rev.append(data[i])            
    return rev
    

# Define the paths to the training and validation JSON files
pathtrain = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
pathval = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
# Define the path for the revised training JSON file (after removing duplicates)
revisedtrain = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train_revised.json'

traindata = load_json(pathtrain)
trainfiles = get_list(traindata)

valdata = load_json(pathval)
valfiles = get_list(valdata)

dups = check_dup(trainfiles,valfiles)
rev = remove_dup(dups,traindata)

jo = json.dumps(rev, indent=4)
with open(revisedtrain, "w") as outfile:
    outfile.write(jo)


