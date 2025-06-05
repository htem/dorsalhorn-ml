import argparse
import caveclient
import cloudvolume
import datetime
import glob
import json
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
import yaml
sys.path.append('../')
from embed.utils import * # custom library with util functions
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Read configuration file (config.yaml)
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Get the current date and time
now = datetime.datetime.now()

# cloud volumes
flat_seg_spinalcord = cloudvolume.CloudVolume(config["SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False, parallel=False)
seg_spinalcord = cloudvolume.CloudVolume(config["PROOFREAD_SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False, parallel=False)

client = caveclient.CAVEclient(config["CAVE_CLIENT_DATASTACK"])

if __name__ == "__main__":


    # Path to emain embedding directory
    TRAIN_DATASET_PATH = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/dataset/embeddings_unnorm'

    # Collect all data files for each syn data
    print("Collect all data files for each syn data")
    syn_data_list = []
    for class_name in config["LABEL_MAP"].keys():
        data_files = glob.glob(os.path.join(TRAIN_DATASET_PATH, class_name, 'data', '*.json'))
        for data_file in data_files:
            with open(data_file) as f:
                root_id_data = json.load(f)            
            syn_data_list.append([root_id_data['initial_pt'], root_id_data['root_id'], class_name, data_file])

    train_data = {key: {} for key in config["LABEL_MAP"]}


    print('Combining data files from the same root id ...')
    for syn_data in tqdm(syn_data_list, desc='Running'):
        class_name = syn_data[-2]
        data_file = syn_data[-1]
        new_seg_id = get_seg_id_from_coord(syn_data[0], seg_spinalcord, seg_mip=(32,32,45), coordinate_mip=(32,32,45))
        if new_seg_id not in train_data[class_name]:
            train_data[class_name][new_seg_id] = []
        train_data[class_name][new_seg_id].append(data_file)


    # Save to pickle
    train_data_file_path = f"datasets/train_data_split_by_pycg_{now.strftime('%d%m%y')}.pkl"
    print("Saving training data to file:", train_data_file_path)
    with open(train_data_file_path, 'wb') as f:
        pickle.dump(train_data, f)

    print("Number of unique IDs and data files per class:")
    for key in train_data:
        data_files = [item for sublist in [val for val in train_data[key].values()] for item in sublist]
        print(" -", key, len(train_data[key]), len(data_files))

    print("Creating train and test split")
    X, y = [],[] # Gather train and test split
    for class_name in train_data:
        for root_id in train_data[class_name]:
            X.append(root_id)
            y.append(class_name)
    assert len(X) == len(y)

    # Create train-test split
    X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = \
        train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    train_data_files = {}
    test_data_files = {}

    for root_id, class_name in zip(X_train_stratified, y_train_stratified):
        if class_name not in train_data_files:
            train_data_files[class_name] = []
        for data_file in train_data[class_name][root_id]:
            train_data_files[class_name].append(data_file)

    for root_id, class_name in zip(X_test_stratified, y_test_stratified):
        if class_name not in test_data_files:
            test_data_files[class_name] = []
        for data_file in train_data[class_name][root_id]:
            test_data_files[class_name].append(data_file)    


    print("###### Training and Test # of data files per class######")
    for class_name in train_data_files.keys():
        print(" -", class_name, len(train_data_files[class_name]), len(test_data_files[class_name]))

    # Save to pickle
    train_test_split_data = {
        'train': train_data_files,
        'test': test_data_files,
    }

    train_test_split_data_file_path = f"datasets/train_test_data_split_{now.strftime('%d%m%y')}.pkl"
    print("Saving test_test_split data to file:", train_test_split_data_file_path)
    with open(train_test_split_data_file_path, 'wb') as f:
        pickle.dump(train_test_split_data, f)
    print("Done.")