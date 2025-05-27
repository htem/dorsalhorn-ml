import os
import numpy as np
import tifffile
import json


def create_dataset_embeds(train_data_files, test_data_files, parent_dir, label_map, CLASS_DIR):
    print("Collecting training/testing cutouts ...")
    
    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    for class_n in label_map.keys():
        
        assert class_n in train_data_files
        assert class_n in test_data_files
    
        class_train_data_files = train_data_files[class_n].copy() # get training data
        #cutouts = []
        for json_file in class_train_data_files[:50]:
            json_file = os.path.join(parent_dir, json_file)
            data_embedding = []
            with open(json_file, 'r') as file:
                center_pt_data = json.load(file)

            cutout_tif_file_path = f"{int(center_pt_data['initial_pt_in_nm'][0])}_{int(center_pt_data['initial_pt_in_nm'][1])}_{int(center_pt_data['initial_pt_in_nm'][2])}.tiff"
            cutout_tif_file_path = os.path.join(CLASS_DIR, class_n, 'cutouts', cutout_tif_file_path)
            if os.path.exists(cutout_tif_file_path):
                vol = tifffile.imread(cutout_tif_file_path)[None,:,:,:]
                #vol = np.repeat(vol, 3, axis=0)  # Shape: (3, 129, 129, 129)
                train_data.append(vol[None,:,:,:,:])
                train_labels.append(label_map[class_n])
            else:  
                raise ValueError(f"Error loading {cutout_tif_file_path}")
        
     
        class_test_data_files = test_data_files[class_n].copy() # get training data
        for json_file in class_test_data_files[:25]:
            json_file = os.path.join(parent_dir, json_file)
            data_embedding = []
            with open(json_file, 'r') as file:
                center_pt_data = json.load(file)
    
            cutout_tif_file_path = f"{int(center_pt_data['initial_pt_in_nm'][0])}_{int(center_pt_data['initial_pt_in_nm'][1])}_{int(center_pt_data['initial_pt_in_nm'][2])}.tiff"
            cutout_tif_file_path = os.path.join(CLASS_DIR, class_n, 'cutouts', cutout_tif_file_path)

            if os.path.exists(cutout_tif_file_path):
                vol = tifffile.imread(cutout_tif_file_path)[None,:,:,:]
                #print(vol.shape, vol.min(), vol.max())
                #raise ValueError()
                #vol = np.repeat(vol, 3, axis=0)
                test_data.append(vol[None,:,:,:,:])
                test_labels.append(label_map[class_n])
            else:
                raise ValueError(f"Error loading {cutout_tif_file_path}")
  
    train_data = np.concatenate(train_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    print('Train data:', train_data.shape, train_labels.shape)
    print('Test data:', test_data.shape, test_labels.shape)

    return train_data, test_data, train_labels, test_labels
