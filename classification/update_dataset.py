import caveclient
import cloudvolume
import os
import json
import numpy as np
import pickle
from utils import *
import random
from tqdm import tqdm
import sys
import yaml

""" Use this script to update the training and testing dataset files """

################################ Parameters ############################
TRAIN_EMBED_FILE_PATH = 'datasets/training_embeds_1050_313_useallsyn.pkl'
TEST_EMBED_FILE_PATH = 'datasets/testing_embeds_1050_313_useallsyn.pkl'

TRAIN_EMBED_FILE_PATH_NEW = 'datasets/training_embeds_1050_313_useallsyn_05222025.pkl'
TEST_EMBED_FILE_PATH_NEW = 'datasets/testing_embeds_1050_313_useallsyn_0522202.pkl'

AGGREGATE_RADIUS_UM = 10

MODELS2USE = ["microns", "h01", "sp"]

########################################################################

# Caveclient
client = caveclient.CAVEclient("brain_and_nerve_cord")

# Segmentation cloud volume
proofread_seg_spinalcord = cloudvolume.CloudVolume("graphene://https://cave.fanc-fly.com/segmentation/table/wclee_mouse_spinalcord_cltmr", use_https=True, mip=(32,32,45), agglomerate=True, progress=False)

print("Updating train and test files:")
train_test_split_file = os.path.join('datasets', 'train_test_split_neurons.json')
with open(train_test_split_file, 'r') as f:
    data = json.load(f)
    class_to_root_data = data['class_to_root_data']
    train_neurons = data['train_neurons']
    test_neurons = data['test_neurons']
print(f'- Loaded train_test_split file {train_test_split_file}')

with open(TRAIN_EMBED_FILE_PATH, 'rb') as f:
    train_data_files = pickle.load(f)
with open(TEST_EMBED_FILE_PATH, 'rb') as f:
    test_data_files = pickle.load(f)
print("- Loaded embedding files")

print('\n')
#####################################################################################################

######################################## Update to data file paths ##################################
for key in train_data_files:
    new_data_files = []
    for data_file in train_data_files[key]:
        if '/embeddings/' in data_file:
            data_file = data_file.replace('/embeddings/', '/embeddings_unnorm/')
        new_data_files.append(data_file)
    
    train_data_files[key] = new_data_files

for key in test_data_files:
    new_data_files = []
    for data_file in test_data_files[key]:
        if '/embeddings/' in data_file:
            data_file = data_file.replace('/embeddings/', '/embeddings_unnorm/')
        new_data_files.append(data_file)
    
    test_data_files[key] = new_data_files

################## Switching improved test performance across all adhtmrp/cheat neurons #############
train_test_swap = [['720575940886024902', '720575940880965810', 'chtmrp']]

for train_root_id, test_root_id, class_name in train_test_swap:
    embed_files_to_move, embed_files_to_move_test = [], []
    for file in train_data_files[class_name]:
        if train_root_id in file:
            embed_files_to_move.append(file)
    train_data_files[class_name] = [item for item in train_data_files[class_name] if item not in embed_files_to_move]
    assert len(embed_files_to_move) > 0

    for file in test_data_files[class_name]:
        if test_root_id in file:
            embed_files_to_move_test.append(file)
    test_data_files[class_name] = [item for item in test_data_files[class_name] if item not in embed_files_to_move_test]

    train_data_files[class_name].extend(embed_files_to_move_test)
    test_data_files[class_name].extend(embed_files_to_move)

print("- Switched neurons from train <-> test")
#######################################################################################################


######################## Manually add more embeddings from infernce (by Wangchu) ######################
added_train_data_files = {}
added_train_embed = {}
added_label_map = {'nonsensory': 'nonsensory', 'cltmr': 'cltmr', 'chtmr-np': 'chtmrnp', 'Aβ-LTMR': 'abltmr', 'Aδ-HTMR & C-Heat': 'adhtmrp', 'Aδ-LTMR': 'adltmr', 'Aδ-HTMR': 'adhtmrp', 'C-HTMR': 'othersn'}

man_additional_annots = [
    ['man_annots/man_gt_wax_cltmr.json', "../dataset/embeddings_13892_4122_2284/inference"],
    ['man_annots/man_gt_wax_abltmr.json', "../dataset/embeddings_21132_6350_1717/inference"],
    ['man_annots/man_gt_wax_adhtmrcheat_1.json', "../dataset/embeddings_31483_2651_816/inference"],
    ['man_annots/man_gt_wax_adhtmrcheat_2.json', "../dataset/embeddings_17587_2897_314/inference"]
]

print("Loading additional training data ...")
for man_annot_file, man_embed_dir in tqdm(man_additional_annots, desc="Processing"):
    print(f"- adding {man_annot_file}")
    for coord, label in load_man_annots(man_annot_file):
        if 'non' in label or 'glia' in label:
            label = 'nonsensory'

        coord[0] *= 32
        coord[1] *= 32
        coord[2] *= 45
        data_file = f"*_{str(coord[0])}_{str(coord[1])}_{str(coord[2])}*"
        data_file_path = glob.glob(man_embed_dir + '/data/' + data_file)[0]
        if added_label_map[label] not in added_train_data_files:
            added_train_data_files[added_label_map[label]] = []
            added_train_embed[added_label_map[label]] = []
        added_train_data_files[added_label_map[label]].append(data_file_path)
        added_train_embed[added_label_map[label]].append(gen_embeddings_from_json(data_file_path, embedding_dir = man_embed_dir, AGG_RADIUS=AGGREGATE_RADIUS_UM, MODELS2USe=MODELS2USE))

for key in added_train_data_files.keys():
    train_data_files[key].extend(added_train_data_files[key])
    added_train_embed[key] = np.concatenate(added_train_embed[key], axis=0)

print('\n')
######################################################################################################

#################### Confirm there are no training root ids/segments in the test dataset #############
train_data_files, test_data_files = validate_train_test_split_edit(seg_vol=proofread_seg_spinalcord, client=client, 
    train_data_files=train_data_files, test_data_files=test_data_files)

######################################################################################################

with open(TRAIN_EMBED_FILE_PATH_NEW, 'wb') as f:
    pickle.dump(train_data_files, f)
print(f"New train data file created: {TRAIN_EMBED_FILE_PATH_NEW}")

with open(TEST_EMBED_FILE_PATH_NEW, 'wb') as f:
    pickle.dump(test_data_files, f)
print(f"New test data file created: {TEST_EMBED_FILE_PATH_NEW}")