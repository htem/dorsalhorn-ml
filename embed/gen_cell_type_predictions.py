import os
import sys
import json
import numpy as np
import pickle
sys.path.append('/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/classification')
from utils import *
from models import MLP
import yaml
import torch
import torch.nn.functional as F
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Config 

# Location of all data
dataset_dir = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/dataset'

# Output directory to save dataframes
output_dir = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/output/v3'

# Output directory to save figures
figures_dir = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/output/v3/figures'

MODELS2USe = ['microns', 'h01', 'sp']
AGG_RADIUS = 10

# Load up MLP
print("Loading classifiers")
cell_type_model_path = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/classification/models/mlp_cell_type_classifier_microns_h01_sp_10umagg_v3.pth'
sensory_non_sensory_model_path = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/classification/models/mlp_sen_nonsen_classifier_99acc_micronsh01_noagg_v2.pth'
cell_type_adhtmr_cheat_model_path = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/classification/models/mlp_cell_type_classifier_adhtmrp_cheat_microns_h01_sp_10umagg.pth'

cell_type_adhtmr_cheat_classifier_version = 'v1_04012025'
sensory_nonsensory_classifier_version = "v2_04292025"
cell_type_classifier_version = "v3_05152025"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read in sensory vs non sensory classifier
sensory_non_sensory_classifier_model = MLP(input_dim=128, num_classes=2) 
_ = sensory_non_sensory_classifier_model.load_state_dict(torch.load(sensory_non_sensory_model_path, map_location=device))
sensory_non_sensory_classifier_model = sensory_non_sensory_classifier_model.to(device)
_ = sensory_non_sensory_classifier_model.eval()
print(f"- loaded sensory vs. non-sensory neuron classifier")

# Read in cell type classifier
cell_type_classifier_model = MLP(input_dim=192, num_classes=7) 
_ = cell_type_classifier_model.load_state_dict(torch.load(cell_type_model_path, map_location=device))
cell_type_classifier_model = cell_type_classifier_model.to(device)
_ = cell_type_classifier_model.eval()
print(f"- loaded cell type classifier")

# Read in cell type (adhtmr-cheat) classifier
cell_type_adhtmr_cheat_classifier_model = MLP(input_dim=192, num_classes=2) 
_ = cell_type_adhtmr_cheat_classifier_model.load_state_dict(torch.load(cell_type_adhtmr_cheat_model_path, map_location=device))
cell_type_adhtmr_cheat_classifier_model = cell_type_adhtmr_cheat_classifier_model.to(device)
_ = cell_type_adhtmr_cheat_classifier_model.eval()
print(f"- loaded adhtmrp vs. c-heat classifier")

# Create util sub directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


def helper_select_embed_dirs():
    """ Helper function to get list of all embeddings directories """
    dataset_dir = '/n/data3_vast/hms/neurobio/htem2/users/kd193/spinal_cord_embedding/dataset'
    pattern = re.compile(r'^embeddings_\d+_\d+_\d+(?:_|$)')
    filtered_dirs = [d for d in os.listdir(dataset_dir) if pattern.match(d)]
    
    SEGMENT_PREFIX_LIST = []
    PREFIX_LIST = []
    for dir_name in filtered_dirs:
        if 'axoaxonic' in dir_name:
            PREFIX_LIST.append('_axoaxonic')
        else:
            PREFIX_LIST.append('')

        _, x, y, z, *_ = dir_name.split('_')
        SEGMENT_PREFIX_LIST.append([int(x), int(y), int(z)])

    return SEGMENT_PREFIX_LIST, PREFIX_LIST


if __name__ == "__main__":

    SEGMENT_PREFIX_LIST, EXTRA_PREFIX_LIST = helper_select_embed_dirs()
    EXTRA_PREFIX_OUT = ''

    for segment_prefix, EXTRA_PREFIX in zip(SEGMENT_PREFIX_LIST, EXTRA_PREFIX_LIST):

        print("\nWith segment prefix:", segment_prefix)
        segment_prefix = f"{str(segment_prefix[0])}_{str(segment_prefix[1])}_{str(segment_prefix[2])}{EXTRA_PREFIX}"

        embedding_dir = os.path.join(dataset_dir, f'embeddings_{segment_prefix}', 'inference')
        all_synapses_data_files = sorted(glob.glob(os.path.join(embedding_dir, 'data', '*.json'))) # Get all JSON files
        all_synapses_data_files_valid = []
        all_embeddings, synapse_ids, synapse_coords = [], [], []
        syn_count = 0
        print("- Number of synapses available:", len(all_synapses_data_files))

        for syn_data_file in tqdm(all_synapses_data_files, desc='- Collecting embeddings'):
            with open(syn_data_file, 'r') as file:
                synapse_meta_data = json.load(file)

            root_id = synapse_meta_data["root_id"]
            coords_2_embed = synapse_meta_data["nm_coords"]
            initial_pt = synapse_meta_data["initial_pt"] # in 32nm
            initial_pt_in_nm = synapse_meta_data["initial_pt_in_nm"]

            if 'synapse_id' in synapse_meta_data:
                if synapse_meta_data['synapse_id'] != None:
                    synapse_id = int(synapse_meta_data['synapse_id'])
                else:
                    synapse_id = syn_count
                    syn_count += 1
            else:
                synapse_id = syn_count
                syn_count += 1
                
            # Filter coordinates to only collect embeddings within ina AGG_RADIUS
            filtered_coords_2_embed = []
            for center_pt_nm in coords_2_embed:
                euc_dist = np.linalg.norm(initial_pt_in_nm - np.array(center_pt_nm))
                if euc_dist <= AGG_RADIUS *1000:
                    filtered_coords_2_embed.append(center_pt_nm)

            # Convert coordinates to .pkl files
            nm_coords_tuple = [(coord[0], coord[1], coord[2]) for coord in filtered_coords_2_embed]
            nm_coords_file_paths = [f"{str(int(coord[0]))}_{str(int(coord[1]))}_{str(int(coord[2]))}.pkl" for coord in filtered_coords_2_embed]

            # Concatenate embeddings here (MICrONs, H01, Own SpinalCord)
            concatenated_embeddings = []

            leave = False
            # Loop through each embedding model
            for model in MODELS2USe:
                # Append to this and aggregate embeddings (take mean)
                aggregate_embeddings = []

                embedding_info_path = os.path.join(embedding_dir, 'embeddings_' + model, f'{root_id}_embeddings.pkl')
                with open(embedding_info_path, 'rb') as file:
                    embedding_info = pickle.load(file)
                for nm_coord_key in nm_coords_tuple: # Through each coordinate
                    embedding = embedding_info['nm_center_pts_2_embeddings'][nm_coord_key]
                    try:
                        aggregate_embeddings.append(embedding[np.newaxis,:])
                    except Exception as e:
                        leave = True
                        break # leave loop if embedding isnt done
                if leave:
                    break
   
                aggregate_embeddings = np.concatenate(aggregate_embeddings, axis=0)
                aggregate_embeddings = np.array([np.mean(aggregate_embeddings, axis=0)])
                concatenated_embeddings.append(aggregate_embeddings)
                        
            if leave:
                continue

            concatenated_embeddings = np.concatenate(concatenated_embeddings, axis=1)
            all_synapses_data_files_valid.append(syn_data_file)
            all_embeddings.append(concatenated_embeddings)
            synapse_ids.append(synapse_id)
            synapse_coords.append(initial_pt)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        synapse_ids = np.array(synapse_ids)
        synapse_coords = np.array(synapse_coords)
        print('- Full embedding shape:', all_embeddings.shape)

        # Labels
        sensory_non_sensory_label_map = {0: True, 1: False,}
        cell_type_label_map = {-1: 'N/A', 0: 'C-LTMR', 1: 'Aβ-LTMR', 2: 'Aδ-LTMR', 3: 'C-HTMR-Non-peptidergic', 
                    4: 'C-HTMR',5: 'Aδ-HTMR & C-Heat', 6: 'C-Cold', 
                    7: 'Aδ-HTMR', 8: 'C-Heat'}
        cell_type_adhtmrp_cheat_label_map = {0: 'Aδ-HTMR', 1: 'C-Heat'}

        print('Running the sensory vs. non-sensory neuron classifier ...')
        # Convert embeddings to tensor and run
        input_data = torch.tensor(all_embeddings[:,:128], dtype=torch.float32)
        input_data = input_data.to(device)

        # Predict synapse as sensory vs non-sensory
        with torch.no_grad():
            outputs = sensory_non_sensory_classifier_model(input_data)
        outputs = F.softmax(outputs, dim=1)
        sensory_non_sensory_class_probabilities = outputs.cpu().detach().numpy()
        sensory_non_sensory_class_probabilities = np.round(sensory_non_sensory_class_probabilities, decimals=4) 
        predicted_sensory_non_sensory_classes = np.argmax(sensory_non_sensory_class_probabilities, axis=1)

        # Only keep synapses that are classified as sensory
        filter_sensory_neurons_only_mask = predicted_sensory_non_sensory_classes == 0 # sensory = 0
        all_embeddings_sensory = all_embeddings[filter_sensory_neurons_only_mask, :]
        print(f'{len(all_embeddings_sensory)}/{len(all_embeddings)} synapses predicted as sensory!')

        print('Predicting cell type per synapse ...')
        # Predict synapse cell type
        input_data = torch.tensor(all_embeddings_sensory, dtype=torch.float32)
        input_data = input_data.to(device)
        with torch.no_grad():
            outputs = cell_type_classifier_model(input_data)
        outputs = F.softmax(outputs, dim=1)
        cell_type_class_probabilities = outputs.cpu().detach().numpy()
        cell_type_class_probabilities = np.round(cell_type_class_probabilities, decimals=4) 
        cell_type_class_probabilities_main = np.ones((sensory_non_sensory_class_probabilities.shape[0], 7)) * -1
        cell_type_class_probabilities_main[filter_sensory_neurons_only_mask, :] = cell_type_class_probabilities
        predicted_cell_type_classes = np.argmax(cell_type_class_probabilities, axis=1)
        predicted_cell_type_classes_main = np.ones((sensory_non_sensory_class_probabilities.shape[0],)) * -1
        predicted_cell_type_classes_main[filter_sensory_neurons_only_mask] = predicted_cell_type_classes

        # Run other classifier for adhtmr-cheat predictions
        filter_adhtmrp_cheat_only_mask = predicted_cell_type_classes == 5 
        all_embeddings_sensory_adhtmr_cheat = all_embeddings_sensory[filter_adhtmrp_cheat_only_mask, :]

        print('Breaking down AD-HMTRP/C-HEAT predictions ...')
        # Predict synapse cell type
        input_data = torch.tensor(all_embeddings_sensory_adhtmr_cheat, dtype=torch.float32)
        input_data = input_data.to(device)
        with torch.no_grad():
            outputs = cell_type_adhtmr_cheat_classifier_model(input_data)
        outputs = F.softmax(outputs, dim=1)
        cell_type_adhtmr_cheat_class_probabilities = outputs.cpu().detach().numpy()
        cell_type_adhtmr_cheat_class_probabilities = np.round(cell_type_adhtmr_cheat_class_probabilities, decimals=4) 
        cell_type_adhtmr_cheat_class_probabilities_main = np.ones((sensory_non_sensory_class_probabilities.shape[0], 2)) * -1

        filter_adhtmrp_cheat_only_mask_main = predicted_cell_type_classes_main == 5
        cell_type_adhtmr_cheat_class_probabilities_main[filter_adhtmrp_cheat_only_mask_main, :] = cell_type_adhtmr_cheat_class_probabilities
        predicted_cell_type_adhtmr_cheat_classes = np.argmax(cell_type_adhtmr_cheat_class_probabilities, axis=1)
        
        # change predicts from 0-1 to 7-8
        predicted_cell_type_adhtmr_cheat_classes[predicted_cell_type_adhtmr_cheat_classes==0] = 7
        predicted_cell_type_adhtmr_cheat_classes[predicted_cell_type_adhtmr_cheat_classes==1] = 8
        predicted_cell_type_classes_main[filter_adhtmrp_cheat_only_mask_main] = predicted_cell_type_adhtmr_cheat_classes

        assert 5 not in predicted_cell_type_classes_main, "Value 6 found in the cell type class predictions!" 

        predicted_sensory_non_sensory_classes_label = [sensory_non_sensory_label_map[pred_label] for pred_label in predicted_sensory_non_sensory_classes]
        predicted_cell_type_classes_label = [cell_type_label_map[pred_label] for pred_label in predicted_cell_type_classes_main]

        print('Done!')

        print("Generating dataframe ...")
        cell_type_classifier_data = [
            synapse_ids, synapse_coords,
            sensory_non_sensory_class_probabilities[:,0],
            sensory_non_sensory_class_probabilities[:,1],
            predicted_sensory_non_sensory_classes_label,
            
            cell_type_class_probabilities_main[:,0],
            cell_type_class_probabilities_main[:,1],
            cell_type_class_probabilities_main[:,2],
            cell_type_class_probabilities_main[:,3],
            cell_type_class_probabilities_main[:,4],
            cell_type_class_probabilities_main[:,5],
            cell_type_class_probabilities_main[:,6],  
            cell_type_adhtmr_cheat_class_probabilities_main[:,0],
            cell_type_adhtmr_cheat_class_probabilities_main[:,1],
            predicted_cell_type_classes_label,

            [sensory_nonsensory_classifier_version for _ in range(len(predicted_cell_type_classes_label))],
            [cell_type_classifier_version for _ in range(len(predicted_cell_type_classes_label))],
            [cell_type_adhtmr_cheat_classifier_version for _ in range(len(predicted_cell_type_classes_label))],
        ]

        synapse_cell_type_pred_df = pd.DataFrame(list(zip(*cell_type_classifier_data)), columns=['id', 'pt_position', 'sensory', 'non-sensory', 'is_sensory',
                                                    'cltmr', 'abltmr', 'adltmr', 'chtmrnp', 'c-htmr', 'htmrp', 'cold', 'adhtmrp', 'cheat', 'cell_type',
                                                    'sen_nonsen_model_ver', 'cell_type_model_ver', 'cell_type_adhtmr_cheat_model_ver'])
        segment_prefix = segment_prefix + EXTRA_PREFIX_OUT
        output_df_path = os.path.join(output_dir, f"cell_type_predictions_{segment_prefix}.feather")
        synapse_cell_type_pred_df.to_feather(output_df_path)
        print("Created", output_df_path)
                
        # Create pie chart
        CELL_TYPE_MIN_PROB = 0.7
        SEN_VS_NONSEN_MIN_PROB = 0.7

        # sensory_non_sensory_label_map = {0: 'Sensory', 1: 'Non-sensory',}
        # cell_type_label_map = {-1: 'N/A', 0: 'C-LTMR', 1: 'Aβ-LTMR', 2: 'Aδ-LTMR', 3: 'C-HTMR-Non-peptidergic', 
        #             4: 'C-HTMR', 5: 'Aδ-HTMR & C-Heat', 6: 'C-Cold'}

        # Get sensory vs. nonsensory synapse count
        sensory_count = len(synapse_cell_type_pred_df[synapse_cell_type_pred_df['is_sensory'] == True])
        nonsensory_count = len(synapse_cell_type_pred_df[synapse_cell_type_pred_df['is_sensory'] == False])
        predicted_sensory_non_sensory_counts = [sensory_count, nonsensory_count]
        predicted_sensory_non_sensory_labels = ['Sensory', 'Non-sensory']

        predicted_cell_type_counts, predicted_cell_type_labels = [], []

        # Only sensory neurons | Filter only synapses with > prob_thresh
        synapse_cell_type_pred_df_only_sensory = synapse_cell_type_pred_df[synapse_cell_type_pred_df['is_sensory'] == True]
        synapse_cell_type_pred_df_only_sensory = synapse_cell_type_pred_df_only_sensory[synapse_cell_type_pred_df_only_sensory['sensory'] >= SEN_VS_NONSEN_MIN_PROB]

        # Get max probability for each row
        synapse_cell_type_pred_df_only_sensory['max_probability'] = synapse_cell_type_pred_df_only_sensory[['cltmr', 'abltmr', 
                                                    'adltmr', 'chtmrnp', 'c-htmr', 'cold', 'adhtmrp', 'cheat']].max(axis=1)

        # Filter only synapses with > prob_thresh
        synapse_cell_type_pred_df_only_sensory = synapse_cell_type_pred_df_only_sensory[synapse_cell_type_pred_df_only_sensory['max_probability'] >= CELL_TYPE_MIN_PROB]

        for cell_type in synapse_cell_type_pred_df_only_sensory['cell_type'].unique():
            predicted_cell_type_counts.append(len(synapse_cell_type_pred_df_only_sensory[synapse_cell_type_pred_df_only_sensory['cell_type']==cell_type]))
            predicted_cell_type_labels.append(cell_type)

        # Sort
        paired = list(zip(predicted_cell_type_counts, predicted_cell_type_labels))
        paired.sort(reverse=True)
        predicted_cell_type_counts, predicted_cell_type_labels = zip(*paired)
        predicted_cell_type_counts, predicted_cell_type_labels = list(predicted_cell_type_counts), list(predicted_cell_type_labels)

        # Create figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(14, 10))

        sen_nonsen_label_color_map = {
            "Sensory": "white",  
            "Non-sensory": "gray", }

        cell_type_label_color_map = {
            "C-Cold": "#1f77b4",  # blue
            "Aβ-LTMR": "#ff7f0e",  # orange
            "Aδ-LTMR": "#2ca02c",  # green
            "Aδ-HTMR": "#d62728",  # red
            "C-Heat": "#b22222",  # darker red
            "C-HTMR": "#9467bd",  # purple
            "C-HTMR-Non-peptidergic": "#8c564b",  # brown
            "C-LTMR": "#e377c2",  # pink
        }

        # Generate color lists for each pie chart using the map
        colors_sensory = [sen_nonsen_label_color_map[label] for label in predicted_sensory_non_sensory_labels]
        colors_cell_types = [cell_type_label_color_map[label] for label in predicted_cell_type_labels]

        wedges, texts, autotexts = ax[0].pie(predicted_sensory_non_sensory_counts, wedgeprops=dict(edgecolor='black'), autopct='%1.1f%%', colors=colors_sensory)
        ax[0].legend(wedges, predicted_sensory_non_sensory_labels, title="Cell types", loc="upper left")

        wedges, texts, autotexts = ax[1].pie(predicted_cell_type_counts, wedgeprops=dict(edgecolor='black'), autopct='%1.0f%%', colors=colors_cell_types)
        ax[1].legend(wedges, predicted_cell_type_labels, title="Cell types", loc="upper left")

        plt.tight_layout()

        output_figure_path = os.path.join(output_dir, "figures", f"chart_{segment_prefix}.png")
        plt.savefig(output_figure_path, dpi=300)
        print("Created pie chart figure", output_figure_path)
        
