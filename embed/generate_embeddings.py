import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
sys.path.append(repo_dir + '/connectomics')
sys.path.append(repo_dir + '/simclr')

from connectomics.segclr.tf2 import legacy_model

import caveclient
import cloudvolume
import glob
import json
import numpy as np
from meshparty import trimesh_io
import pickle
import tensorflow as tf
from tqdm import tqdm
import yaml
import tifffile

from embed.utils import * # custom library with util functions

# read configuration file (config.yaml)
with open("config_embed.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Load cloudvolume for em and segmentation layers
em_spinalcord = cloudvolume.CloudVolume(config["EM_PATH"], use_https=True, fill_missing=True, mip=config["MIP_EM_VOL"], progress=False)
seg_spinalcord = cloudvolume.CloudVolume(config["SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False)

# MICrONS embedding model
microns_model = legacy_model.LegacySegClrModel()
input_microns = tf.keras.Input(shape=config["INPUT_SHAPE"], batch_size=1, dtype=tf.float32)
output_microns = microns_model(input_microns, training=False)
ckpt_microns = tf.train.Checkpoint(microns_model)
_ = ckpt_microns.restore(config["MICRONS_EMBED_MODEL_PATH"]).expect_partial()

# H01 embedding model
h01_model = legacy_model.LegacySegClrModel()
input_h01 = tf.keras.Input(shape=config["INPUT_SHAPE"], batch_size=1, dtype=tf.float32)
output_h01 = h01_model(input_h01, training=False)
ckpt_h01 = tf.train.Checkpoint(h01_model)
_ = ckpt_h01.restore(config["H01_EMBED_MODEL_PATH"]).expect_partial()

# CaveClient for spinal cord dataset
client = caveclient.CAVEclient(config["CAVECLIENT_DATASTACK"])

# Load MeshParty
mm = trimesh_io.MeshMeta(
    cv_path=config["SEG_PATH"],
    disk_cache_path=config["SEG_MESHES"],
    map_gs_to_https=True
)


if __name__ == "__main__":

    # Path to annotations json file
    # - make sure json is in the dataset/annots folder

    annot_json_file = config["ANNOT_FILE"]

    print(f"\n---- Embedding coordinates in {annot_json_file} ----")

    # coordinates converted to seg_vol mip
    root_id_2_coords = create_rootid_2_coord_map(annot_json_file, gt_label=config["GT_LABEL_NAME"],
                                                 seg_vol=seg_spinalcord, output_dir=config["ANNOTS_OUTPUT_DIR"],
                                                 NG_MIP=config["NG_MIP"], MIP_SEG_VOL=config["MIP_SEG_VOL"])
    # Create root id data files
    for root_id in root_id_2_coords:
        print(f"WORKING W/ ROOT ID: {root_id}")

        if root_id in ['720575940537497676', '720575940531007242', 
                       '720575940531010058', '720575940535288488',
                       '720575940539856067', '720575940530227826',
                       '720575940536375789', '720575940534800232']:
            continue
        
        for center_coord in root_id_2_coords[root_id]['coords']:
            print(f"\tCOORDINATE: {center_coord}")
            print(f"\tGROUND-TRUTH: {root_id_2_coords[root_id]['label']}")
            gt_label = root_id_2_coords[root_id]['label']
            
            #### 1. Create graphs and generate nm points to embed 
            initial_center_coord = center_coord.copy() # expected in 32x32x45nm
        
            # convert original coordinate to real-world nm coordinates
            center_coord = [center_coord[0]*config["MIP_SEG_VOL"][0], 
                            center_coord[1]*config["MIP_SEG_VOL"][1], 
                            center_coord[2]*config["MIP_SEG_VOL"][2]]
            
            new_key = f"{str(root_id)}_{str(int(center_coord[0]))}_{str(int(center_coord[1]))}_{str(int(center_coord[2]))}_data"
            json_file = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, config["DATA_FOLDER"], new_key + '.json') # save to ground-truth class folder
        
            if not os.path.exists(json_file):
                if config["TRAINED_AGG_RADIUS_UM"] >= 5:
                    # skeletonize using computed meshes
                    sk = get_root_id_skeleton(seg_id=int(root_id), mm=mm)
                
                    # find closest sk node to the provided coordinate
                    closest_sk_node_to_coord = find_closest_sk_node(center_coord, sk, verbose=True)
                    sk_center_coord = list(sk.vertices[closest_sk_node_to_coord])
                    
                    # only gather sk nodes within a {TRAINED_AGG_RADIUS_UM} radius
                    valid_sk_node_indices = []
                    max_dist_from_center_coord = 0 # keep track of furthest node from center coord aggregated (20um max)
                    for i in range(len(sk.vertices)):
                        euc_dist = np.linalg.norm(center_coord - sk.vertices[i])
                        if euc_dist <= config["TRAINED_AGG_RADIUS_UM"]*1000:
                            valid_sk_node_indices.append(i)
                            if euc_dist > max_dist_from_center_coord:
                                max_dist_from_center_coord = euc_dist
            
                    # create simplified graph with all potential nodes to embed
                    G = create_graph_w_filter(sk, valid_sk_node_indices, CONNECT_MIN = 1000, verbose=False)
                    # MG = use_best_G(G, [closest_sk_node_to_coord])
                    #MG = simplify_graph(MG, MINIMUM_DISTANCE = 1500)
                    G_list = get_simplified_graphs(G, MINIMUM_DISTANCE=1500)
                    node_nm_coords = []
                    for MG in G_list:
                        node_nm_coords.extend([list(sk.vertices[node]) for node in MG.nodes()])
                else:
                    node_nm_coords = []
                    max_dist_from_center_coord = 0
                    sk_center_coord = center_coord
        
                # add initial pt to coord list
                if center_coord not in node_nm_coords:
                    node_nm_coords.append(center_coord)
        
                root_id_data = {
                    "root_id": str(root_id),
                    "nm_coords": node_nm_coords,
                    "ground_truth_label":  root_id_2_coords[root_id]['label'], #temp
                    "initial_pt": initial_center_coord, #in 32nm
                    "initial_pt_in_nm": center_coord, # in rw nm
                    "furthest_pt_in_nm": max_dist_from_center_coord,
                    "closest_sk_coord": sk_center_coord
                }
            
                os.makedirs(os.path.dirname(json_file), exist_ok=True) # create directories if needed
                with open(json_file, 'w') as f:
                    json.dump(root_id_data, f)
                print(f"\t{os.path.basename(json_file)} created.")


    ########################################################################################################################

    DATA_FOLDER_PATH = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, config["DATA_FOLDER"])
    data_file_list = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.json"))
    num_of_files_in_list = len(data_file_list)

    for fi, data_file in enumerate(data_file_list):
        with open(data_file, 'r') as file:
            root_id_data = json.load(file)

        root_id = root_id_data["root_id"]
        gt_label = root_id_data['ground_truth_label']
        coords_2_embed = root_id_data["nm_coords"]
        initial_pt = root_id_data["initial_pt"] # in 32nm
        initial_pt_in_nm = root_id_data["initial_pt_in_nm"]

        print(f"{fi}/{num_of_files_in_list} WORKING ON EMBEDDING ROOT ID: {root_id}")
        print(f"\tCOORDINATE: {initial_pt}")
        print(f"\tGROUND-TRUTH: {gt_label}")

        # filter coordinates
        filtered_coords_2_embed = []
        if config["EMBED_AGG_RADIUS_UM"] < 5:
            filtered_coords_2_embed.append(initial_pt_in_nm)
        else:
            for center_pt_nm in coords_2_embed:
                euc_dist = np.linalg.norm(initial_pt_in_nm - np.array(center_pt_nm))
                if euc_dist <= config["EMBED_AGG_RADIUS_UM"] *1000:
                    filtered_coords_2_embed.append(center_pt_nm)

        for center_pt_nm in tqdm(filtered_coords_2_embed, desc="Generating embeddings"):
            embed_key = f"{str(int(center_pt_nm[0]))}_{str(int(center_pt_nm[1]))}_{str(int(center_pt_nm[2]))}"
            
            embed_done = True
            for model_key in config["MODELS_2_USE"]:
                pickle_file_path = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'embeddings_'+model_key, embed_key + '.pkl')
                if not config["OVERWRITE_FILES"]: # skip if file for coordinate already exists
                    if not os.path.exists(pickle_file_path):
                        embed_done = False
            if embed_done:
                pass
            
            vol_cutout, success = get_local_3d_view(center_pt_nm, root_id, config["MIP_EM_VOL"], 
                    config["MIP_SEG_VOL"], config["VOL_MIP"], em_spinalcord, seg_spinalcord, client, initial_pt=initial_pt)
            
            if not success:
                continue

            original_vol_cutout = vol_cutout.copy()
            if np.all(vol_cutout==0):
                raise ValueError("3D volume is empty.")
                
            assert np.array_equal(original_vol_cutout, vol_cutout)

            if config['SAVE_VOL_CUTOUTS']:
                tiff_file_path = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'cutouts', embed_key + '.tiff')
                tifffile.imwrite(tiff_file_path, vol_cutout.transpose())

            if config['SKIP_EMBEDDING']: # skip embedding cutouts
                continue

            for model_key in config["MODELS_2_USE"]:
                if model_key == "microns":
                    assert np.array_equal(original_vol_cutout, vol_cutout)
                    embedding = run_embedding(vol_cutout, microns_model)
                else:
                    assert np.array_equal(original_vol_cutout, vol_cutout)
                    embedding = run_embedding(vol_cutout, h01_model)

                pickle_file_path = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'embeddings_'+model_key, embed_key + '.pkl')
                os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
                
                embed_info = {
                    'root_id': root_id,
                    'nm_center_pt': center_pt_nm, 
                    'embedding': embedding,
                    'model_used': model_key
                }
        
                # Save using pickle
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(embed_info, f)