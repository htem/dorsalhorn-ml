import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
sys.path.append(repo_dir + '/connectomics')
sys.path.append(repo_dir + '/simclr')

import caveclient
import cloudvolume
import glob
import json
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
import yaml
import tifffile

from prepare_cutouts.utils import * # custom library with util functions

# read configuration file (config.yaml)
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Load cloudvolume for em and segmentation layers
em_spinalcord = cloudvolume.CloudVolume(config["EM_PATH"], use_https=True, fill_missing=True, mip=config["MIP_EM_VOL"], progress=False)
seg_spinalcord = cloudvolume.CloudVolume(config["SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False)

# CaveClient for spinal cord dataset
client = caveclient.CAVEclient(config["CAVECLIENT_DATASTACK"])

MIN_DISTANCE_FROM_PAIR = 10 #um

os.makedirs('data', exist_ok=True)


if __name__ == "__main__":
    ########################################################################################################################

    for class_name in config["CLASS_NAMES"]:
        print("Working with class:", class_name)

        DATA_FOLDER_PATH = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], class_name, config["DATA_FOLDER"])
        data_file_list = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.json"))
        num_of_files_in_list = len(data_file_list)


        for fi, data_file in enumerate(tqdm(data_file_list, "Generating cutouts")):
            with open(data_file, 'r') as file:
                root_id_data = json.load(file)

            root_id = root_id_data["root_id"]
            gt_label = root_id_data['ground_truth_label']
            coords_2_embed = root_id_data["nm_coords"]
            initial_pt = root_id_data["initial_pt"] # in 32nm
            initial_pt_in_nm = root_id_data["initial_pt_in_nm"]

            # find closest point that is MIN_DISTANCE_FROM_PAIR away. initial pair is always a synapse
            filtered_coords_2_embed = []
            distances = []
            for center_pt_nm in coords_2_embed:
                euc_dist = np.linalg.norm(initial_pt_in_nm - np.array(center_pt_nm))
                distances.append([center_pt_nm, euc_dist])

            closest_pt, closest_distance = min(distances, key=lambda x: abs(x[1] - MIN_DISTANCE_FROM_PAIR*1000))
            filtered_coords_2_embed = [initial_pt_in_nm, closest_pt]

            for center_pt_nm in filtered_coords_2_embed:
                cutout_key = f"{str(int(center_pt_nm[0]))}_{str(int(center_pt_nm[1]))}_{str(int(center_pt_nm[2]))}"

                em_vol_cutout, seg_vol_cutout, success = get_local_3d_view(center_pt_nm, root_id, config["MIP_EM_VOL"], 
                        config["MIP_SEG_VOL"], config["VOL_MIP"], em_spinalcord, seg_spinalcord, client, initial_pt=initial_pt, get_mask=True)
                
                if not success:
                    continue

                if np.all(em_vol_cutout==0):
                    raise ValueError("3D volume is empty.")
                
                store_dir = os.path.join('data', gt_label, f'pair_{fi}')
                os.makedirs(store_dir, exist_ok=True)
                em_vol_file_name = os.path.join(store_dir, cutout_key+'_em.tiff')
                seg_vol_file_name = os.path.join(store_dir, cutout_key+'_seg.tiff')

                tifffile.imwrite(em_vol_file_name, em_vol_cutout.transpose())
                tifffile.imwrite(seg_vol_file_name, seg_vol_cutout.transpose())
        