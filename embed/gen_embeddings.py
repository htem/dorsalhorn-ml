import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
sys.path.append(repo_dir + '/connectomics')
sys.path.append(repo_dir + '/simclr')
from connectomics.segclr.tf2 import legacy_model
import cloudvolume
import glob
import json
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
import yaml
from embed.utils import * # custom library with util functions
import concurrent.futures

# Read config file
parser = argparse.ArgumentParser(description="Run script with a config file")
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--coord", type=str, required=False, help="Optional soma coordinate")
args = parser.parse_args()

with open(args.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Soma coordinate
use_soma_coord = False
if args.coord is not None:
    SOMA_COORD = args.coord
    print("COORDINATE USED:", SOMA_COORD)
    use_soma_coord = True
else:
    print("Ignoring soma coordinate. Using list from config.")

# Load cloudvolume for em and segmentation layers
em_spinalcord = cloudvolume.CloudVolume(config["EM_PATH"], use_https=True, fill_missing=True, mip=config["MIP_EM_VOL"], progress=False)
seg_spinalcord = cloudvolume.CloudVolume(config["SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False, parallel=False)

# Create embedding model instances
models_2_use_instances = {}
for model_key in config["MODELS_2_USE"]:
    model_path = config["MODELS_2_USE"][model_key]
    model = legacy_model.LegacySegClrModel()
    input = tf.keras.Input(shape=config["INPUT_SHAPE"], dtype=tf.float32)
    output = model(input, training=False)
    ckpt = tf.train.Checkpoint(model)
    _ = ckpt.restore(model_path).expect_partial()
    models_2_use_instances[model_key] = model


def save_embed_info(args):
    """ Multi-thread function to save embedding information """
    root_id, model_key, embed_info, config = args
    model_key_dir = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'embeddings_'+model_key)
    pickle_file_path = os.path.join(model_key_dir, f"{root_id}_embeddings.pkl")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(embed_info, f)


def pull_cutout_info(args):
    """ Multi-thread function to pull data from JSON files """
    data_file, config = args # args path to json data file and config
    with open(data_file, 'r') as file:
        root_id_data = json.load(file)

    root_id = root_id_data["root_id"]
    coords_2_embed = root_id_data["nm_coords"]
    #initial_pt = root_id_data["initial_pt"] # in 32nm
    initial_pt_in_nm = root_id_data["initial_pt_in_nm"]

    # filter coordinates
    filtered_coords_2_embed = []
    if config["EMBED_AGG_RADIUS_UM"] < 5:
        filtered_coords_2_embed.append(initial_pt_in_nm)
    else:
        for center_pt_nm in coords_2_embed:
            euc_dist = np.linalg.norm(initial_pt_in_nm - np.array(center_pt_nm))
            if euc_dist <= config["EMBED_AGG_RADIUS_UM"] *1000:
                filtered_coords_2_embed.append(center_pt_nm)

    pickle_file_path = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'embeddings_'+model_key, str(root_id) + '_embeddings.pkl')
    if os.path.exists(pickle_file_path):
        return [],[],[]

    local_tif_file_paths, local_center_coords_used, local_root_ids = [], [], []
    for center_pt_nm in filtered_coords_2_embed:
        embed_key = f"{str(int(center_pt_nm[0]))}_{str(int(center_pt_nm[1]))}_{str(int(center_pt_nm[2]))}"

        if embed_key == "1034222_143445_54452": #TODO: fix this. bug with this single volume
            continue

        #embed_done = True
        #for model_key in config["MODELS_2_USE"]:
        #    pickle_file_path = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'embeddings_'+model_key, embed_key + '.pkl')
        #    if not config["OVERWRITE_FILES"]: # skip if file for coordinate already exists
        #        if not os.path.exists(pickle_file_path):
        #            embed_done = False
        #if embed_done:
        #    continue #TODO: Update this if we don't want to overwrite embeddings files

        tiff_file_path = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'cutouts', embed_key + '.tiff')
        if os.path.exists(tiff_file_path): # if volume cutout already exists
            if config['SKIP_EMBEDDING']:
                continue
            else:
                success = True
        else:
            raise ValueError("All volume cututs should have been generated. Report this")
        if not success:
            continue

        local_tif_file_paths.append(tiff_file_path)
        local_center_coords_used.append(center_pt_nm)
        local_root_ids.append(root_id)

    return local_tif_file_paths, local_center_coords_used, local_root_ids

def load_tif(filename):
    """ Tensorflow intermediate function to run per batch """
    input_data = tifffile.imread(filename.numpy().decode('utf-8'))
    try:
       input_data = input_data[:,:,:,np.newaxis].copy()
       input_data = input_data.astype(np.float32) / 255.0
    except:
       print("Error with", filename)
    return input_data

def load_wrapper(filename):
    input_data = tf.py_function(load_tif, [filename], tf.float32)
    input_data.set_shape([None, None, None, 1])  # Set shape manually if possible
    return input_data


if __name__ == "__main__":
    global_gt_label = 'inference' # meant for inference

    #if True:
    for coord, label_name in zip(config["COORD_LIST"], config["LABEL_NAME_LIST"]):
        if config["IS_TRAIN"]:
            gt_label = label_name
            config["FOLDER_EXT"] = f"_unnorm/{gt_label}"
        else:
            gt_label = config["GLOBAL_GT_INFFERENCE"] # overwrite

            if use_soma_coord:
                coord = [int(pt) for pt in SOMA_COORD.split("_")] # split string and convert each to integer
                config["FOLDER_EXT"] = f"_{str(coord[0])}_{str(coord[1])}_{str(coord[2])}" # add folder_ext
            else:
                # using config provided coord
                config["FOLDER_EXT"] = f"_{label_name}_axoaxonic"

        print(f"\n---- GEN_EMBEDDINGS.py ----")
        print("- config folder_ext =", config["FOLDER_EXT"])

        DATA_FOLDER_PATH = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, config["DATA_FOLDER"])
        data_file_list = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.json"))
        par_args_list = [(data_file, config) for data_file in data_file_list]

        print("Pulling cutout information:")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            cutouts_infos = list(tqdm(executor.map(pull_cutout_info, par_args_list), total=len(par_args_list)))

        tif_file_paths, center_coords_used, root_ids = [], [], []
        for res in cutouts_infos:
            tif_file_paths.extend(res[0])
            center_coords_used.extend(res[1])
            root_ids.extend(res[2])

        # root_ids_masked = np.array(root_ids) == '76210312822485624'
        #tif_file_paths = np.array(tif_file_paths) #[root_ids_masked]
        #center_coords_used = np.array(center_coords_used) #[root_ids_masked]
        #root_ids = np.array(root_ids) #[root_ids_masked]

        #tif_file_paths = tif_file_paths[60000:]
        #center_coords_used = center_coords_used[60000:]
        #root_ids = root_ids[60000:]

        # tif_file_path = '../dataset/embeddings_21132_6350_1717/inference/cutouts/608173_187368_75605.tiff'
        # # Using your py_function version
        # a = load_tif(tf.convert_to_tensor(tif_file_path))
        # print(a.shape)

        # # Using TF-only version
        # b = load_wrapper_tf(tf.convert_to_tensor(tif_file_path))
        # print(b.shape)
        # # Evaluate (if inside eager mode or session)
        # np.testing.assert_allclose(a, b)
        # break

        print("Number of local cutouts(tifs) to embed:", len(tif_file_paths))
        dataset = tf.data.Dataset.from_tensor_slices(tif_file_paths)
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        batch_size = config["INF_BATCH_SIZE"]
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        n_batches = len(tif_file_paths) // batch_size + int(len(tif_file_paths) % batch_size != 0)

        print("Running inference:")
        embeddings_per_model_key = {model_key: [] for model_key in config["MODELS_2_USE"]}
        for batch in tqdm(dataset, total=n_batches):
        # for i,tif_file_path in enumerate(tif_file_paths):
        #     vol_cutout = tifffile.imread(tif_file_path).transpose()
            for model_key in config["MODELS_2_USE"]:
        #         embeddings = run_embedding(vol_cutout, models_2_use_instances[model_key])
        #         if tif_file_path == '../dataset/embeddings_21132_6350_1717/inference/cutouts/608173_187368_75605.tiff':
        #             print(tif_file_path, center_coords_used[i], root_ids[i])
        #             print(model_key, vol_cutout.shape)
        #             print(embeddings)
                embeddings = run_embedding_batched(batch, models_2_use_instances[model_key])
                embeddings_per_model_key[model_key].append(embeddings)

        print("Organizing results:")
        embeddings_info_save = {}
        for model_key in config["MODELS_2_USE"]:
            model_embeddings = embeddings_per_model_key[model_key]
            model_embeddings = np.concatenate(model_embeddings, axis=0)
            model_key_dir = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'embeddings_'+model_key)
            os.makedirs(model_key_dir, exist_ok=True)

            print(f'- for {model_key}')
            for ei in range(model_embeddings.shape[0]):
                root_id = root_ids[ei]
                center_pt_nm = center_coords_used[ei]
                info_key = (root_id, model_key)
                if info_key not in embeddings_info_save:
                    embeddings_info_save[info_key] = {
                        'root_id': root_id,
                        'nm_center_pts_2_embeddings': {},
                        'model_used': model_key
                    }
                embeddings_info_save[info_key]['nm_center_pts_2_embeddings'][tuple(center_pt_nm)] = model_embeddings[ei]

        save_args = []
        for root_id, model_key in embeddings_info_save:
            save_args.append((root_id, model_key, embeddings_info_save[(root_id, model_key)], config))
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(save_embed_info, save_args), total=len(save_args), desc=f"Saving embedding data"))
        
        print("\nDone!")


