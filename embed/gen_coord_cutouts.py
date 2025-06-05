import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
import caveclient
import cloudvolume
import glob
import json
from meshparty import trimesh_io
import numpy as np
from tqdm import tqdm
import yaml
from embed.utils import * # custom library with util functions
import multiprocessing

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
proofread_seg_spinalcord = cloudvolume.CloudVolume(config["PROOFREAD_SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False)
seg_spinalcord = cloudvolume.CloudVolume(config["SEG_PATH"], use_https=True, mip=config["MIP_SEG_VOL"], agglomerate=True, progress=False, parallel=False)

# CaveClient for spinal cord dataset
if config["CAVECLIENT_DATASTACK"] == None or config["CAVECLIENT_DATASTACK"] == "":
    print("ignoring caveclient setup.")
    client = None
else:
    client = caveclient.CAVEclient(config["CAVECLIENT_DATASTACK"])

# Load MeshParty
mm = trimesh_io.MeshMeta(cv_path=config["SEG_PATH"], disk_cache_path=config["SEG_MESHES"], map_gs_to_https=True)

def generate_root_id_coordinates(root_id, coordinates, synapse_ids, label, config):
    """ Multi-process function to generate coordinate per synapse"""
    try:
        for center_coord, synapse_id in zip(coordinates, synapse_ids):
            # Create graphs and generate nm points to embed
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
                    sk, success = get_root_id_skeleton(seg_id=int(root_id), mm=mm, center_coord=center_coord)

                    if not success:
                        continue # skip if skeletonizing mesh failed

                    # find closest sk node to the provided coordinate
                    closest_sk_node_to_coord = find_closest_sk_node(center_coord, sk, verbose=False)
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

                    # MG = use_best_G(G, [closest_sk_node_to_coord]) #MG = simplify_graph(MG, MINIMUM_DISTANCE = 1500)
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
                    "ground_truth_label":  label, #temp
                    "initial_pt": initial_center_coord, #in 32nm
                    "initial_pt_in_nm": center_coord, # in rw nm
                    "furthest_pt_in_nm": max_dist_from_center_coord,
                    "closest_sk_coord": sk_center_coord,
                    "synapse_id": synapse_id
                }

                os.makedirs(os.path.dirname(json_file), exist_ok=True) # create directories if needed
                with open(json_file, 'w') as f:
                    json.dump(root_id_data, f)

                #print(f"\t{os.path.basename(json_file)} created.")

    except Exception as e:
        print('Error:', e)

    return None # Return something

def generate_root_id_coordinates_wrapper(args):
    return generate_root_id_coordinates(*args)
    

def generate_local_cutout(data_file, config):
    """ Multi-par function to generate local cutouts for each coordinate """
    with open(data_file, 'r') as file:
        root_id_data = json.load(file)

    root_id = root_id_data["root_id"]
    coords_2_embed = root_id_data["nm_coords"]
    initial_pt = root_id_data["initial_pt"] # in 32nm
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
    
    tif_dir = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, 'cutouts')
    os.makedirs(tif_dir, exist_ok=True)

    for center_pt_nm in filtered_coords_2_embed:
        embed_key = f"{str(int(center_pt_nm[0]))}_{str(int(center_pt_nm[1]))}_{str(int(center_pt_nm[2]))}"
        tiff_file_path = os.path.join(tif_dir, embed_key + '.tiff')
        if not os.path.exists(tiff_file_path): # if volume doesn't exist
            vol_cutout, success = get_local_3d_view(center_pt_nm, root_id, config["MIP_EM_VOL"], 
                    config["MIP_SEG_VOL"], config["VOL_MIP"], em_spinalcord, seg_spinalcord, client=None, initial_pt=initial_pt)

            if success:
                if config['SAVE_VOL_CUTOUTS']:
                    tifffile.imwrite(tiff_file_path, vol_cutout.transpose())
    
    return None # Return something

def generate_local_cutout_wrapper(args):
    return generate_local_cutout(*args)


def download_meshes_from_precompute(seg_ids, target_dir):
    seg_mesh_exists = seg_spinalcord.mesh.exists(seg_ids)
    seg_ids_filtered = [seg_id for seg_id, mesh_manifest in zip(seg_ids,seg_mesh_exists) if mesh_manifest != None]
    cv_meshes = seg_spinalcord.mesh.get(seg_ids_filtered)
    for segid, cv_mesh in cv_meshes.items():
        mesh = trimesh_io.Mesh(
            vertices=cv_mesh.vertices,
            faces=cv_mesh.faces,
            process=False,
        )
        trimesh_io.write_mesh_h5(
            f"{target_dir}/{segid}.h5",
            mesh.vertices,
            mesh.faces.flatten(),
            link_edges=mesh.link_edges,
            draco=False,
            overwrite=True,
        )
    return None

def download_meshes_from_precompute_wrapper(args):
    return download_meshes_from_precompute(*args)
    

if __name__ == "__main__":

    for coord, label_name in zip(config["COORD_LIST"], config["LABEL_NAME_LIST"]):
    # if True:
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

        # create own mesh dir for each coordinate. Avoid lock between processes (stale file handle)
        config["SEG_MESHES"] = config["SEG_MESHES"] #+ config["FOLDER_EXT"]
        os.makedirs(config["SEG_MESHES"], exist_ok=True)

        print(f"\n---- GEN_COORD_CUTOUTS.py ----")
        print(f" - {coord}")

        # coordinates converted to seg_vol mip
        root_id_2_coords = create_rootid_2_coord_map(coord, gt_label=gt_label, seg_vol=proofread_seg_spinalcord, flat_seg_vol=seg_spinalcord, 
                                                     output_dir=config["ANNOTS_OUTPUT_DIR"], NG_MIP=config["NG_MIP"], 
                                                     MIP_SEG_VOL=config["MIP_SEG_VOL"], client=client, issynapsecoords=config["USE_SYN_ANNOT_STRUCT"])

        for root_id in root_id_2_coords:
            if root_id == 0:
                raise ValueError("One of the root ids is 0 (background. Fix!")

        # Download all meshes from each segid in volume
        all_seg_ids = [int(seg_id) for seg_id in list(root_id_2_coords.keys())]

        seg_ids_to_download = []
        for seg_id in all_seg_ids:
            if not os.path.exists(os.path.join(config["SEG_MESHES"], f"{seg_id}_0.h5")) and not os.path.exists(os.path.join(config["SEG_MESHES"], f"{seg_id}.h5")):
                seg_ids_to_download.append(seg_id)

        if len(seg_ids_to_download) > 0:
            print(f'Downloading {len(seg_ids_to_download)} meshes ...')
            # trimesh_io.download_meshes(seg_ids=seg_ids_to_download, target_dir=config["SEG_MESHES"], 
            #                         cv_path=config["SEG_PATH"], n_threads=8, verbose=False, progress=False)
            def split_into_chunks(lst, n):
                """Split a list into n approximately equal chunks."""
                k, m = divmod(len(lst), n)
                return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

            seg_ids_chunked = split_into_chunks(seg_ids_to_download, n=10)

            par_args_list = [(seg_id_chunk, config["SEG_MESHES"]) for seg_id_chunk in seg_ids_chunked]
            with multiprocessing.Pool(processes=8) as pool:
                #_ = pool.starmap(generate_root_id_coordinates, par_args_list)
                results = list(tqdm(pool.imap_unordered(download_meshes_from_precompute_wrapper, par_args_list), total=len(par_args_list)))

            
        # Generate arguments list to run in par
        print("Generating coordinates:")
        par_args_list = [(root_id, root_id_2_coords[root_id]['coords'], root_id_2_coords[root_id]['synapse_ids'], root_id_2_coords[root_id]['label'], config) for root_id in root_id_2_coords.keys()]
        with multiprocessing.Pool(processes=8) as pool:
            #_ = pool.starmap(generate_root_id_coordinates, par_args_list)
            results = list(tqdm(pool.imap_unordered(generate_root_id_coordinates_wrapper, par_args_list), total=len(par_args_list)))
        print('Done!\n')

        print("Generating cutouts for each coordinate:")
        DATA_FOLDER_PATH = os.path.join(config["ROOT_SAVE_FOLDER"]+config["FOLDER_EXT"], gt_label, config["DATA_FOLDER"])
        data_file_list = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.json"))
        par_args_list = [(data_file, config) for data_file in data_file_list]
        with multiprocessing.Pool(processes=12) as pool:
            #_ = pool.starmap(generate_local_cutout, par_args_list)
            results = list(tqdm(pool.imap_unordered(generate_local_cutout_wrapper, par_args_list), total=len(par_args_list)))
        print('Done!\n')
