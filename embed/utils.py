import cloudvolume
import cv2
import glob
import json
import numpy as np
from meshparty import skeletonize # meshparty 1.16.14
from meshparty import trimesh_io
import networkx as nx
import os
import tifffile
from tqdm import tqdm
import pandas as pd

""" Helper functions for segclr """

def get_seg_id_from_coord(coord, seg_vol, seg_mip=(32,32,45), coordinate_mip=(32,32,45)):
    """ get segid from coordinate """

    mip_scale = np.array(seg_mip)/np.array(coordinate_mip)
    seg_coord = (coord[0]//mip_scale[0], coord[1]//mip_scale[1], coord[2]//mip_scale[2])

    voxel_seg_id = seg_vol[seg_coord][0][0][0][0] # segmentation ID of neuron coordinate
    return voxel_seg_id


def load_annots(file_path, gt_label=None):
    """ load annotations and parse """
    with open(file_path, 'r') as file:
        data = json.load(file)
    final_list = []
    for line in data['annotations']:
        centroid_coord = line["point"] # single point
        final_list.append({"coord": centroid_coord, "label": gt_label})
    return final_list


def load_man_synapse_annots(file_path, use_desc_gt = False, use_global_gt = False, gt_label=None, use_point_key = "pointB"):
    """ load annotations from ng. pointA(presyn) and pointB(postsyn). if use_desc_gt, use des as gt name (must all match some structure """
    with open(file_path, 'r') as file:
        data = json.load(file)

    if use_global_gt:
        global_gt_label = gt_label
    else:
        global_gt_label = None

    final_list = []
    for line in data["annotations"]:
        presyn_coord = line[use_point_key] #point A should be presyn
        if use_desc_gt:
            if "description" in line:
                # use some mapping if needed
                gt_label = line["description"]
            else:
                raise ValueError(f"Description key missing for {line}")
        else:
            if use_global_gt:
                gt_label = global_gt_label
            else:
                gt_label = None
        final_list.append({"coord": presyn_coord, "label": gt_label})
    return final_list


def create_post_synapse_data_from_synapse_table(post_synaptic_neuron_root_id, client, min_size=20, verbose=True):
    """ Generating post synaptic dataframe and creating annot list of coordinates """

    if verbose:
        print(f'Working with post synaptic neuron: {post_synaptic_neuron_root_id}')

    # Get information from post synaptic neuron
    post_synapse_df = client.materialize.query_table('synapses_v2', filter_equal_dict = {'post_pt_root_id': post_synaptic_neuron_root_id})
    post_synapse_df = post_synapse_df[post_synapse_df['size']>=min_size] # Filter by size
    post_synapse_df = post_synapse_df[post_synapse_df['pre_pt_root_id'] != post_synapse_df['post_pt_root_id']] # Remove synapses onto same neuron

    # Get presynapse coordinates from post synapse neuron
    presynapse_nm_coordinates = []
    for i, row in post_synapse_df.iterrows():
        pre_pt_position = row['pre_pt_position']
        pre_pt_position[0] =  pre_pt_position[0]//32
        pre_pt_position[1] =  pre_pt_position[1]//32
        pre_pt_position[2] = pre_pt_position[2]//45

        # NOTE: From Wangchu. Any synapses with z-coordinates > 2900 can be ignored
        if pre_pt_position[2] > 2900: # at 45nm resolution
            continue
        
        pre_pt_position = [int(x) for x in pre_pt_position]

        presynapse_nm_coordinates.append({
            "synapse_id": str(row["id"]), # synapse id
            "coord": pre_pt_position,
            "label": None
        })

    if verbose:
        print(f'Total number of presynapse coordinates: {len(presynapse_nm_coordinates)}')

    return presynapse_nm_coordinates


def create_rootid_2_coord_map(annot_json_file, gt_label=None, seg_vol=None, flat_seg_vol=None, output_dir=None, 
NG_MIP=None, MIP_SEG_VOL=None, client=None, issynapsecoords=False):
    """ Create a rootid 2 coord mapping. If file already exists, ignore """

    # scale to segmentation volume coordinates from ng annot coordinates
    scale_factor = [MIP_SEG_VOL[0]//NG_MIP[0], MIP_SEG_VOL[1]/NG_MIP[1], MIP_SEG_VOL[2]//NG_MIP[2]]
    use_synapse_table = False

    if ".json" in annot_json_file:
        annot_file_name = os.path.basename(annot_json_file).split(".")[0]
        output_file_path = os.path.join(output_dir, annot_file_name + "_" + "rootid_2_coords.json")
    else:
        print("Assuming this annot is coordinate")
        coord = annot_json_file
        seg_id = get_root_id_from_coord(coord, vol=seg_vol) # use proofread volume
        seg_id = str(seg_id)

        # convert coord to string
        coord_str = f"{str(coord[0])}_{str(coord[1])}_{str(coord[2])}"
        output_file_path = os.path.join(output_dir, coord_str + "_" + "rootid_2_coords.json")
        use_synapse_table = True

    if os.path.exists(output_file_path):
        print("Rootid 2 coord mapping already exists. Using this data.")
        with open(output_file_path, 'r') as file:
            root_id_2_coords = json.load(file)
        return root_id_2_coords

    if gt_label:
        use_global_gt = True
    else:
        use_global_gt = False

    if use_synapse_table:
        annot_data = create_post_synapse_data_from_synapse_table(seg_id, client=client)
    else:
        if issynapsecoords:
            annot_data = load_man_synapse_annots(annot_json_file, use_desc_gt = False, use_global_gt = use_global_gt, gt_label=gt_label)
        else:
            annot_data = load_annots(annot_json_file, gt_label=gt_label)

    root_id_2_coords = {}
    print("Creating root_id -> coordinates mapping w/ grouth truth labels ...")
    for coord_data in tqdm(annot_data):
        coord = coord_data["coord"]
        coord = [coord[0]//scale_factor[0], coord[1]//scale_factor[1], coord[2]//scale_factor[2]] 
        root_id = get_root_id_from_coord(coord, vol=flat_seg_vol)

        if 'synapse_id' in coord_data:
            synapse_id = coord_data['synapse_id']
        else:
            synapse_id = None

        if str(root_id) not in root_id_2_coords:
            root_id_2_coords[str(root_id)] = {'coords': [], 'label': coord_data['label'], 'synapse_ids': []}
        root_id_2_coords[str(root_id)]['coords'].append(coord)
        root_id_2_coords[str(root_id)]['synapse_ids'].append(synapse_id)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        json.dump(root_id_2_coords, outfile)

    return root_id_2_coords


def downsample_vol(vol, size):
    """ downsample volume """
    downsampled_vol = []
    for z in range(vol.shape[2]):
        slice_resized = cv2.resize(vol[:,:,z].astype(np.uint8), size, interpolation=cv2.INTER_AREA)
        downsampled_vol.append(slice_resized[:,:,np.newaxis])

    downsampled_vol = np.concatenate(downsampled_vol, axis=2)
    return downsampled_vol


def get_local_3d_view(center_pt_nm, segment_id, em_mip, seg_mip, exp_mip, em_vol, seg_vol, client = None, initial_pt = None, scale_factor=[1,1,1]):
    """ get local 3d volume cutout masked by segment id 
    em_mip: EM volume mip
    seg_mip: Segmentation volume mip
    exp_mip: Expected mip before embedding
    """

    #em_vol = cloudvolume.CloudVolume(em_vol, use_https=True, fill_missing=True, mip=(32,32,45), progress=False)
    #seg_vol = cloudvolume.CloudVolume(seg_vol, use_https=True, mip=(32,32,45), agglomerate=True, progress=False, parallel=True)

    if client != None:
        # update segment id using the initial_pt since proofreading is ongoing ... (08/30/2024)
        segment_id = int(segment_id) # convert segment_id to integer
        if not client.chunkedgraph.is_latest_roots([segment_id])[0]:
            print("Switching seg_id")
            potential_new_seg_id = seg_vol[int(initial_pt[0]), int(initial_pt[1]), int(initial_pt[2])][0][0][0][0]
            segment_id = potential_new_seg_id

    BBOX_SIZE = 129 # voxel size assuming resolution is 32x32x45nm
    segment_id = int(segment_id)

    # get bounds for cloudvolumes
    em_min, em_max = list(em_vol.bounds.minpt), list(em_vol.bounds.maxpt)
    seg_min, seg_max = list(seg_vol.bounds.minpt), list(seg_vol.bounds.maxpt)

    # change range for zdim (since both are always 45nm/voxel
    em_min[-1], em_max[-1] = seg_min[-1], seg_max[-1]

    # em # convert from nm to em_dim coordinates
    center_pt = np.array([center_pt_nm[0]//em_mip[0], center_pt_nm[1]//em_mip[1], center_pt_nm[2]//em_mip[2]]) 
    bbox_scale = [exp_mip[i]//em_mip[i] for i in range(len(em_mip))]

    minb = [center_pt[i] - ((BBOX_SIZE//2) * bbox_scale[i]) for i in range(len(em_mip))]
    maxb = [center_pt[i] + ((BBOX_SIZE//2 + 1) * bbox_scale[i]) for i in range(len(em_mip))]

    shift_pads = [[0,0], [0,0], [0,0]]
    for i in range(len(maxb)):
        if maxb[i] > em_max[i]:
            shift_pads[i][1] = int(maxb[i] - em_max[i])
            maxb[i] = em_max[i]
        if minb[i] < em_min[i]:
            shift_pads[i][0] = int(em_min[i] - minb[i])
            minb[i] = em_min[i]
    vol_cutout = em_vol.download(cloudvolume.Bbox(minb, maxb),)
    vol_cutout = vol_cutout[:,:,:,0]

    if vol_cutout.shape != tuple(BBOX_SIZE*bbox_scale[i] for i in range(3)): #(BBOX_SIZE, BBOX_SIZE, BBOX_SIZE):
        print(shift_pads)
        vol_w_pads = np.pad(vol_cutout,
                                pad_width=shift_pads,
                                mode='constant',
                                constant_values=0)
        vol_cutout = vol_w_pads

    # downsample if needed
    if exp_mip != em_mip:
        vol_cutout = downsample_vol(vol_cutout, (BBOX_SIZE, BBOX_SIZE))

    # segmentation
    center_pt = np.array([center_pt_nm[0]//seg_mip[0], center_pt_nm[1]//seg_mip[1], center_pt_nm[2]//seg_mip[2]]) 
    bbox_scale = [exp_mip[i]//seg_mip[i] for i in range(len(seg_mip))]

    minb = [center_pt[i] - ((BBOX_SIZE//2) * bbox_scale[i]) for i in range(len(seg_mip))]
    maxb = [center_pt[i] + ((BBOX_SIZE//2 + 1) * bbox_scale[i]) for i in range(len(seg_mip))]

    shift_pads_seg = [[0,0], [0,0], [0,0]]
    for i in range(len(maxb)):
        if maxb[i] > seg_max[i]:
            shift_pads_seg[i][1] = int(maxb[i] - seg_max[i])
            maxb[i] = seg_max[i]
        if minb[i] < seg_min[i]:
            shift_pads_seg[i][0] = int(seg_min[i] - minb[i])
            minb[i] = seg_min[i]
    vol_cutout_seg = seg_vol.download(cloudvolume.Bbox(minb, maxb),)
    vol_cutout_seg = vol_cutout_seg[:,:,:,0]

    if vol_cutout_seg.shape != (BBOX_SIZE, BBOX_SIZE, BBOX_SIZE):
        print(shift_pads_seg)
        vol_w_pads = np.pad(vol_cutout_seg,
                                pad_width=shift_pads_seg,
                                mode='constant',
                                constant_values=0)
        vol_cutout_seg = vol_w_pads

    # downsample if needed
    if exp_mip != seg_mip:
        vol_cutout_seg = downsample_vol(vol_cutout_seg, BBOX_SIZE)

    # create mask using segment id
    # print(segment_id, np.unique(vol_cutout_seg))
    seg_mask = vol_cutout_seg[:,:,:] == segment_id

    # z-normalization of cutout before masked
    #vol_cutout = vol_cutout.astype(np.float32) / 255.0
    #vol_cutout = (vol_cutout - vol_cutout.mean()) / (vol_cutout.std() + 1e-8)

    masked_em_vol = vol_cutout.copy()
    masked_em_vol[~seg_mask] = 0

    # save 3d vol if it doesnt meet BBOX_SIZE shape
    if masked_em_vol.shape != (BBOX_SIZE, BBOX_SIZE, BBOX_SIZE):
        padding = [(0,BBOX_SIZE-ax_size) for ax_size in masked_em_vol.shape]
        if np.max(padding) >= 30:
            print(f"(WARNING) Padding is larger than 30. {np.max(padding)}")
        if np.max(padding) >= 100:
            raise IndexError(f"centroid is too far on the edge of volume. {np.max(padding)}")
        masked_em_vol = np.pad(masked_em_vol, padding, mode='constant', constant_values=0)

    # DEBUG PURPOSE
    #tifffile.imwrite('test.tif', masked_em_vol.transpose())
    #raise ValueError("Stopping early")

    assert masked_em_vol.shape == (BBOX_SIZE, BBOX_SIZE, BBOX_SIZE)
    return masked_em_vol, True


def run_embedding(vol_cutout, model):
    """ embed 3d volume """
    input_data = vol_cutout[:,:,:,np.newaxis].copy()
    input_data = input_data.astype(np.float32) / 255.0
    input_data = input_data.transpose(2, 1, 0, 3)  # The model expects axes in ZYX order.
    input_data = np.reshape(input_data, (1, 129, 129, 129, 1))
    embeddings = model(input_data, training=False)
    embedding = np.array(embeddings[0])
    return embedding

def run_embedding_batched(batch_vol_cutouts, model):
    """ embed 3d volume """
    # The model expects axes in ZYX order.
    embeddings = model(batch_vol_cutouts, training=False)
    embeddings = np.array(embeddings[0])
    #print(embeddings.shape)
    return embeddings


def get_root_id_from_coord(coord, vol):
    """ using a xyz coordinate, find the voxel id number associated """
    voxel_root_id = vol[coord][0,0,0,0]
    return voxel_root_id


def filter_meshes(mesh, center_nm, radius_nm):
    vertices = mesh.vertices
    faces = mesh.faces

    distances = np.linalg.norm(vertices - center_nm, axis=1)

    vertex_mask = distances < radius_nm
    selected_indices = np.where(vertex_mask)[0]

    old_to_new = {old: new for new, old in enumerate(selected_indices)}

    face_mask = np.all(np.isin(faces, selected_indices), axis=1)
    filtered_faces = faces[face_mask]

    remapped_faces = np.vectorize(old_to_new.get)(filtered_faces)

    cropped_mesh = trimesh_io.Mesh(vertices=vertices[selected_indices], faces=remapped_faces)

    return cropped_mesh


def get_root_id_skeleton(seg_id, mm, verbose=False, force_download=False, center_coord=[0,0,0], vertice_thresh=100):
    """ create skeleton using meshparty """

    if force_download:
        neuron_mesh = mm.mesh(seg_id=seg_id)
        if verbose:
            print("Mesh downloaded")

    potential_mesh_file_names = [
        os.path.join(mm.disk_cache_path, str(seg_id) + ".h5"),
        os.path.join(mm.disk_cache_path, str(seg_id) + "_0.h5"),                 
    ]

    mesh_file_exists = False
    for mesh_file_path in potential_mesh_file_names:
        if os.path.exists(mesh_file_path):
            expected_cache_path = mesh_file_path
            mesh_file_exists = True
        break

    if not mesh_file_exists:
        neuron_mesh = mm.mesh(seg_id=seg_id)
    else:
        vertices, faces, normals, link_edges, node_mask = trimesh_io.read_mesh(expected_cache_path)

        if node_mask == None:
            neuron_mesh = trimesh_io.Mesh(vertices=vertices, faces=faces, link_edges=link_edges, normals=normals,
                        node_mask=node_mask)
        else:
            neuron_mesh = trimesh_io.Mesh(vertices=vertices, faces=faces, link_edges=link_edges, normals=normals,
                                node_mask=node_mask[:vertices.shape[0]])
        
    if neuron_mesh.n_vertices < vertice_thresh: # mesh is too small. Ignore
        return None, False
    
    # Get local mesh
    #neuron_mesh = neuron_mesh.get_local_mesh(max_dist=10000, center_coord=center_coord)
    neuron_mesh = filter_meshes(neuron_mesh, center_coord, 15000)

    # Skeletonize
    sk = skeletonize.skeletonize_mesh(neuron_mesh, invalidation_d=12000, compute_radius=False, verbose=True) # skeletonize

    return sk, True


def find_closest_sk_node(coordinate, sk, verbose=False):
    """ Using coordinate and mip, find closest skeletone node to coordinate """
    nm_coord = np.array(coordinate)

    # get euc dist between nm_coord and all other vertices in skeleton
    dif_in_nodes = np.linalg.norm(sk.vertices - nm_coord, axis=1)

    # find shortest distance
    closest_vertex_idx = np.argmin(dif_in_nodes)
    if verbose:
        print("Distance to closest SK node(nm):", round(dif_in_nodes[closest_vertex_idx], 2))
    return closest_vertex_idx # return vertex idx


def create_graph_w_filter(sk, valid_sk_nodes, CONNECT_MIN = 1000, verbose=False):
    """ Create graph of skeleton nodes """
    G = nx.Graph() # create initial graph
    for i in range(len(sk.edges)):
        idx1, idx2 = sk.edges[i]
        if idx1 in valid_sk_nodes and idx2 in valid_sk_nodes:
            euc_dist = np.linalg.norm(sk.vertices[idx1] - sk.vertices[idx2]) #euclidian distance between skeleton nodes ... in nm
            G.add_edge(idx1, idx2, weight=euc_dist)
    return G


def get_simplified_graphs(G, MINIMUM_DISTANCE=1500):
    """ simplify all graphs/subgraphs """
    if nx.is_connected(G):
        return [simplify_graph(G, MINIMUM_DISTANCE = MINIMUM_DISTANCE)]
    else:
        # keep subgraph with highest # of nodes
        components_list = list(nx.connected_components(G))

        graph_list = []
        for component in components_list:
            G_sub = G.subgraph(component).copy()
            G_sub = simplify_graph(G_sub, MINIMUM_DISTANCE = MINIMUM_DISTANCE)
            graph_list.append(G_sub)
        return graph_list

def use_best_G(G, important_nodes):
    """ if graph is disconnected, use subgraph with the important nodes """
    if nx.is_connected(G):
        return G
    else:
        # keep subgraph with highest # of nodes
        components_list = list(nx.connected_components(G))
        for component in components_list:
            is_in = True
            for im_n in important_nodes:
                if im_n not in component:
                    is_in = False
            if is_in:
                return G.subgraph(component).copy()
        raise ValueError(f"Important nodes {important_nodes} not found for any subgraphs!")


def simplify_graph(G, MINIMUM_DISTANCE = 1500, verbose=False):
    """ Simplify the graph provided, minimizing distance between nodes to {MINIMUM_DISTANCE} """
    # gather leaf nodes
    leaf_nodes_list = [node for node in G.nodes() if G.degree(node) == 1]
    if len(leaf_nodes_list) <= 1:
        raise ValueError("Error: Cannot simplify graph with less than 2 leaf nodes.""")

    source = leaf_nodes_list[0]

    if verbose:
        print(f"\tLeaf nodes found: {leaf_nodes_list}")
    
    subgraph_list = []
    for leaf_node in leaf_nodes_list:
        if source == leaf_node:
            continue # ignore if leaf node is source
        path = nx.shortest_path(G, source, leaf_node) # find shortest path from source to leaf

        distance_cum = 0
        prev_node = prev_anchor = source 

        G_sub = nx.Graph() # create initial graph
        for node in path[1:]:
            weight = G.get_edge_data(prev_node, node)['weight']
            prev_node = node

            distance_cum += weight
            if distance_cum >= MINIMUM_DISTANCE: # minimum reached
                G_sub.add_edge(prev_anchor, node, weight=distance_cum)
                prev_anchor = node
                distance_cum = 0
        
        if distance_cum > 0: # add leaf node portion if min distance wasn't met
            G_sub.add_edge(prev_anchor, path[-1], weight=distance_cum)
        subgraph_list.append(G_sub)

    MG = nx.Graph()
    for graph in subgraph_list:
        MG = nx.compose(MG, graph)

    if verbose:
        print(f"\tSimplified graph! # of nodes for graph now: {MG.number_of_nodes()}")
    return MG
