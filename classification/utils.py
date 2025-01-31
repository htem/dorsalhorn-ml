import glob
import os
import json
import random
import pickle
import numpy as np
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt

""" Utils for training an embedding classifier (for spinal cord dataset) """

def agg_embedding_v2(center_pt_data, all_data, radius):
    """ aggregate embeddings given radius (um) """
    
    nm_coord_center_pt = center_pt_data["initial_pt_in_nm"]
    root_id = center_pt_data["root_id"]
    aggregate_embeddings = []

    for di in all_data:
        if str(di["root_id"]) != str(root_id):
            continue

        if di["nm_center_pt"] == nm_coord_center_pt:
            if radius < 2: # return this if true and radius < 2
                return di["embedding"]

        distance_to_center_pt = np.linalg.norm(np.array(di["nm_center_pt"]) - np.array(nm_coord_center_pt)) // 1000 # convert nm to um
  
        if distance_to_center_pt < radius:
            aggregate_embeddings.append(di["embedding"])

    aggregate_embeddings = np.concatenate(aggregate_embeddings, axis=0)
    aggregate_embeddings = np.array([np.mean(aggregate_embeddings, axis=0)])
    return aggregate_embeddings


def create_dataset_embeds(train_data_files, test_data_files, parent_dir, label_map, models2use, CLASS_DIR, AGGREGATE_RADIUS_UM):
    print("Collecting training/testing embeddings ...")
    train_embed, test_embed = {}, {}
    
    print(f"Working with embedding models -> {models2use}")
    for class_n in label_map.keys():
        
        assert class_n in train_data_files
        assert class_n in test_data_files
    
        # get all embedding files from specific model and class
        all_class_pkl_files_dict = {}
        for model_in_use in models2use:
            all_class_pkl_files = glob.glob(os.path.join(CLASS_DIR, class_n, "embeddings_" + model_in_use, "*.pkl"))
            all_class_data = [] # collect all files from specific class
            for pkl_file in all_class_pkl_files:
                pkl_file = os.path.join(parent_dir, pkl_file)
                assert model_in_use in pkl_file
                assert class_n in pkl_file
                with open(pkl_file, 'rb') as file:
                    data = pickle.load(file)
                all_class_data.append(data)
            assert len(all_class_data) > 1 # check to make sure files were correctly collected
            all_class_pkl_files_dict[model_in_use] = all_class_data
    
        class_train_data_files = train_data_files[class_n].copy() # get training data
        embeds = []
        for json_file in class_train_data_files[:]:
            json_file = os.path.join(parent_dir, json_file)
            data_embedding = []
            with open(json_file, 'r') as file:
                center_pt_data = json.load(file)
            for model_in_use in models2use:
                assert class_n in json_file
                aggregated_embedding = agg_embedding_v2(center_pt_data, all_class_pkl_files_dict[model_in_use], AGGREGATE_RADIUS_UM)
                data_embedding.append(aggregated_embedding)
    
            assert len(data_embedding) == len(models2use)
            data_embedding = np.concatenate(data_embedding, axis=1) # concat embeddings
            embeds.append(data_embedding)    
        train_embed[class_n] = np.concatenate(embeds, axis=0) # concat all embeddings to make training set


        class_test_data_files = test_data_files[class_n].copy() # get training data
        embeds = []
        for json_file in class_test_data_files[:]:
            json_file = os.path.join(parent_dir, json_file)
            data_embedding = []
            with open(json_file, 'r') as file:
                center_pt_data = json.load(file)
            for model_in_use in models2use:
                assert class_n in json_file
                aggregated_embedding = agg_embedding_v2(center_pt_data, all_class_pkl_files_dict[model_in_use], AGGREGATE_RADIUS_UM)
                data_embedding.append(aggregated_embedding)
    
            assert len(data_embedding) == len(models2use)
            data_embedding = np.concatenate(data_embedding, axis=1) # concat embeddings
            embeds.append(data_embedding)    
        test_embed[class_n] = np.concatenate(embeds, axis=0) # concat all embeddings to make training set


    return train_embed, test_embed

def setup_embeds_and_labels(embed_data, label_map):
    """ quick function to prepare the embeddings and label arrays """
    all_embeddings, labels = [], []
    for class_n in embed_data.keys():
        labels.extend([label_map[class_n] for _ in range(embed_data[class_n].shape[0])])
        all_embeddings.append(embed_data[class_n])
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(labels)
    return all_embeddings, labels


def convert_file_path_to_model(file_name, model_in_use):
    """ convert file path to needed model in use embedding file """
    if model_in_use in file_name:
        return file_name
    else:
        if 'h01' in file_name:
            assert model_in_use == 'microns'
            return file_name.replace('h01', 'microns')
        elif 'microns' in file_name:
            assert model_in_use == 'h01'
            return file_name.replace('microns', 'h01')
        else:
            raise ValueError(f'Error with file {file_name}')

def visualize_dataset(train_data_files, test_data_files, parent_dir, label_map, only_overall=False):
    """ Visualize traing and testing split """

    class_n_in_use = list(label_map.keys())

    train_synapse_count, train_neurons_count = 0, 0
    seg_ids_used = {}
    for class_n in train_data_files.keys():
        if class_n not in class_n_in_use:
            continue
        seg_ids_used[class_n] = {}
        train_synapse_count += len(train_data_files[class_n])
        for json_file in train_data_files[class_n]:
            json_file = os.path.join(parent_dir, json_file)
            with open(json_file, 'r') as file:
                data = json.load(file)
            if data["root_id"] not in seg_ids_used[class_n]:
                seg_ids_used[class_n][data["root_id"]] = []
            seg_ids_used[class_n][data["root_id"]].append(json_file)
        train_neurons_count += len(seg_ids_used[class_n])

    test_synapse_count, test_neurons_count = 0, 0
    seg_ids_used_test = {}
    for class_n in test_data_files.keys():
        if class_n not in class_n_in_use:
            continue
        seg_ids_used_test[class_n] = {}
        test_synapse_count += len(test_data_files[class_n])
        for json_file in test_data_files[class_n]:
            json_file = os.path.join(parent_dir, json_file)
            with open(json_file, 'r') as file:
                data = json.load(file)
            if data["root_id"] not in seg_ids_used_test[class_n]:
                seg_ids_used_test[class_n][data["root_id"]] = []
            seg_ids_used_test[class_n][data["root_id"]].append(json_file)
        
        test_neurons_count += len(seg_ids_used_test[class_n])
    

    print(f"TOTAL NUMBER OF NEURONS IN TRAIN/TEST DATASET: {train_neurons_count}/{test_neurons_count}")        
    print(f"TOTAL NUMBER OF SYNAPSES/PTS IN TRAIN/TEST DATASET: {train_synapse_count}/{test_synapse_count}")
    
    if only_overall: # return if only showing overall dataset information
        return

    print("Train/Test split breakdown per class:")
    for class_n in seg_ids_used:
        print(class_n)
        print(f"\t# of neurons: {len(seg_ids_used[class_n].keys())}/{len(seg_ids_used_test[class_n].keys())}")
    
        train_embed_count = 0
        for segid in seg_ids_used[class_n]:
            train_embed_count += len(seg_ids_used[class_n][segid])
        test_embed_count = 0
        for segid in seg_ids_used_test[class_n]:
            test_embed_count += len(seg_ids_used_test[class_n][segid])
    
        print(f"\t# of synapses/pts: {train_embed_count}/{test_embed_count}")
            

def create_graphs(train_labels, test_labels, predicted_classes_train, predicted_classes_test, classnames, train_history, added_str=''):
    confmat_norm = sklearn.metrics.confusion_matrix(train_labels, predicted_classes_train, normalize='true')
    confmat = sklearn.metrics.confusion_matrix(train_labels, predicted_classes_train,)

    conf_labeled = []
    for i in range(confmat.shape[0]):
        conf_row = []
        for j in range(confmat.shape[1]):
            conf_row.append(f"{round(confmat_norm[i,j],2)}/{confmat[i,j]}")
        conf_labeled.append(conf_row)
    conf_labeled = np.array(conf_labeled)

    plt.figure(figsize=(14,12))
    sns.set(font_scale=1.5)
    sns.heatmap(confmat_norm, annot=conf_labeled, fmt='', xticklabels=classnames,yticklabels=classnames, cmap='Reds')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.savefig(f'results/train_confmat_{added_str}.png',  bbox_inches='tight')
    
    confmat_norm = sklearn.metrics.confusion_matrix(test_labels, predicted_classes_test, normalize='true')
    confmat = sklearn.metrics.confusion_matrix(test_labels, predicted_classes_test,) 
    
    conf_labeled = []
    for i in range(confmat.shape[0]):
        conf_row = []
        for j in range(confmat.shape[1]):
            conf_row.append(f"{round(confmat_norm[i,j],2)}/{confmat[i,j]}")
        conf_labeled.append(conf_row)
    conf_labeled = np.array(conf_labeled)


    plt.figure(figsize=(14, 12))
    sns.set(font_scale=1.5)
    sns.heatmap(confmat_norm, annot=conf_labeled, fmt='', xticklabels=classnames, yticklabels=classnames, cmap='Reds')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.savefig(f'results/test_confmat_{added_str}.png',  bbox_inches='tight')
    
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].plot(train_history.history['loss'])
    axs[0].plot(train_history.history['val_loss'])
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss')
    axs[0].legend(['Train', 'Val'])
    
    axs[1].plot(train_history.history['sparse_categorical_accuracy'])
    axs[1].plot(train_history.history['val_sparse_categorical_accuracy'])
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend(['Train', 'Val'])
    fig.savefig(f'results/loss_curve_{added_str}.png') 
    print("Created figures.")
    
    
    
def create_neuron_train_test_split(class_dir, class_names, label_map, num_train_neurons, num_test_neurons, use_all_train_neurons=False, verbose=False, save_to_json=False):
    """  Create new training/testing dataset split """

    label_to_classes = {}
    for class_name, label in label_map.items():
        if label not in label_to_classes:
            label_to_classes[label] = []
        label_to_classes[label].append(class_name)

    class_to_root_data = {}
    
    for class_name in class_names:
        if verbose:
            print(f"Working with class: {class_name}")
    
        # collect all JSON files associated with class
        json_files = glob.glob(os.path.join(class_dir, class_name, 'data', "*.json"))
    
        # map json files to segment/root ids (root_id -> json files)
        segment_id_data = {}
        for json_file in json_files:
            with open(json_file, 'rb') as file:
                data = json.load(file)
            if data["root_id"] not in segment_id_data.keys():
                segment_id_data[data["root_id"]] = []
            segment_id_data[data["root_id"]].append(json_file)
    
        all_segment_ids = list(segment_id_data.keys())
        if verbose:
            print(f"\t# of neurons: {len(all_segment_ids)}")
        
        # total number of points/synapses for class
        num_pts = sum(len(lst) for lst in segment_id_data.values())
        if verbose:
            print(f"\t# of synapses: {num_pts}")
    
        class_to_root_data[class_name] = segment_id_data

    train_neurons, test_neurons = {}, {}
    for label, classes in label_to_classes.items():
        # randomly sample from classes belonging to the same label
        neuron_per_class = num_test_neurons // len(classes) # evenly divide # of neurons
        
        if num_test_neurons % len(classes) != 0:
            # if not perfectly divided, distribute remaining neurons
            extra_neurons = num_test_neurons % len(classes)
            for i in range(extra_neurons):
                neurons_in_class = list(class_to_root_data[classes[i]].keys())
                test_neurons[classes[i]] = random.sample(neurons_in_class, neuron_per_class+1)
            for i in range(extra_neurons, len(classes)):
                neurons_in_class = list(class_to_root_data[classes[i]].keys())
                test_neurons[classes[i]] = random.sample(neurons_in_class, neuron_per_class)
        else:
            for class_name in classes:
                neurons_in_class = list(class_to_root_data[class_name].keys())
                test_neurons[class_name] = random.sample(neurons_in_class, neuron_per_class)

        # find which class has the least number of neurons/ or manual train count
        train_min_neuron_count = min(len([neuron for neuron in list(class_to_root_data[class_name].keys()) if neuron not in test_neurons[class_name]]) for class_name in classes)
        train_min_neuron_count = min(train_min_neuron_count, num_train_neurons)

        # sample from each class in list using minimum
        for class_name in classes:
            # sample for train set. Make sure neuron not included in test aswell
            neurons_in_class = list(class_to_root_data[class_name].keys())
            neurons_in_class = [neuron for neuron in neurons_in_class if neuron not in test_neurons[class_name]]
            if use_all_train_neurons:
                train_min_neuron_count = len(neurons_in_class)
            train_neurons[class_name] = random.sample(neurons_in_class, train_min_neuron_count)

    if save_to_json:
        # Save dictionaries into a single file
        training_dataset_dir = 'datasets'
        train_test_split_file = os.path.join(training_dataset_dir, 'train_test_split_neurons.json')
        with open(train_test_split_file, 'w') as f:
            json.dump({'class_to_root_data': class_to_root_data, 'train_neurons': train_neurons, 'test_neurons': test_neurons}, f)
        print(f'Created train_test_split file: {train_test_split_file}')
    return class_to_root_data, train_neurons, test_neurons


def convert_neurons_to_embedding_files(train_neurons, test_neurons, class_to_root_data, use_all_synapses=False, use_all_synapses_for_test=False, verbose=False):
    """ convert the list of neurons to the embedding files """

    train_data_files, test_data_files = {}, {}

    for class_name in class_to_root_data.keys():

        bucket_train_files = []
        for neuron_id in train_neurons[class_name]:
            neuron_synapse_list = class_to_root_data[class_name][neuron_id]
            if not use_all_synapses:
                neuron_synapse_list = [neuron_synapse_list[0]]
            bucket_train_files.extend(neuron_synapse_list)
        random.shuffle(bucket_train_files)
        train_data_files[class_name] = bucket_train_files
    
        bucket_test_files = []
        for neuron_id in test_neurons[class_name]:
            neuron_synapse_list = class_to_root_data[class_name][neuron_id]
            if not use_all_synapses_for_test:
                neuron_synapse_list = [neuron_synapse_list[0]]
            bucket_test_files.extend(neuron_synapse_list)
        random.shuffle(bucket_test_files)
        test_data_files[class_name] = bucket_test_files

        if verbose:
            print(f'Class: {class_name}')
            print(f"\t# of train embeddings: {len(bucket_train_files)}")
            print(f"\t# of test embeddings: {len(bucket_test_files)}")

    # save training dataset file
    training_dataset_dir = 'datasets'

    train_embeds_sum = sum(len(embeds) for embeds in train_data_files.values())
    test_embeds_sum = sum(len(embeds) for embeds in test_data_files.values())
    use_all_syn = 'useallsyn' if use_all_synapses else 'onlysinglesyn'
    file_name_ext = f'_{train_embeds_sum}_{test_embeds_sum}_{use_all_syn}.pkl'
    new_training_dataset_file = os.path.join(training_dataset_dir, 'training_embeds'+file_name_ext) 
    with open(new_training_dataset_file, 'wb') as f:
        pickle.dump(train_data_files, f)
    print(f'Created train file: {new_training_dataset_file}')

    # save testing dataset file
    new_testing_dataset_file = os.path.join(training_dataset_dir, 'testing_embeds'+file_name_ext) 
    with open(new_testing_dataset_file, 'wb') as f:
        pickle.dump(test_data_files, f)
    print(f'Created test file: {new_testing_dataset_file}')

    return train_data_files, test_data_files
