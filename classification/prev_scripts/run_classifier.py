import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
sys.path.append(repo_dir + '/connectomics')
sys.path.append(repo_dir + '/simclr')

from connectomics.segclr.tf2 import legacy_model
from connectomics.segclr.classification import model_configs
from connectomics.segclr.classification import model_handler
import json
import numpy as np
import pickle
from utils import *
import yaml
from sklearn.metrics import f1_score

# read configuration file (config.yaml)
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Set model configurations here
model_config_use = model_configs.BERT_SNGP_v1.copy()
model_config_use["num_layers"] = 6
model_config_use["dropout_rate"] = 0.2
model_config_use["use_bn"] = True

# create util folders
for folder in ["datasets", "models"]:
    os.makedirs(folder, exist_ok=True)

if __name__ == "__main__":    

    # Set up training testing dataset
    if config["CREATE_NEW_DATASET"]: 
        if config["new_train_test_split"]:
            class_to_root_data, train_neurons, test_neurons = create_neuron_train_test_split(config["EMBEDDING_DIR"], 
                                                config["CLASS_NAMES"], config["LABEL_MAP"], 
                                                num_train_neurons=config["num_train_neurons"], 
                                                num_test_neurons=config["num_test_neurons"], 
                                                use_all_train_neurons=True, save_to_json=True)
        else:
            train_test_split_file = os.path.join('datasets', 'train_test_split_neurons.json')
            with open(train_test_split_file, 'r') as f:
                data = json.load(f)
                class_to_root_data = data['class_to_root_data']
                train_neurons = data['train_neurons']
                test_neurons = data['test_neurons']
            print(f'Loaded train_test_split file: {train_test_split_file}')

        train_data_files, test_data_files = convert_neurons_to_embedding_files(train_neurons, test_neurons, 
                                                                   class_to_root_data, use_all_synapses=True, 
                                                                   use_all_synapses_for_test=True)
    else:
        print("Using previous dataset ...")

        train_test_split_file = os.path.join('datasets', 'train_test_split_neurons.json')
        with open(train_test_split_file, 'r') as f:
            data = json.load(f)
            class_to_root_data = data['class_to_root_data']
            train_neurons = data['train_neurons']
            test_neurons = data['test_neurons']
        print(f'Loaded train_test_split file: {train_test_split_file}')

        with open(config["TRAIN_EMBED_FILE_PATH"], 'rb') as f:
            train_data_files = pickle.load(f)
        with open(config["TEST_EMBED_FILE_PATH"], 'rb') as f:
            test_data_files = pickle.load(f)    
        print("Loaded.")
            
    # Visualize data set being used
    visualize_dataset(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"], only_overall=False)
    
    train_embed, test_embed = create_dataset_embeds(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"], 
                                                    config["MODELS2USE"], config["EMBEDDING_DIR"], config["AGGREGATE_RADIUS_UM"])    
    
    # Gather all embeddings and create labels
    all_train_embeddings, train_labels = setup_embeds_and_labels(train_embed, config["LABEL_MAP"])
    print("Train embedding shape:", all_train_embeddings.shape)
    print("Train label shape:", train_labels.shape)
    
    # Calculate sample weights based on class weights
    sample_weights = np.array([config["CLASS_WEIGHT_MAP"][label] for label in train_labels])
    
    all_test_embeddings, test_labels = setup_embeds_and_labels(test_embed, config["LABEL_MAP"])
    print("Test embedding shape:", all_test_embeddings.shape)
    print("Test label shape:", test_labels.shape)    
    
    print(f"Model configuration: {model_config_use}")
    
    train_history, model_config, train_config, model = model_handler.train_model(
        train_data=all_train_embeddings,
        train_labels=train_labels,
        train_weights=sample_weights,
        valid_data=all_test_embeddings,
        valid_labels=test_labels,
        model_config=model_config_use,
        training_epochs=config["EPOCHS"], 
        batch_size=config["BATCH_SIZE"],
        verbose=True,
        #balance_labels=config['BALANCE_LABELS'],
        learning_rate=config["LEARNING_RATE"],
        save_model_path=config["SAVE_MODEL_FOLDER_NAME"]
    )

    print("Finished.")

    print('RADIUS:', config['AGGREGATE_RADIUS_UM'])

    model, model_config  = model_handler.load_model(config["SAVE_MODEL_FOLDER_NAME"])

    visualize_dataset(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"], only_overall=False)
                
    #variances, class_probabilities, logits = model_handler.predict_data(all_train_embeddings, model, block_size=1000)
    #predicted_classes_train = np.argmax(class_probabilities, axis=1)

    #f1_macro = f1_score(train_labels, predicted_classes_train, average='macro')
    #print('Train f1 score per synapse:', f1_macro)
    
    variances, class_probabilities, logits = model_handler.predict_data(all_test_embeddings, model, block_size=1000)
    predicted_classes_test = np.argmax(class_probabilities, axis=1)

    f1_macro = f1_score(test_labels, predicted_classes_test, average='macro')
    print('Test f1 score per synapse:', f1_macro)










    

    # test per test synapse
    #train_data_files, test_data_files = convert_neurons_to_embedding_files(train_neurons, test_neurons,
    #                                                               class_to_root_data, use_all_synapses=True,
    #                                                               use_all_synapses_for_test=True)
    
    #train_embed, test_embed = create_dataset_embeds(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"],
    #                                                config["MODELS2USE"], config["EMBEDDING_DIR"], config["AGGREGATE_RADIUS_UM"])
    #all_test_embeddings, test_labels = setup_embeds_and_labels(test_embed, config["LABEL_MAP"])

    #model, model_config  = model_handler.load_model(config["SAVE_MODEL_FOLDER_NAME"])

    #variances, class_probabilities, logits = model_handler.predict_data(all_test_embeddings, model, block_size=1000)
    #predicted_classes_test = np.argmax(class_probabilities, axis=1)

    #f1_macro = f1_score(test_labels, predicted_classes_test, average='macro')
    #print('Test f1 score per synapse:', f1_macro)

