# Dorsal-Horn Cell Type Classification using machine learning
- SegCLR (Constrastive learning) https://www.nature.com/articles/s41592-023-02059-8 from Sven Dorkenwald

## ML Logic

### Self-Supervised Learning (Contrastive Learning)
Using embeddings models trained on the MICrONs and H01 datasets (from paper) to generate 64-dim embeddings from local 3D cutouts centered at neurons synaptic butons (manual synapses selected and auto synapse detection).

Details on pretrained models and SegCLR (https://github.com/google-research/connectomics/wiki/SegCLR)
- H01: `gs://h01-release/data/20230118/models/segclr-355200/`
- MICrONS: `gs://iarpa_microns/minnie/minnie65/embeddings/models/segclr-216000/`
- Lee Lab mouse spinal cord embedding model

### Classification (Multi-Layer Perceptron)
Using the embeddings generated from the previous step, train a simple MLP to predict cell type. 

## Run
### Generating embeddings
All code to generate embeddings (`embed/`)

`cd embeds`

Update the config file `configs/config.yaml` as needed. 
- COORD_LIST: List of file paths to annotation layers from neuroglancer (one point per synapse) or list of coordinates centered at the neurons soma. Latest root ID is pulled to generate the list of auto predicted synapses.
- LABEL_NAME_LIST: List of strings that correspond to the label name for each item in COORD_LIST
- ROOT_SAVE_FOLDER + ROOT_SAVE_FOLDER: Path to save data

Note: Update cloudvolume paths depending on dataset used.

1. `python3 gen_coord_cutouts.py --config configs/config.yaml`
Generate local 3D cutouts given a list of synapse coordinates.

2. `python3 gen_embeddings.py --config configs/config.yaml`
Generate embeddings from the local 3D cutouts using the mebedding models (Used Tensorflow)


### Training downstream classifier
All code to train a classifier (`classification/`)

`cd classification`

Update the config file `config.yaml` as needed. 
- CLASS_NAMES: List of labels (folder names must match)
- EMBEDDING_DIR: Path to the main embedding dir used from step 1
- LABEL_MAP: Label mapping from label to value. Option to merge different classes together.
- EMBEDDING_DIR: Should be the same as ROOT_SAVE_FOLDER+FOLDER_EXT (locaiton of data)
- SAVE_MODEL_PATH: Path to .pth torch model

1. `python3 train.py`
Train a MLP (with Torch) using the generated embeddings.

####################################################################################################
### Figure 2
(TBC)
