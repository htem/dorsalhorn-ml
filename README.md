## Dorsal-Horn Cell Type Classification using machine learning
- SegCLR (Constrastive learning) https://www.nature.com/articles/s41592-023-02059-8

### Logic

#### Self-Supervised Learning (Contrastive Learning)
Using embeddings models trained on the MICrONs and H01 datasets (from paper) to generate embeddings from local 3D cutouts centered at synaptic sites.

#### Classifier (Multi-Layer P)
Using the embeddings generated from the previous step, train a simple MLP to predict cell type.

### Figure 2
