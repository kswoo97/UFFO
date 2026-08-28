# Official implementation of UFFO

### Paper information
- **[Title]** User-Adaptive Fusion of Multiple Foundation Models for Recommender Systems
- **[Venue]** Submitted to WSDM 2027

### Overview
- **[TL;DR]** We present an implementation of UFFO, a user-adaptive score-level fusion of foundation models for recommendations.
- **[Detail]** We present model code and running code for UFFO under both sequential and static settings.

### Datasets
- **[Source]** We use four Amazon 2023 review datasets [1]: (1) Sports & Outdoors, (2) Toys & Games, (3) Beauty & Personal Care, and (4) Arts, Crafts, & Sewing.
- **[Preprocessing]** We use the preprocessed version of these datasets from [2].
- **[Link]** You need to download the files in [LINK](https://www.dropbox.com/scl/fo/dx104zyjye68qkpqlo4yt/AFm_wwrM2hHH6zSwZfK-21U?rlkey=25yogmq65wovkkrn9jk9ws132&st=qpkf04v4&dl=0) and locate them in ``./datasets`` directory.

### How to run
- There are two files: ``model_uf2o_seq.py`` and ``model_uf2o_stat.py``.

#### Running under the sequential setting:
- One can use the following code to reproduce the experimental results of UF2O.
```
python3 model_uf2o_seq.py --dataset sports_outdoors --device cuda:0 --lr 0.001 --tau 2.0
```
- ``--dataset``indicates the dataset one aims to use. You can select among ``sports_outdoors``, ``toys_games``, ``beauty_care``, and ``arts``.
- ``--device`` indicates the GPU one aims to use (e.g., ``cuda:0`` or ``cuda:3``.)
- ``--lr`` indicates the learning rate one aims to use (e.g., ``0.001`` or ``0.0005``).
- ``--tau`` indicates the temperature hyperparameter for user-attention weight computation (e.g., ``1.0`` or ``0.1``).

#### Running under the static setting:
- One can use the following code to reproduce the experimental results of UF2O.
```
python3 model_uf2o_stat.py --dataset sports_outdoors --device cuda:0 --lr 0.001 --tau 2.0
```
- ``--dataset``indicates the dataset one aims to use. You can select among ``sports_outdoors``, ``toys_games``, ``beauty_care``, and ``arts``.
- ``--device`` indicates the GPU one aims to use (e.g., ``cuda:0`` or ``cuda:3``.)
- ``--lr`` indicates the learning rate one aims to use (e.g., ``0.001`` or ``0.0005``).
- ``--tau`` indicates the temperature hyperparameter for user-attention weight computation (e.g., ``1.0`` or ``0.1``).


### References
- [1] Hou et al., Bridging Language and Items for Retrieval and Recommendation, arXiv 2024.
- [2] Kim et al., Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation, The Web Conference, 2026.
