# Repository for Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems

### Usage examples

**MovieLens 100k**
`python main.py --dataset ml-100k --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550 --SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780 --VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006`

**Amazon Books**
`python main.py --dataset books@5 --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550 --SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780 --VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006 --runs 1`

<hr>
The generated data can be found in `/data/generated` for all three synthetization variants.

### Train on custom data
`python preprocess.py --input_path <json data> --dataset <name> --quantile 0.0`

Then place the files in `data`.
Pickles contain csr_matrix objects (scipy~=1.6.2) for data and python lists for id to name matching.
Python version 3.8 or greater was used. 
