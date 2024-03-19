# SOUL-SRFlow-Project

**Code for the research work: Empirical Bayesian Imaging with Large-Scale Push-Forward Generative Priors**

[Access to the paper here!](https://ieeexplore.ieee.org/abstract/document/10419008)

**Abstract:** We propose a new methodology for leveraging deep generative priors for Bayesian inference in imaging inverse problems. Modern Bayesian imaging often relies on score-based diffusion generative priors, which deliver remarkable point estimates but significantly underestimate uncertainty. Push-forward models such as variational autoencoders and generative adversarial networks provide a robust alternative, leading to Bayesian models that are provably well-posed and which produce accurate uncertainty quantification results for small problems. However, push-forward models scale poorly to large problems because of issues related to bias, mode collapse and multimodality. We propose to address this difficulty by embedding a conditional deep generative prior within an empirical Bayesian framework. We consider generative priors with a super-resolution architecture, and perform inference by using a Bayesian computation strategy that simultaneously computes the maximum marginal likelihood estimate (MMLE) of the low-resolution image of interest, and draws Monte Carlo samples from the posterior distribution of the high-resolution image, conditionally to the observed data and the MMLE. The methodology is demonstrated with an image deblurring experiment and comparisons with the state-of-the-art.

![fox_images](https://github.com/SavvasMel/SOUL-SRFlow-Project/assets/79579567/72526390-ece9-4933-917c-3cfa21eab449)

## Setup

* Anaconda
* Python 3.7
* Pytorch 1.13

First clone the repository and set the path

```
git clone https://github.com/SavvasMel/SOUL-SRFlow-Project.git
cd SOUL-SRFlow-Project
```

Create a conda environment
```bash
conda env create -f environment.yml
conda activate SRFlow_conda_env
```

The pretrained **SRFLOW-DA models** can be downloaded [here](https://github.com/yhjo09/SRFlow-DA). The pretrained **SRFlow models** can be downloaded as
```bash
sh ./download.sh
```
The pretrained models should be unzipped and placed in the folder ```pretrained_models``` as
```bash
├── pretrained_models # pretrained_models folder
│   ├── SRFlow-DA/models
        ├── 200000_G_X4.pth
        ├── 200000_G_X8.pth
│   ├── SRFlow-DA-D/models
        ├── 200000_G_X4.pth
        ├── 200000_G_X8.pth
│   ├── SRFlow-DA-R/models
        ├── 200000_G_X4.pth
        ├── 200000_G_X8.pth
│   ├── SRFlow-DA-S/models
        ├── 200000_G_X4.pth
        ├── 200000_G_X8.pth
│   ├── RRDB_CelebA_8X.pth
│   ├── RRDB_DF2K_4X.pth
│   ├── RRDB_DF2K_8X.pth
│   ├── SRFlow_CelebA_8X.pth
│   ├── SRFlow_DF2K_4X.pth
│   ├── SRFlow_DF2K_8X.pth
```

**Note:** If you wish to have a different structure for the pretrained models you might need to modify some directories/paths in ```./confs/config_paths.py``` and ```./confs/*.yml```.

## Run SOUL-SRFlow

1. You may have to modify some directories/paths in the config file ```./confs/config_paths.py```.
2. You can **run SOUL-SRFlow** as below. We suggest to use the SRFlow-DA model since it reduces the computational time.

```bash
python3 SOUL-SRFlow.py --conf_path=./confs/SRFlow-DA_DF2K_8X.yml        # SRFlow-DA 8X SR
python3 SOUL-SRFlow.py --conf_path=./confs/SRFlow_DF2K_8X.yml           # SRFlow 8X SR
```

3. If you do not acquire a GPU in your system, please try with prefix CUDA_VISIBLE_DEVICES=-1 (CPU only):
```bash
CUDA_VISIBLE_DEVICES=-1 python3 SOUL-SRFlow.py --conf_path=./confs/SRFlow-DA_DF2K_8X.yml        # SRFlow-DA 8X SR
```

### Hyperparameters

You can use the parser of ```SOUL-SRFlow.py``` to change the hyperameters of the sampling and optimisation parts of the algorithm. See [paper](https://ieeexplore.ieee.org/abstract/document/10419008) and below for more details.

**Data generation:** To change image for an experiment you can change the ```--index_image```. We provide to type of blurs in the code, uniform and Gaussian blurs. You can change between the two by changing the ```--type_blur```. To change the kernel size or the standard deviation of the Gaussian blur you can use ```--kernel_size``` and ```--kernel_std``` respectively. Example:
```bash
python3 SOUL-SRFlow.py --conf_path=./confs/SRFlow-DA_DF2K_8X.yml --index_image=1 --type_blur=Gaussian --kernel_size=9 --kernel_std=3
```

**Sampling:** internal number of sampling steps $m_n$ , stepsize $\gamma_n$ , number of burn-in period steps $b$. You can change those as:
```bash
python3 SOUL-SRFlow.py --conf_path=./confs/SRFlow-DA_DF2K_8X.yml --mn =5 --stepsize=5e-5 --b_iter=100  # default choices for mn and stepsize and burn-in period.
```

**Optimisation:** total number of optimisation steps $N$, scales $c$ and $p$ (```d_scale``` and ```d_exp``` respectively in the code) which both determine the optimisation stepsize as $\delta_n = c\cdot n^{-p}/d$. You can change those as:
```bash
python3 SOUL-SRFlow.py --conf_path=./confs/SRFlow-DA_DF2K_8X.yml --niter=6e4 --d_scale = 0.01 --d_exp=0.7   # default choices for N and scales c and p.
```

**Prior distribution:** You can change the standard deviation of the Gaussian prior on $z$ by passing ```--sigma_z``` as the previous examples. Note that $\sigma_z\in (0,1]$.

## Acknowledgments

This repo contains parts of code taken from :

Super-Resolution using Normalizing Flow in PyTorch (SRFlow) : https://github.com/andreas128/SRFlow

Super-Resolution Using Normalizing Flow with Deep Convolutional Block (SRFlow-DA) : https://github.com/yhjo09/SRFlow-DA

## Citation

```
@ARTICLE{10419008,
  author={Melidonis, S. and Holden, M. and Altmann, Y. and Pereyra, M. and Zygalakis, K. C.},
  journal={IEEE Signal Processing Letters}, 
  title={Empirical Bayesian Imaging With Large-Scale Push-Forward Generative Priors}, 
  year={2024},
  volume={31},
  number={},
  pages={631-635},
  keywords={Bayes methods;Imaging;Uncertainty;Monte Carlo methods;Estimation;Superresolution;Optimization;Computational imaging;inverse problems;Bayesian inference;uncertainty quantification;deep generative models;Markov chain Monte Carlo;stochastic optimisation},
  doi={10.1109/LSP.2024.3361806}}
```












