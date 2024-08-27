Supplementary Material for the paper `Recurrent Inertial Graph-based Estimator (RING)`.

This repository contains the coda and data (data is only used for evaluation, not training) that allows to:
- Use the trained RING network.
- Retrain the RING network from scratch.
- Recreate all published experimental validation results of RING and the comparison SOTA methods.

# Installation
These are the installation steps of the `ring` Python package.
1) Create virtual Python environment with Python >= 3.10
2) cd into `pkg` folder
3) install `ring` package using `pip install .`

Note: This installs jax as cpu-only, if you have a GPU you may want to install the gpu-enabled version.

# Application of `RING`
After the installation steps, `RING` can be used for inertial motion tracking.
```python
import ring
import numpy as np

T  : int       = 30        # sequence length     [s]
Ts : float     = 0.01      # sampling interval   [s]
B  : int       = 1         # batch size
# in python counting begins at 0, in this convention the label of the base body changes from 0 to -1
lam: list[int] = [-1, 0, 1] # parent array
N  : int       = len(lam)  # number of bodies
T_i: int       = int(T/Ts) # number of timesteps

X              = np.zeros((B, T_i, N, 9))
# where X is structured as follows:
# X[..., :3]   = acc
# X[..., 3:6]  = gyr
# X[..., 6:9]  = jointaxis

# let's assume we have an IMU on each outer segment of the
# three-segment kinematic chain
X[..., 0, :3]  = acc_segment1
X[..., 2, :3]  = acc_segment3
X[..., 0, 3:6] = gyr_segment1
X[..., 2, 3:6] = gyr_segment3

ringnet = ring.RING(lam, Ts)
yhat, _ = ringnet.apply(X)
# yhat: unit quaternions, shape = (B, T_i, N, 4)
```

# Retraining of `RING`

After the installation steps, you can use the two files `train_*.py` to
1) Create training data. Usage: `python train_step1_generateData.py CONFIG OUTPUT_PATH SIZE SEED SAMPLING_RATES ANCHORS` where
    - CONFIG: str, one of ['standard', 'expSlow', 'expFast', 'hinUndHer']
    - OUTPUT_PATH: str, path to where the data will be stored
    - SIZE: int, this many 1-minutes long sequences will be created
    - SEED: int, seed of PRNG
    - SAMPLING_RATES: list of floats
    - ANCHORS: list of strings (advanced, leave as is)

Example: `python train_step1_generateData.py standard ring_data 32 1 "[100]" seg3_2Seg,`

For retraining of RING (creates 750 GBs of data!!!): `python train_step1_generateData.py standard ~/ring_data 32256 1 && python train_step1_generateData.py expSlow ~/ring_data 32256 2 && python train_step1_generateData.py expFast ~/ring_data 32256 3 && python train_step1_generateData.py hinUndHer ~/ring_data 32256 4`

2) Retrain RING. Usage: `python train_step2_trainRing.py BS EPISODES PATH_DATA_FOLDER PATH_TRAINED_PARAMS USE_WANDB WANDB_PROJECT PARAMS_WARMSTART SEED DRY_RUN` where
    - BS: int, batchsize
    - EPISODES: int, number of training epochs
    - PATH_DATA_FOLDER: str, path to where the data was stored before
    - PATH_TRAINED_PARAMS: str, path to where the trained parameters will be stored
    - USE_WANDB: bool, whether or not wandb is used. Default is False.
    - WANDB_PROJECT: str, wandb project name. Default is RING.
    - PARAMS_WARMSTART: str, path to parameters from which the training is started. Default is None (= no warmstart).
    - SEED: int, seed for initialization of parameters
    - DRY_RUN: bool, if True network size is tiny for testing. Default is False.

Example: `python train_step2_trainRing.py 2 10 ring_data params/trained --dry-run`

For retraining of RING: `python train_step2_trainRing.py 512 4800 ~/ring_data ~/params/trained_ring_params.pickle`

# Recompute Validation Metrices

After the installation steps, you can use the files `eval_*.py` to recreate the experimental validation results published in the paper. Just execute them and they will print the metrices of RING and SOTA methods to the stdout.

Example: `python eval_section_5_3_3.py` prints 
    
    Method `RING` achieved 6.776087284088135 +/- 1.4104136228561401