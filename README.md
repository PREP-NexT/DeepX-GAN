# DeepX-GAN
The code repo for the paper "Capturing Unseen Spatial Extremes Through Knowledge-Informed Generative Modeling" (not officially published yet). A preprint of this manuscript is available at [arXiv: 2507.09211](https://arxiv.org/abs/2507.09211). 

This repository is still under active development. Please note that the code has not been fully documented or annotated yet. We are sharing this version to accompany our manuscript and to promote transparency and reproducibility. Variable names and function structures may change, and detailed comments are still being added. Future updates will include improved documentation and cleaner structure.


## Data
The example dataset is shared on Google Drive due to GitHub's file size limit: 
- lgcp (Log-Gaussian Cox Process) [link](https://drive.google.com/file/d/1uWmAV_UhnjZkc6govwxVxhFRN2uPK-_u/view?usp=sharing).
- tmax (daily maximum temperature) [link](https://drive.google.com/file/d/1jWOCox6btoBEMG6vLkh_Qv_YUZuSSJ9f/view?usp=sharing).

These datasets need to be downloaded to "DATA" folder under the project path.


## Environment
The Python environment to run the codes can be reproduced using "deepx-gan_ENV.yml" file by conda in the console:
``` bash
$ conda env create -f deepx-gan_ENV.yml
```

## Structure
- The model architecture is defined in `models_parallel.py`;
- The functions used for data loading and preprocessing are defined in `data_utils.py` (note that the data will be loaded from `/DATA/` folder so remember to download datasets and store to the right location);
- The functions related to loss function are defined in `gan_utils.py`;
- The functions related to computing the spatial dependence structure metric DeepX are defined in `spatial_utils.py`;
- The training details are defined in `train_parallel.py`;
- The main function to start the training is defined in `main_parallel.py`.


## Get started
You may change all the relevant hyperparameters in `main_parallel.py` or input into the terminal as, e.g., `python main_parallel.py -d tmax -ne 1000`. Then, run `main_parallel.py` to start training the model. 


## Expected output
The log file and trained models will be stored in `/trained/run_name/`, where `run_name` is automatically created using the training dataset, training method, date, and time information.

The generator and discriminator losses will be output in the terminal and stored in a log file, which could be retrieved in a tensorboard by:
``` bash
tensorboard --logdir=log
```

Then go to the URL it provides for visualization OR to http://localhost:6006/.

## Warnings
During training, you may encounter the following ignorable warnings:
```
UserWarning: geopandas not available. Some functionality will be disabled.
  warn("geopandas not available. Some functionality will be disabled.")
```
which usually comes from packages like `xarray` or `cartopy` that can optionally use geopandas for geographic features, and
```
UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
```
which commonly happens when using `nn.DataParallel` as a performance warning.
