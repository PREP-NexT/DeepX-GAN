# DeepX-GAN
The code repo for the paper "Capturing Unseen Spatial Extremes Through Knowledge-Informed Generative Modeling" (not officially published yet). A preprint of this manuscript is available at [arXiv: 2507.09211](https://arxiv.org/abs/2507.09211). 

This repository is still under active development. Please note that the code has not been fully documented or annotated yet. We are sharing this version to accompany our manuscript and to promote transparency and reproducibility. Variable names and function structures may change, and detailed comments are still being added. Future updates will include improved documentation and cleaner structure.


## Data
The example dataset is shared on Google Drive due to GitHub's file size limit: 
- lgcp (Log-Gaussian Cox Process) [link](https://drive.google.com/file/d/1uWmAV_UhnjZkc6govwxVxhFRN2uPK-_u/view?usp=sharing).
- tmax (daily maximum temperature) [link](https://drive.google.com/file/d/1jWOCox6btoBEMG6vLkh_Qv_YUZuSSJ9f/view?usp=sharing).

These datasets need to be downloaded to "DATA" folder under the project path.


## Environment
To reproduce the Python environment, use the provided Conda environment file `deepx-gan_ENV.yml`:
```bash
conda env create -f deepx-gan_ENV.yml
```

This will create a new environment with all required dependencies.


## Code structure
- `main_parallel.py`: the entry point to configure and start training.
- `train_parallel.py`: defines the training process.
- `models_parallel.py`: defines the model architecture.
- `data_utils.py`: handles data loading and preprocessing.
    > **Note**: The code expects datasets to be stored in the `/DATA/` directory. Please ensure you download and place the datasets in the correct location.
- `gan_utils.py`: contains the loss functions used for training.
- `spatial_utils.py`: implements the computation of the DeepX spatial dependence metric.


## Get started
### Running the model
You can configure all relevant hyperparameters either by modifying `main_parallel.py` directly or by passing arguments via the command line. For example:
```bash
python main_parallel.py -d tmax -ne 1000
```

This will start training the model with `tmax` as the target data and `1000` training epochs.

### Parallel GPU training
The code is designed for parallel training on multiple GPUs. Please ensure you configure the following hyperparameters in `main_parallel.py`:
- `parallel_ids`: a list specifying the GPU device IDs to use in parallel (e.g., `[0, 1]`).
- `cuda_num`: the ID of the main GPU that handles model aggregation and loss computation.

Make sure the specified GPUs are available on your machine.

## Expected output
The log file and trained models will be stored in `/trained/run_name/`, where `run_name` is automatically created using the training dataset, training method, date, and time information.

The generator and discriminator losses will be output in the terminal and stored in a log file, which could be retrieved in a tensorboard by:
```bash
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

## Reproducibility & performance
The code has been tested for reproducibility on **Ubuntu 20.04** using **two NVIDIA RTX A6000 GPUs**. However, it should be compatible with other operating systems and GPU models, provided that your Python environment is correctly configured. 

- **Environment setup**: Creating the Conda environment typically takes about **10–20 minutes**.
- **Training speed**: On the above setup, training takes approximately **4.5 seconds per iteration** with a batch size of `32` and the dataset `tmax`.