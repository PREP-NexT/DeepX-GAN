# DeepX-GAN

The code repository for the paper **"Capturing Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling"**, published in [Earth's Future](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2026EF008861) in Sep 2026.  

---

## Method overview

**DeepX-GAN** (**D**ependence-**E**nhanced **E**mbedding for **P**hysical e**X**tremes - **G**enerative **A**dversarial **N**etwork) explicitly incorporates extremal dependence structures in climate fields. We infuse the knowledge of spatial tail dependence structures into the deep generative model by embedding a novel **DeepX** (**D**ependence-**E**nhanced **E**mbedding for **P**hysical e**X**tremes) metric, which identifies evolving patterns across space and time while explicitly accounting for the spatial correlation of extreme events. It ensures the generated sequences are closely aligned with real data in a transformed space where extremal spatiotemporal patterns are easier to learn. The embedding metric is fused with real (or generated) data along the channel dimension. When minimizing the embedding loss, the generator is optimized to reconstruct the spatial tail dependence structure observed in the real dataset. This integration enables DeepX-GAN to gain insight into the collective behaviors of spatial extreme events, facilitating more reliable simulation of spatially compounding events crucial for risk assessment.

---

## Data

The example dataset (ERA5 daily maximum 2-m temperature for the MENA region) [link](https://drive.google.com/file/d/1agxGISWhy1zEz_efa7uxoIW_ct4402or/view?usp=sharing) is required to run the code. It is shared on Google Drive due to GitHub's file size limit and needs to be placed in the `DATA/` directory one level above this folder (i.e., `../DATA/`):

| File | Description |
|------|-------------|
| `era5.reanalysis.t2m.daymax.32x64.1979-2014.MENA.ncep_t62.nc` | ERA5 t2m daily max, 1979–2014, MENA (32 × 64 grid) |

---

## Environment

To reproduce the Python environment, install the dependencies with pip:

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. GPU training is strongly recommended.

---

## Code structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — parse arguments and launch training |
| `train.py` | Training loop |
| `models.py` | Generator (`VideoDCG`) and Discriminator (`VideoDCD`) |
| `spatial_utils.py` | DeepX embedding computation |
| `gan_utils.py` | Sinkhorn loss and martingale regularisation |
| `data_utils.py` | Climate data loading and seasonal preprocessing |
| `requirements.txt` | Python dependencies |

> **Note**: The code expects datasets to be stored in the `../DATA/` directory relative to this folder.

---

## Get started

### Running the model

You can configure all relevant hyperparameters either by modifying `main.py` directly or by passing arguments via the command line. For example:

```bash
python main.py --dname era5-t2m-daymax --n_epochs 1000
```

This will start training the model on ERA5 daily maximum temperature data for 1000 epochs.

**Quick debug run** (64 samples, verifies everything runs end-to-end):

```bash
python main.py --debug_run True --n_epochs 10 --save_freq 5
```

**Resume from checkpoint**:

```bash
python main.py \
    --pre_trained_path ./trained/<run_name>/ckpts \
    --iter_final <checkpoint_iteration>
```

### Adding a new dataset

Add an entry to the `DATASET_REGISTRY` dictionary in `data_utils.py`:

```python
DATASET_REGISTRY = {
    'era5-t2m-daymax': { ... },   # existing
    'my-new-dataset': {
        'filename':    'my_data.nc',
        'nc_var':      'variable_name',
        'time_dim':    'time',
        'description': 'Short description',
    },
}
```

No other code changes are needed — pass `--dname my-new-dataset` on the command line.

---

## Expected output

Trained model checkpoints and TensorBoard logs are stored in `../trained/<run_name>/`, where `run_name` is automatically created from the dataset name, embedding method, date, and time (e.g. `20260101_2045_era5-t2m-daymax-tdc_masked`).

A structured JSON configuration file (`train_notes.json`) is saved alongside the checkpoints for reproducibility.

To visualise training losses:

```bash
tensorboard --logdir ../trained/<run_name>/log
```

Then open the URL shown in the terminal or go to http://localhost:6006/.

---

## Key hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--dname` | `era5-t2m-daymax` | Dataset name (key in `DATASET_REGISTRY`) |
| `--stx_method` | `tdc_masked` | DeepX metric variant (`tdc_masked`, `tdc`, `skw`) |
| `--u` | `0.7` | Quantile threshold for extremes (top 30%) |
| `--time_steps` | `30` | Days per sequence window |
| `--batch_size` | `32` | Training batch size |
| `--n_epochs` | `50000` | Total training epochs |
| `--sinkhorn_eps` | `0.8` | Sinkhorn regularisation ε |
| `--reg_penalty` | `1.5` | Martingale penalty coefficient λ |
| `--save_freq` | `100` | Checkpoint interval (iterations) |
| `--cuda_num` | `0` | GPU device index |

---

## Reproducibility & performance

The code has been tested on **Ubuntu 20.04** using **NVIDIA RTX A6000 GPUs**.  It should be compatible with other operating systems and GPU models, provided the Python environment is correctly configured.

---

## Citation

You are very welcome to use this code, and please cite our paper.

Liu, Xinyue, Xiao Peng, Shuyue Yan, et al. “Capturing Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling.” Earth’s Future 14, no. 9 (2026): e2026EF008861. https://doi.org/10.1029/2026EF008861.

@article{Liu2026Capturing,
  title = {Capturing {{Unseen Spatial Heat Extremes Through Dependence-Aware Generative Modeling}}},
  author = {Liu, Xinyue and Peng, Xiao and Yan, Shuyue and Chen, Yuntian and Zhang, Dongxiao and Niu, Zhixiao and Wang, Hui-Min and He, Xiaogang},
  date = {2026},
  journaltitle = {Earth's Future},
  volume = {14},
  number = {9},
  pages = {e2026EF008861},
  issn = {2328-4277},
  doi = {10.1029/2026EF008861},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1029/2026EF008861}
}



---

## Acknowledgements

This work builds on the SPATE-GAN framework (Klemmer et al. 2022).  ERA5 data were obtained from the Copernicus Climate Change Service.


