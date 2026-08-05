# RMTFNet

##  Reliable Motion-Guided Temporal-Frequency Network for Efficient All-in-One Video Restoration

RMTFNet is a lightweight video restoration framework for all-in-one video restoration and dynamic scene deblurring. It is built around reliable temporal propagation and contains three compact modules:

- **Motion Reliability Gate (MRG)** estimates motion reliability and suppresses unreliable propagated features during deformable alignment.
- **Temporal Selective Fusion (TSF)** adaptively fuses spatial, forward-propagated, and backward-propagated features.
- **Frequency Detail Refinement (FDR)** enhances high-frequency residual details with fixed Laplacian guidance and a lightweight depthwise branch.

## Download

### Datasets

DAVIS data used by this project can be downloaded from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1AGI4b8Z8AOYH1HRP7Ncbxg?pwd=s28n
Extraction code: s28n
```

GoPro should be downloaded from the official GOPRO_Large dataset page:

```text
Official page: https://seungjunnah.github.io/Datasets/gopro.html
```

After downloading, place the datasets under the project `datasets/` directory, or modify the corresponding paths in `options/train/*.yml` and `options/test/*.yml`.

### Trained Models

The trained RMTFNet models can be downloaded from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1Ggl2MeM4_aY7A3hvAesTkw?pwd=a2eu
Extraction code: a2eu
```

## Installation

```bash
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
```

## Data Preparation

### DAVIS


To synthesize low-quality DAVIS videos from clean training videos, run:

```bash
python scripts/data_preparation/synthesize_datasets.py --input_dir datasets/DAVIS_training --output_dir datasets/DAVIS_training/ --continuous_frames 6
```

Generate meta information if needed:

```bash
python scripts/data_preparation/generate_meta_info.py --dataset_path datasets/DAVIS_training --output_path basicsr/data/meta_info/DAVIS_meta_info.txt
```

## Training

Train RMTFNet on DAVIS:

```bash
python basicsr/train.py -opt options/train/train_RMTFNet_DAVIS.yml
```

Train RMTFNet on GoPro:

```bash
python basicsr/train.py -opt options/train/train_RMTFNet_GOPRO.yml
```

## Testing

Test on DAVIS:

```bash
python basicsr/test.py -opt options/test/test_RMTFNet_DAVIS_T6.yml
python basicsr/test.py -opt options/test/test_RMTFNet_DAVIS_T12.yml
python basicsr/test.py -opt options/test/test_RMTFNet_DAVIS_T24.yml
```


Test on Set8:

```bash
python basicsr/test.py -opt options/test/test_RMTFNet_Set8_T6.yml
python basicsr/test.py -opt options/test/test_RMTFNet_Set8_T12.yml
python basicsr/test.py -opt options/test/test_RMTFNet_Set8_T24.yml
```

Test on GoPro:

```bash
python basicsr/test.py -opt options/test/test_RMTFNet_GOPRO.yml
```


## Acknowledgements
The codes are based on BasicSR.The data synthesis code is from AverNet. Thanks the authors for their codes!
