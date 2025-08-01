# Data Preparation Instructions

This document outlines the steps to prepare the dataset for reproducing our experiments. The process involves acquiring data from multiple sources, pre-processing the images, and creating manifest files (CSVs) to define the data splits.

### Step 1: Collect Data
To reproduce our experiment results, please follow our data composition design. For the training, validation, and in-distribution test datasets, use CXR images (AP or PA views) from the Valencian Region Medical ImageBank (BIMCV) network, specifically [BIMCV-COVID-19+](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/) and [Pachest](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590857662078-c30d2790-05dc) for COVID-19 and pneumonia, respectively. 

For out-of-distribution test dataset, the COVID-19 cases come from [COVID-19-AR](https://www.doi.org/10.7937/tcia.2020.py71-5978) and [V2-COV19-NII](https://doi.org/10.6084/m9.figshare.12275009), and Pneumonia cases from [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) and [Chexpert](https://doi.org/10.71718/y7pj-4v93).

### Step 2: Preprocess Data

Our model is trained on images of cropped lung areas from full chest X-rays. We used the **[HybridGNet](https://github.com/ngaggion/HybridGNet)** model for this segmentation task. Run this or a similar lung segmentation model on all your acquired images before proceeding.

### Step 3: Create CSV Manifest Files

After pre-processing, create four `.csv` files and place them in the `data/` directory:

1.  `train.csv`
2.  `val.csv`
3.  `seen_test.csv` (for in-distribution testing)
4.  `unseen_test.csv` (for out-of-distribution testing)

#### CSV File Format

Each CSV file must contain the following four columns:
* `image_path`: The absolute or relative path to a processed (cropped) image.
* `class_idx`: The integer label for the class (e.g., `0` for COVID-19, `1` for Pneumonia).
* `class_name`: The string name for the class.
* `source_name`: The name of the dataset source (e.g., `padchest`, `germany`).

#### Data Split Logic

Our experimental design simulates a real-world scenario where a model is trained on data from a limited number of sources and tested on new, unseen sources.

* **In-Distribution (`train.csv`, `val.csv`, `seen_test.csv`):** These files should contain data from **only two** of the six sources.
* **Out-of-Distribution (`unseen_test.csv`):** This file should contain data from the **remaining four** sources that were not used for training, validation, or in-distribution testing.

#### Example Rows

**`train.csv` (In-Distribution Example):**
```csv
image_path,class_idx,class_name,source_name
dataset/covid19/velancia/inverted_sub-S12551_ses-E25629_run-1_bp-chest_vp-pa_cr_lung_crop.png,0,covid19,velancia
dataset/pneumonia/padchest/164400068288269695106145841373485849494_z3v12l_lung_crop.png,1,pneumonia,padchest
```

**`unseen_test.csv` (Out-of-Distribution Example):**
```csv
image_path,class_idx,class_name,source_name
/dataset/covid19/germany/93fd0adb.nii_lung_crop.png,0,covid19,germany
/dataset/covid19/tcia/COVID-19-AR-16406505_lung_crop.png,0,covid19,tcia
/dataset/pneumonia/nih/00016291_015_lung_crop.png,1,pneumonia,nih
/dataset/pneumonia/chexpert/atient04298_lung_crop.png,1,pneumonia,chexpert
```