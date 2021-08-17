# ab-line-classifer
![Deep Breathe Logo](img/readme/deep-breathe-logo.jpg "Deep Breath AI")   

We at [Deep Breathe](https://www.deepbreathe.ai/) sought to train a deep learning model for the task
of automating the distinction between normal and abnormal lung tissue based of a lung ultrasound.


This repository contains work relating to development and validation of an A-line vs. B-line
ultrasound image classifier that was used for the creation of the paper [INSERT PAPER NAME](link-to-the-peper.com).

## Table of Contents
1. [**_Getting Started_**](#getting-started)
2. [**_Data Pre-Processing_**](#data-pre-processing)
3. [**_Use Cases_**](#use-cases)  
   i)[**_Train Single Experiment_**](#train-single-experiment)  
   ii) [**_K-Fold Cross Validation_**](#k-fold-cross-validation)  
   iii) [**_Hyper Parameter Optimization_**](#hyper-parameter-optimization)  
   iv) [**_Predictions_**](#predictions)  
   v) [**_Grad-CAM for Individual Frame Predictions_**](#grad-cam-for-individual-frame-predictions)  
4. [**_Project Configuration_**](#project-configuration)
5. [**_Project Structure_**](#project-structure)
6. [**_Contacts_**](#contacts)

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
3. Obtain lung ultrasound (LUS) data and preprocess it accordingly. See
   [Data preprocessing](#data-preprocessing) for more details.
4. Update the _TRAIN >> MODEL_DEF_ field of [_config.yml_](config.yml) with
   the appropriate string representing the model type you wish to
   train. To train a model, ensure the _TRAIN >>
   EXPERIMENT_TYPE_ field is set to _'train_single'_.
5. Execute [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be serialized within
   _results/models/_, and its filename will resemble the following
   structure: _{modeltype}{yyyymmdd-hhmmss}.{ext}_, where _{modeltype}_
   is the type of model trained, _{yyyymmdd-hhmmss}_ is the current
   time, and _{ext}_ is the appropriate file extension.
6. Navigate to _results/experiments/_ to see the performance metrics
   achieved by the model's forecast on the test set. The file name will
   be _{modeltype}_eval_{yyyymmdd-hhmmss}.csv_. You can find a
   visualization of the test set forecast in
   _img/UPDATE_ME/_. Its filename will be
   _UPDATE_ME_.
   
## Data Pre-Processing
for new section about data preprocessing
add a description of the data required - data type, maybe ping derek for an example of the data types
basically say if they have data like ours this is how to use
2 data types, csv of the clip and individual frame csv
second can be made using a script in the repo
csvs contain image path
   
## Use Cases

### Train Single Experiment

### K-Fold Cross Validation

### Hyper Parameter Optimization 
(in train.py)

### Predictions
(frame preds, clip preds)

### Grad-CAM for Individual Frame Predictions 
(run gradcam.py and pick frame)


(mention the model we chose and the config key)

## Project Configuration
This project contains several configurable variables that are defined in
the project config file: [config.yml](config.yml). When loaded into
Python scripts, the contents of this file become a dictionary through
which the developer can easily access its members.

For user convenience, the config file is organized into major components
of the model development pipeline. Many fields need not be modified by
the typical user, but others may be modified to suit the user's specific
goals. A summary of the major configurable elements in this file is
below.
<details closed> 
<summary>Paths</summary>

- **CLIPS_TABLE**: 'data/clips_by_patient_cropped.csv'
- **FRAME_TABLE**: 'data/frames_actually_cropped.csv'
- **DATABASE_QUERY**: 'data/parenchymal_clips.csv'
- **RAW_CLIPS**: 'data/raw_clips/'
- **FRAMES**: 'B:/Datasets/Ottawa/pure_and_muggle/frames/'
- **PARTITIONS**: 'data/partitions/'
- **TEST_DF**: 'data/partitions/test_set_final.csv'
- **EXT_VAL_CLIPS_TABLE**: 'data/clips_by_patient_mini.csv'
- **EXT_VAL_FRAME_TABLE**: 'data/frames_mini.csv'
- **EXT_VAL_FRAMES**: 'data/frames_mini/'
- **HEATMAPS**: 'img/heatmaps'
- **LOGS**: 'results/logs/'
- **IMAGES**: 'results/figures/'
- **MODEL_WEIGHTS**: 'results/models/'
- **MODEL_TO_LOAD**: 'results/models/cutoffvgg16_final_cropped.h5'
- **CLASS_NAME_MAP**: 'data/serializations/output_class_indices.pkl'
- **BATCH_PREDS**: 'results/predictions/'
- **METRICS**: './results/metrics/'
- **EXPERIMENTS**: './results/experiments/'
- **EXPERIMENT_VISUALIZATIONS**: './img/experiments/'
</details>

<details closed> 
<summary>Data</summary>

- **IMG_DIM**: [128, 128]
- **VAL_SPLIT**: 0.1
- **TEST_SPLIT**: 0.1
- **CLASSES**: ['a_lines', 'b_lines']  
</details>

<details closed> 
<summary>Train</summary>

- **MODEL_DEF**: 'cutoffvgg16'   # One of {'vgg16', 'mobilenetv2', 'xception', 'efficientnetb7', 'custom_resnetv2', 'cutoffvgg16'}
- **EXPERIMENT_TYPE**: 'single_train'               # One of {'single_train', 'cross_validation', 'hparam_search'}
- **N_CLASSES**: 2
- **BATCH_SIZE**: 256
- **EPOCHS**: 15
- **PATIENCE**: 15
- **METRIC_PREFERENCE**: ['auc', 'recall', 'precision', 'loss']
- **NUM_GPUS**: 1
- **MIXED_PRECISION**: false                         # Necessary for training with Tensor Cores
- **N_FOLDS**: 10
- **DATA_AUG**:
  - **ZOOM_RANGE**: 0.1
  - **HORIZONTAL_FLIP**: true
  - **WIDTH_SHIFT_RANGE**: 0.2
  - **HEIGHT_SHIFT_RANGE**: 0.2
  - **SHEAR_RANGE**: 10
  - **ROTATION_RANGE**: 45
  - **BRIGHTNESS_RANGE**: [0.7, 1.3]
- **HPARAM_SEARCH**:
  - **N_EVALS**: 10
  - **HPARAM_OBJECTIVE**: 'auc'
</details>

<details closed> 
<summary>Hyper Parameters</summary>

- **MOBILENETV2**:
  - **LR**: 0.001
  - **DROPOU**T: 0.35
  - **L2_LAMBDA**: 0.0001
  - **NODES_DENSE0**: 32
  - **FROZEN_LAYERS**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25,26,26,27,28,29,30]
- **SHUFFLENETV2**:
  - **LR**: 0.1
  - **DROPOUT**: 0.5
  - **L2_LAMBDA**: 0.01
- **VGG16**:
  - **LR**: 0.01
  - **DROPOUT**: 0.5
  - **L2_LAMBDA**: 0.01
  - **NODES_DENSE0**: 64
  - **FROZEN_LAYERS**: []
- **XCEPTION**:
  - **LR**: 0.01
  - **DROPOUT**: 0.5
  - **FROZEN_LAYERS**: []
  - **L2_LAMBDA**: 0.01
- **BiTR50x1**:
  - **LR**: 0.1   #https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html
  - **DROPOUT**: 0.5
  - **L2_LAMBDA**: 0.01
- **EFFICIENTNETB7**:
  - **LR**: 0.1
  - **DROPOUT**: 0.5
  - **L2_LAMBDA**: 0.01
  - **FROZEN_LAYERS**: []
- **CNN0**:
  - **LR**: 0.001
  - **DROPOUT**: 0.35
  - **L2_LAMBDA**: 0.0001
  - **NODES_DENSE0**: 64
  - **KERNEL_SIZE**: 3
  - **STRIDES**: 1
  - **MAXPOOL_SIZE**: 2
  - **BLOCKS**: 4
  - **INIT_FILTERS**: 32
  - **FILTER_EXP_BASE**: 2
- **CUSTOM_RESNETV2**:
  - **LR**: 0.000046
  - **DROPOUT0**: 0.45
  - **DROPOUT1**: 0.40
  - **STRIDE**S: 1
  - **BLOCKS**: 2
  - **INIT_FILTERS**: 16
- **CUTOFFVGG16**:
  - **LR_EXTRACT**: 0.0003
  - **LR_FINETUNE**: 0.0000093
  - **DROPOUT**: 0.45
  - **CUTOFF_LAYER**: 10
  - **FINETUNE_LAYER**: 7
  - **EXTRACT_EPOCHS**: 6
</details>

<details closed> 
<summary>Hyper Parameter Search</summary>

- **MOBILENETV2**:
  - **LR**:
    - **TYPE**: 'float_log'
    - **RANGE**: [0.00001, 0.001]
  - **DROPOUT**:
    - **TYPE**: 'float_uniform'
    - **RANGE**: [0.0, 0.5]
- **CUTOFFVGG16**:
  - **LR_EXTRACT**:
    - **TYP**E: 'float_log'
    - **RANGE**: [0.00001, 0.001]
  - **LR_FINETUNE**:
    - **TYPE**: 'float_log'
    - **RANGE**: [0.000001, 0.00001]
  - **DROPOUT**:
    - **TYPE**: 'float_uniform'
    - **RANGE**: [0.0, 0.5]
  - **EXTRACT_EPOCHS**:
    - **TYPE**: 'int_uniform'
    - **RANGE**: [2,10]
- **CUSTOM_RESNETV2**:
  - **LR**:
    - **TYPE**: 'float_log'
    - **RANGE**: [ 0.00001, 0.001 ]
  - **DROPOUT0**:
    - **TYPE**: 'float_uniform'
    - **RANGE**: [ 0.0, 0.5 ]
  - **DROPOUT1**:
    - **TYPE**: 'float_uniform'
    - **RANGE**: [ 0.2, 0.5 ]
  - **BLOCKS**:
    - **TYPE**: 'int_uniform'
    - **RANGE**: [1, 3]
  - **INIT_FILTERS**:
    - **TYPE**: 'set'
    - **RANGE**: [16, 32]
</details>

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories. Disregard any _.\__init\__.py_ files, as they are
empty files that enable Python to recognize certain directories as
packages.

```
├── img
|   ├── experiments                  <- Visualizations for experiments
|   ├── heatmaps                     <- Grad-CAM heatmap images
|   └── readme                       <- Image assets for README.md
├── results
│   ├── figures                      <- UPDATE ME
│   └── logs                         <- TensorBoard logs
├── src
│   ├── data
|   |   ├── build-database.py        <- UPDATE ME!
|   |   ├── batabase_pull.py         <- UPDATE ME!
|   |   └── query_to_df.py           <- UPDATE ME!
│   ├── explainability
|   |   └── gradcam.py               <- Script containing gradcam application and heatmap generation
│   ├── models                       
|   |   └── models.py                <- Script containing all model definitions
│   ├── notebooks
|   |   └── .gitkeep                 <- UPDATE ME!
|   ├── visualization                
|   |   └── visualize.py             <- Script for visualization production
|   ├── predict.py                   <- Script for prediction on raw data using trained models
|   └── train.py                     <- Script for training experiments
|
├── .gitignore                       <- Files to be be ignored by git.
├── config.yml                       <- Values of several constants used throughout project
├── README.md                        <- Project description
└── requirements.txt                 <- Lists all dependencies and their respective versions
```

## Contacts

**Blake VanBerlo**  
Title   
Org  
Email (waterloo email)

**Robert Arntfield**  
Title   
Org  
Email (lhsc email) 

**Derek Wu**  
Title  
Org  
Email (lhsc email) 

