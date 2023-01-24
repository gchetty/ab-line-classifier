import os
import datetime
import numpy as np
from math import ceil
from typing import Tuple, Dict

import gc
import sys

import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from wandb.keras import WandbMetricsLogger

import pandas as pd
import yaml
import dill
import wandb

from src.models.models import *
from src.visualization.visualization import *
from src.data.preprocessor import Preprocessor

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

def get_training_artifacts(
        run: wandb.sdk.wandb_run,
        cfg: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get training, validation and test DataFrames from wandb artifact registry
    :param run: wandb run
    :param cfg: project configuration dictionary
    :return: (Training DataFrame, validation DataFrame, test DataFrame)
    """

    # uses the latest version of Images artifact if no version is specified
    train_val_test_version = cfg['WANDB']['TRAIN_VAL_TEST_ARTIFACT_VERSION'] if \
        cfg['WANDB']['TRAIN_VAL_TEST_ARTIFACT_VERSION'] else 'latest'

    # downloads previously logged artifacts
    train_val_test_artifact = run.use_artifact(f'TrainValTest:{train_val_test_version}')
    model_dev_artifact_version = train_val_test_artifact.metadata['model_dev_artifact_version']
    model_dev_artifact = run.use_artifact(f'ModelDev:{model_dev_artifact_version}')
    images_artifact_version = model_dev_artifact.metadata['images_artifact_version']
    images_artifact = run.use_artifact(f'Images:{images_artifact_version}')

    frames_dir = f"{images_artifact.download()}/images"
    train_val_test_images_path = f"{train_val_test_artifact.download()}/images"

    train_df = pd.read_csv(f"{train_val_test_images_path}/train.csv")
    val_df = pd.read_csv(f"{train_val_test_images_path}/val.csv")
    test_df = pd.read_csv(f"{train_val_test_images_path}/test.csv")

    return train_df, val_df, test_df, frames_dir


def train_model_single(cfg, model_def, preprocessing_fn, train_df, val_df, test_df, frames_dir, hparams,
                       pretrained_path=None, save_weights=False, log_dir=None, verbose=True):
    """
    :param
    :param model_def: Model definition function
    :param preprocessing_fn: Model-specific preprocessing function
    :param train_df: Training set of LUS frames
    :param val_df: Validation set of LUS frames
    :param test_df: Test set of LUS frames
    :param hparams: Dict of hyperparameters
    :param pretrained_path: Path to pretrained weights. If None, trains the network from scratch.
    :param save_weights: Flag indicating whether to save the model's weights
    :param log_dir: TensorBoard logs directory
    :param verbose: Whether to print out all epoch details
    :return: (model, test_metrics, test_generator)
    """

    # Create TF datasets for training, validation and test sets
    train_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in train_df['Frame Path'].tolist()], train_df['Class']))
    val_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in val_df['Frame Path'].tolist()], val_df['Class']))
    test_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in test_df['Frame Path'].tolist()], test_df['Class']))

    # Set up preprocessing transformations to apply to each item in dataset
    preprocessor = Preprocessor(preprocessing_fn)
    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    # Apply class imbalance strategy. We have many more X-rays negative for COVID-19 than positive.
    histogram = np.bincount(train_df['Class'].to_numpy().astype(int))  # Get class distribution
    class_weight = get_class_weights_wandb(histogram)

    # Define performance metrics
    n_classes = len(cfg['DATA']['CLASSES'])
    threshold = 1.0 / n_classes # Binary classification threshold for a class
    metrics = ['accuracy', AUC(name='auc')]
    metrics += [Precision(name='precision_' + cfg['DATA']['CLASSES'][c], thresholds=threshold, class_id=c) for c in range(n_classes)]
    metrics += [Recall(name='recall_' + cfg['DATA']['CLASSES'][c], thresholds=threshold, class_id=c) for c in range(n_classes)]

    # Log distribution data to WandB
    distribution_table = wandb.Table(
        columns=['Class', 'Num Samples'],
        data=[[cfg['DATA']['CLASSES'][i], histogram[i]] for (i, hist_data) in enumerate(histogram)]
    )
    wandb.log({'Training distribution': distribution_table})

    input_shape = cfg['DATA']['IMG_DIM'] + [3]

    # Compute output bias
    histogram = np.bincount(train_df['Class'].astype(int))
    output_bias = Constant(np.log([histogram[i] / (np.sum(histogram) - histogram[i])
                                      for i in range(histogram.shape[0])]))

    # Define the model
    model = model_def(hparams, input_shape, metrics, cfg['TRAIN']['N_CLASSES'],
                      mixed_precision=cfg['TRAIN']['MIXED_PRECISION'], output_bias=output_bias,
                      weights_path=pretrained_path)

    # Set training callbacks.
    callbacks = define_callbacks()
    callbacks.append(WandbMetricsLogger())

    # Train the model.
    history = model.fit(train_set, epochs=cfg['TRAIN']['EPOCHS'], validation_data=val_set, callbacks=callbacks,
                         verbose=verbose, class_weight=class_weight)

    # Save the model's weights
    if save_weights:
        model_path = cfg['PATHS']['MODEL_WEIGHTS'] + 'model' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
        if cfg['TRAIN']['MODEL_DEF'] == 'cutoffvgg16':
            save_model(model.model, model_path)
        else:
            save_model(model, model_path)  # Save the model's weights

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(test_set, verbose=1)
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        test_summary_str.append([metric, str(value)])
    if log_dir is not None:
        log_test_results(model, test_set, test_df, test_metrics, log_dir)
    return model, test_metrics, test_set

def get_class_weights(histogram):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param histogram: A list depicting the number of each item in different class
    :param class_multiplier: List of values to multiply the calculated class weights by. For further control of class weighting.
    :return: A dictionary containing weights for each class
    '''
    weights = [None] * len(histogram)
    for i in range(len(histogram)):
        weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
    class_weight = {i: weights[i] for i in range(len(histogram))}
    print("Class weights: ", class_weight)
    return class_weight

def get_class_weights_wandb(histogram):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param histogram: A list depicting the number of each item in different class
    :param class_multiplier: List of values to multiply the calculated class weights by. For further control of class weighting.
    :return: A dictionary containing weights for each class
    '''
    weights = [None] * len(histogram)
    for i in range(len(histogram)):
        weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
    class_weight = {i: weights[i] for i in range(len(histogram))}
    # Log distribution data to WandB
    class_weight_table = wandb.Table(columns=['Class', 'Weight'],
                                     data=[[i, weights[i]] for i in range(len(histogram))])
    wandb.log({'Class Weight': class_weight_table})
    return class_weight


def define_callbacks():
    '''
    Defines a list of Keras callbacks to be applied to model training loop
    :param cfg: Project config object
    :return: list of Keras callbacks
    '''
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min',
                                   restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=cfg['TRAIN']['PATIENCE'] // 2, verbose=1,
                                  min_lr=1e-8, min_delta=0.0001)

    class ClearMemory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            k.clear_session()

    callbacks = [early_stopping, reduce_lr, ClearMemory()]

    return callbacks


def partition_dataset(val_split, test_split, save_dfs=True, seed=None):
    '''
    Partition the frame_df into training, validation and test sets by patient_id
    :param val_split: Validation split (in range [0, 1])
    :param test_split: Test split (in range [0, 1])
    :param save_dfs: Flag indicating whether to save the splits
    :return: (Training DataFrame, validation DataFrame, test DataFrame)
    '''

    frame_df = pd.read_csv(cfg['PATHS']['FRAME_TABLE'])
    all_pts = frame_df['patient_id'].unique()  # Get list of patient_ids
    relative_val_split = val_split / (1 - (test_split))
    trainval_pts, test_pts = train_test_split(all_pts, test_size=test_split, random_state=seed)
    train_pts, val_pts = train_test_split(trainval_pts, test_size=relative_val_split, random_state=seed)

    train_df = frame_df[frame_df['patient_id'].isin(train_pts)]
    val_df = frame_df[frame_df['patient_id'].isin(val_pts)]
    test_df = frame_df[frame_df['patient_id'].isin(test_pts)]

    if not os.path.exists(cfg['PATHS']['PARTITIONS']):
        os.makedirs(cfg['PATHS']['PARTITIONS'])

    if save_dfs:
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_df.to_csv(cfg['PATHS']['PARTITIONS'] + 'train_set.csv')
        val_df.to_csv(cfg['PATHS']['PARTITIONS'] + 'val_set.csv')
        test_df.to_csv(cfg['PATHS']['PARTITIONS'] + 'test_set.csv')
    return train_df, val_df, test_df


def log_test_results(model, test_set, test_df, test_metrics, log_dir):
    '''
    Visualize performance of a trained model on the test set. Optionally save the model.
    :param model: A trained TensorFlow model
    :param test_set: A tf.data Dataset for the test set
    :param test_df: A pd.DataFrame containing frame paths and labels for the test set
    :param test_metrics: Dict of test set performance metrics
    :param log_dir: Path to write TensorBoard logs
    '''

    # Visualization of test results
    test_predictions = model.predict(test_set, verbose=0)
    test_labels = test_df['Class']
    plt = plot_roc(test_labels, test_predictions, cfg['DATA']['CLASSES'], dir_path=cfg['PATHS']['IMAGES'])
    roc_img = plot_to_tensor()
    plt = plot_confusion_matrix(test_labels, test_predictions, cfg['DATA']['CLASSES'], dir_path=cfg['PATHS']['IMAGES'])
    cm_img = plot_to_tensor()

    # Log test set results and plots in TensorBoard
    writer = tf.summary.create_file_writer(logdir=log_dir)

    # Create table of test set metrics
    test_summary_str = [['**Metric**','**Value**']]
    for metric in test_metrics:
        metric_values = test_metrics[metric]
        test_summary_str.append([metric, str(metric_values)])

    # Create table of model and train hyperparameters used in this experiment
    hparam_summary_str = [['**Variable**', '**Value**']]
    for key in cfg['TRAIN']:
        hparam_summary_str.append([key, str(cfg['TRAIN'][key])])
    for key in cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()]:
        hparam_summary_str.append([key, str(cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()][key])])

    # Write to TensorBoard logs
    with writer.as_default():
        tf.summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
        tf.summary.text(name='Run hyperparameters', data=tf.convert_to_tensor(hparam_summary_str), step=0)
        tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
        tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)
    return


def train_model(model_def, preprocessing_fn, train_df, val_df, test_df, hparams, pretrained_path=None,
                save_weights=False, log_dir=None, verbose=True):
    '''

    :param model_def: Model definition function
    :param preprocessing_fn: Model-specific preprocessing function
    :param train_df: Training set of LUS frames
    :param val_df: Validation set of LUS frames
    :param test_df: Test set of LUS frames
    :param hparams: Dict of hyperparameters
    :param pretrained_path: Path to pretrained weights. If None, trains the network from scratch.
    :param save_weights: Flag indicating whether to save the model's weights
    :param log_dir: TensorBoard logs directory
    :param verbose: Whether to print out all epoch details
    :return: (model, test_metrics, test_generator)
    '''

    # Create TF datasets for training, validation and test sets
    frames_dir = cfg['PATHS']['FRAMES']
    train_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in train_df['Frame Path'].tolist()], train_df['Class']))
    val_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in val_df['Frame Path'].tolist()], val_df['Class']))
    test_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in test_df['Frame Path'].tolist()], test_df['Class']))

    # Set up preprocessing transformations to apply to each item in dataset
    preprocessor = Preprocessor(preprocessing_fn)
    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    # Apply class imbalance strategy. We have many more X-rays negative for COVID-19 than positive.
    histogram = np.bincount(train_df['Class'].to_numpy().astype(int))  # Get class distribution
    class_weight = get_class_weights(histogram)

    # Define performance metrics
    n_classes = len(cfg['DATA']['CLASSES'])
    threshold = 1.0 / n_classes # Binary classification threshold for a class
    metrics = ['accuracy', AUC(name='auc')]
    metrics += [Precision(name='precision_' + cfg['DATA']['CLASSES'][c], thresholds=threshold, class_id=c) for c in range(n_classes)]
    metrics += [Recall(name='recall_' + cfg['DATA']['CLASSES'][c], thresholds=threshold, class_id=c) for c in range(n_classes)]

    print('Training distribution: ',
          ['Class ' + cfg['DATA']['CLASSES'][i] + ': ' + str(histogram[i]) + '. '
           for i in range(len(histogram))])
    input_shape = cfg['DATA']['IMG_DIM'] + [3]

    # Compute output bias
    histogram = np.bincount(train_df['Class'].astype(int))
    output_bias = Constant(np.log([histogram[i] / (np.sum(histogram) - histogram[i])
                                      for i in range(histogram.shape[0])]))

    # Define the model
    model = model_def(hparams, input_shape, metrics, cfg['TRAIN']['N_CLASSES'],
                      mixed_precision=cfg['TRAIN']['MIXED_PRECISION'], output_bias=output_bias,
                      weights_path=pretrained_path)

    # Set training callbacks.
    callbacks = define_callbacks()
    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Train the model.
    history = model.fit(train_set, epochs=cfg['TRAIN']['EPOCHS'], validation_data=val_set, callbacks=callbacks,
                         verbose=verbose, class_weight=class_weight)

    # Save the model's weights
    if save_weights:
        model_path = cfg['PATHS']['MODEL_WEIGHTS'] + 'model' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
        if cfg['TRAIN']['MODEL_DEF'] == 'cutoffvgg16':
            save_model(model.model, model_path)
        else:
            save_model(model, model_path)  # Save the model's weights

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(test_set, verbose=1)
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        test_summary_str.append([metric, str(value)])
    if log_dir is not None:
        log_test_results(model, test_set, test_df, test_metrics, log_dir)
    return model, test_metrics, test_set


def train_single(hparams=None, save_weights=False, write_logs=False):
    '''
    Train a single model. Use the passed hyperparameters if possible; otherwise, use those in config.
    :param hparams: Dict of hyperparameters
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: Dictionary of test set performance metrics
    '''

    if cfg['TRAIN']['USE_MEMORY_LIMIT']:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(cfg['TRAIN']['MEMORY_LIMIT'])])

    run = wandb.init(
        project=cfg['WANDB']['PROJECT_NAME'],
        job_type='train',
        entity=cfg['WANDB']['ENTITY'],
        config=cfg
    )

    train_df, val_df, test_df, frames_dir = get_training_artifacts(run, cfg)

    model_def, preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])
    if write_logs:
        log_dir = os.path.join(cfg['PATHS']['LOGS'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        log_dir = None

    # Specify hyperparameters if not already done
    if hparams is None:
        hparams = cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()]

    # Optionally get path to pretrained weights
    pretrained_path = cfg['PATHS']['PRETRAINED_WEIGHTS'] if cfg['TRAIN']['USE_PRETRAINED'] else None

    # Train the model
    model, test_metrics, _ = train_model_single(cfg, model_def, preprocessing_fn, train_df, val_df, test_df, frames_dir,
                                                hparams, save_weights=save_weights, log_dir=log_dir,
                                                pretrained_path=pretrained_path)

    print('Test set metrics: ', test_metrics)

    wandb.finish()

    return test_metrics, model


def cross_validation(frame_df=None, hparams=None, write_logs=False, save_weights=False):
    '''
    Perform k-fold cross-validation. Results are saved in CSV format.
    :param frame_df: A DataFrame consisting of the entire dataset of LUS frames
    :param save_weights: Flag indicating whether to save model weights
    :return DataFrame of metrics
    '''

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=cfg['TRAIN']['MEMORY_LIMIT'])])

    n_folds = cfg['TRAIN']['N_FOLDS']
    if frame_df is None:
        frame_df = pd.read_csv(cfg['PATHS']['FRAME_TABLE'])

    metrics = ['accuracy', 'auc', 'f1score']
    metrics += ['precision_' + c for c in cfg['DATA']['CLASSES']]
    metrics += ['recall_' + c for c in cfg['DATA']['CLASSES']]
    metrics_df = pd.DataFrame(np.zeros((n_folds + 2, len(metrics) + 1)), columns=['Fold'] + metrics)
    metrics_df['Fold'] = list(range(n_folds)) + ['mean', 'std']

    model_name = cfg['TRAIN']['MODEL_DEF'].lower()
    model_def, preprocessing_fn = get_model(model_name)
    hparams = cfg['HPARAMS'][model_name.upper()] if hparams is None else hparams

    if write_logs:
        log_dir = os.path.join(cfg['PATHS']['LOGS'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        log_dir = None

    all_pts = frame_df['patient_id'].unique()
    val_split = 1.0 / n_folds
    pt_k_fold = KFold(n_splits=n_folds, shuffle=True)

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    partition_path = os.path.join(cfg['PATHS']['PARTITIONS'], 'kfold' + cur_date)
    if not os.path.exists(partition_path):
        os.makedirs(partition_path)

    # Train a model n_folds times with different folds
    cur_fold = 0
    row_idx = 0
    for train_index, test_index in pt_k_fold.split(all_pts):
        print('Fitting model for fold ' + str(cur_fold))

        # Partition into training, validation and test sets for this fold
        trainval_pts = all_pts[train_index]
        train_pts, val_pts = train_test_split(trainval_pts, test_size=val_split)
        test_pts = all_pts[test_index]
        train_df = frame_df[frame_df['patient_id'].isin(train_pts)]
        val_df = frame_df[frame_df['patient_id'].isin(val_pts)]
        test_df = frame_df[frame_df['patient_id'].isin(test_pts)]
        train_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_train_set.csv'))
        val_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_val_set.csv'))
        test_df.to_csv(os.path.join(partition_path, 'fold_' + str(cur_fold) + '_test_set.csv'))

        # Train the model and evaluate performance on test set
        log_dir_fold = log_dir
        if write_logs:
            log_dir_fold = log_dir + 'fold' + str(cur_fold)
        model, test_metrics, _ = train_model(model_def, preprocessing_fn, train_df, val_df, test_df, hparams,
                                             save_weights=save_weights, log_dir=log_dir_fold)

        metrics_df['f1score'] = metrics_df['f1score'].astype(object)

        for metric in test_metrics:
            if metric in metrics_df.columns:
                metrics_df[metric][row_idx] = test_metrics[metric]
        row_idx += 1
        cur_fold += 1
        gc.collect()
        tf.keras.backend.clear_session()
        del model

    # Record mean and standard deviation of test set results
    for metric in metrics:
        metrics_df[metric][n_folds] = metrics_df[metric][0:-2].mean()

        if metric == 'f1score':
            f1_reshape = np.vstack(metrics_df[metric][0:-2])
            metrics_df[metric][n_folds + 1] = f1_reshape.std(axis=0, ddof=1)

        else:
            metrics_df[metric][n_folds + 1] = metrics_df[metric][0:-2].std()

    # Save results
    file_path = cfg['PATHS']['EXPERIMENTS'] + 'cross_val_' + model_name + \
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    metrics_df.to_csv(file_path, columns=metrics_df.columns, index_label=False, index=False)
    return metrics_df


def generate_sweep_cfg():
    sweep_cfg = {'method': 'bayes', 'parameters': {}}

    model_name = cfg['TRAIN']['MODEL_DEF'].upper()
    for hparam_name in cfg['HPARAM_SEARCH'][model_name]:
        if cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'] is not None:
            parameter_config = {}
            if cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'set':
                parameter_config['distribution'] = 'categorical'
                parameter_config['values'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE']
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'int_uniform':
                parameter_config['distribution'] = 'int_uniform'
                parameter_config['min'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0]
                parameter_config['max'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1]
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_log':
                parameter_config['distribution'] = 'log_uniform'
                parameter_config['min'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0]
                parameter_config['max'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1]
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_uniform':
                parameter_config['distribution'] = 'uniform'
                parameter_config['min'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0]
                parameter_config['max'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1]
            sweep_cfg['parameters'][hparam_name] = parameter_config

    return sweep_cfg

def bayesian_hparam_optimization():
    '''
    Conducts a Bayesian hyperparameter optimization, given the parameter ranges and selected model
    :return: Dict of hyperparameters deemed optimal
    '''

    sweep_cfg = generate_sweep_cfg()
    frame_df = pd.read_csv(cfg['PATHS']['FRAME_TABLE'])
    model_name = cfg['TRAIN']['MODEL_DEF'].upper()
    objective_metric = cfg['TRAIN']['HPARAM_SEARCH']['HPARAM_OBJECTIVE']
    results = {'Trial': [], objective_metric: []}
    dimensions = []
    default_params = []
    hparam_names = []
    for hparam_name in cfg['HPARAM_SEARCH'][model_name]:
        if cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'] is not None:
            if cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'set':
                dimensions.append(Categorical(categories=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'],
                                              name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'int_uniform':
                dimensions.append(Integer(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                          high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                          prior='uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_log':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='log-uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_uniform':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='uniform', name=hparam_name))
            default_params.append(cfg['HPARAMS'][model_name][hparam_name])
            hparam_names.append(hparam_name)
            results[hparam_name] = []
    print(hparam_names)


    def objective(vals):
        hparams = dict(zip(hparam_names, vals))
        for hparam in cfg['HPARAMS'][model_name]:
            if hparam not in hparams:
                hparams[hparam] = cfg['HPARAMS'][model_name][hparam]  # Add hyperparameters being held constant
        print('HPARAM VALUES: ', hparams)
        #scores = cross_validation(frame_df=frame_df, hparams=hparams, write_logs=False, save_weights=False)[objective_metric]
        #score = scores[scores.shape[0] - 2]     # Get the mean value for the error metric from the cross validation
        test_metrics, _ = train_single(hparams=hparams, save_weights=False, write_logs=False)
        score = 1.0 - test_metrics['auc']
        tf.keras.backend.clear_session()
        return score   # We aim to minimize error
    search_results = gp_minimize(func=objective, dimensions=dimensions, acq_func='EI',
                                 n_calls=cfg['TRAIN']['HPARAM_SEARCH']['N_EVALS'], verbose=True)
    print(search_results)

    # Create table to detail results
    trial_idx = 0
    for t in search_results.x_iters:
        results['Trial'].append(str(trial_idx))
        results[objective_metric].append(1.0 - search_results.func_vals[trial_idx])
        for i in range(len(hparam_names)):
            results[hparam_names[i]].append(t[i])
        trial_idx += 1
    results['Trial'].append('Best')
    results[objective_metric].append(1.0 - search_results.fun)
    for i in range(len(hparam_names)):
        results[hparam_names[i]].append(search_results.x[i])
    results_df = pd.DataFrame(results)
    results_path = cfg['PATHS']['EXPERIMENTS'] + 'hparam_search_' + model_name + \
                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    results_df.to_csv(results_path, index_label=False, index=False)
    plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=True)
    return search_results


def train_experiment(experiment='single_train', save_weights=False, write_logs=False):
    '''
    Run a training experiment
    :param experiment: String defining which experiment to run
    :param save_weights: Flag indicating whether to save any models trained during the experiment
    :param write_logs: Flag indicating whether to write logs for training
    '''

    # Conduct the desired train experiment
    if experiment == 'single_train':
        train_single(save_weights=save_weights, write_logs=write_logs)
    elif experiment == 'hparam_search':
        print("Performing hyperparameter sweep")
        bayesian_hparam_optimization()
    elif experiment == 'cross_validation':
        cross_validation(save_weights=save_weights, write_logs=write_logs)
    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT_TYPE field of config.yml.")
    return


if __name__=='__main__':
    train_experiment(cfg['TRAIN']['EXPERIMENT_TYPE'], write_logs=True, save_weights=True)
