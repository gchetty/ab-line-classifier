import os
import datetime
import numpy as np
from math import ceil
from typing import Tuple, Dict, Callable, Union

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
from src.train_utils import get_train_val_test_artifact, get_datasets

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


def compute_class_weight(train_df: pd.DataFrame) -> Dict:
    """
    Computes weights for each class to be applied in the loss function during training. For class imbalance strategy.
    :param train_df: training DataFrame
    :return: A dictionary containing weights for each class
    """

    # Get class distribution
    histogram = np.bincount(train_df['Class'].to_numpy().astype(int))

    weights = [None] * len(histogram)
    for i in range(len(histogram)):
        weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
    class_weight = {i: weights[i] for i in range(len(histogram))}

    # Log class weight data to WandB
    class_weight_table = wandb.Table(columns=['Class', 'Weight'],
                                     data=[[i, weights[i]] for i in range(len(histogram))])
    wandb.log({'Class Weight': class_weight_table})

    # Log distribution data to WandB
    distribution_table = wandb.Table(
        columns=['Class', 'Num Samples'],
        data=[[cfg['DATA']['CLASSES'][i], histogram[i]] for (i, hist_data) in enumerate(histogram)]
    )
    wandb.log({'Training distribution': distribution_table})

    return class_weight


def compute_output_bias(train_df: pd.DataFrame) -> Constant:
    """
    Determines the bias on the model final layer
    :param train_df: training DataFrame
    :return: initializer that has constant values corresponding to the bias value
    """
    # Compute output bias
    histogram = np.bincount(train_df['Class'].astype(int))
    output_bias = Constant(np.log([histogram[i] / (np.sum(histogram) - histogram[i])
                                   for i in range(histogram.shape[0])]))
    return output_bias


def train_classifier(
        model_def: tf.data.Dataset,
        train_set: tf.data.Dataset,
        val_set: tf.data.Dataset,
        hparams: Dict,
        output_bias: Constant,
        class_weight: Dict,
        pretrained_path: str = None,
        save_weights: bool = False,
        verbose: bool = True):
    """
    :param model_def: Model definition function
    :param train_set: Training set of LUS frames
    :param val_set: Validation set of LUS frames
    :param hparams: Dict of hyperparameters
    :param output_bias: bias on the model final layer
    :param class_weight: a dictionary containing weights for each class
    :param pretrained_path: Path to pretrained weights. If None, trains the network from scratch.
    :param save_weights: Flag indicating whether to save the model's weights
    :param verbose: Whether to print out all epoch details
    :return: (model)
    """

    # Define performance metrics
    n_classes = len(cfg['DATA']['CLASSES'])
    threshold = 1.0 / n_classes  # Binary classification threshold for a class
    metrics = ['accuracy', AUC(name='auc')]
    metrics += [Precision(name='precision_' + cfg['DATA']['CLASSES'][c], thresholds=threshold, class_id=c) for c in
                range(n_classes)]
    metrics += [Recall(name='recall_' + cfg['DATA']['CLASSES'][c], thresholds=threshold, class_id=c) for c in
                range(n_classes)]

    # Define the model
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    model = model_def(hparams, input_shape, metrics, cfg['TRAIN']['N_CLASSES'],
                      mixed_precision=cfg['TRAIN']['MIXED_PRECISION'], output_bias=output_bias,
                      weights_path=pretrained_path)

    # Set training callbacks.
    callbacks = define_callbacks()

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

    return model


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

    callbacks = [early_stopping, reduce_lr, ClearMemory(), WandbMetricsLogger()]

    return callbacks


def log_test_results(model, test_set, test_df):
    """
    Visualize performance of a trained model on the test set. Optionally save the model.
    :param model: A trained TensorFlow model
    :param test_set: Test set of LUS frames
    :param test_df: A pd.DataFrame containing frame paths and labels for the test set
    """

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(test_set, verbose=1)
    test_metrics = {}
    for metric, value in zip(model.metrics_names, test_results):
        wandb.log[f"test/{metric}"] = value

    print(test_metrics)
    # Visualization of test results
    test_predictions = model.predict(test_set, verbose=0)
    test_labels = test_df['Class']
    plt = plot_roc(test_labels, test_predictions, cfg['DATA']['CLASSES'], dir_path=cfg['PATHS']['IMAGES'])
    roc_img = plot_to_tensor()
    plt = plot_confusion_matrix(test_labels, test_predictions, cfg['DATA']['CLASSES'], dir_path=cfg['PATHS']['IMAGES'])
    cm_img = plot_to_tensor()
    #
    # # Log test set results and plots in TensorBoard
    # writer = tf.summary.create_file_writer(logdir=log_dir)

    # Create table of test set metrics
    test_summary_str = [['**Metric**', '**Value**']]
    for metric in test_metrics:
        metric_values = test_metrics[metric]
        test_summary_str.append([metric, str(metric_values)])

    print(test_metrics)

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


def perform_single_run(hparams=None, save_weights=False, write_logs=False):
    '''
    Train a single model. Use the passed hyperparameters if possible; otherwise, use those in config.
    :param hparams: Dict of hyperparameters
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: Dictionary of test set performance metrics
    '''

    # Hardware configuration
    # If configuration is set, the model will train with a specified memory limit
    if cfg['TRAIN']['USE_MEMORY_LIMIT']:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(cfg['TRAIN']['MEMORY_LIMIT'])])

    # Initialize WandB run
    run = wandb.init(
        project=cfg['WANDB']['PROJECT_NAME'],
        job_type='train',
        entity=cfg['WANDB']['ENTITY'],
    )

    model_name = cfg['TRAIN']['MODEL_DEF'].upper()
    if cfg['TRAIN']['EXPERIMENT_TYPE'] == 'hparam_search':
        hparams = wandb.config
    else:
        wandb.config.update(cfg)

    # Used to specify hyperparameters if not specified or to add hyperparameters being held constant during
    # hyperparameter search
    if hparams is None:
        hparams = cfg['HPARAMS'][model_name]
    else:
        for hparam in cfg['HPARAMS'][model_name]:
            if hparam not in hparams:
                hparams[hparam] = cfg['HPARAMS'][model_name][hparam]

    model_def, preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])
    if write_logs:
        log_dir = os.path.join(cfg['PATHS']['LOGS'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        log_dir = None

    # Get TrainValTest artifact from wandb
    train_df, val_df, test_df, frames_dir = get_train_val_test_artifact(run,
                                                                        cfg['WANDB']['TRAIN_VAL_TEST_ARTIFACT_VERSION'])

    # Get datasets for training
    train_set, val_set, test_set = get_datasets(train_df, val_df, test_df, frames_key='Frame Path', target_key='Class',
                                                frames_dir=frames_dir, preprocessing_class=Preprocessor,
                                                preprocessing_fn=preprocessing_fn)

    # Determine class weight and output_bias to manage class_imbalances
    class_weight = compute_class_weight(train_df)
    output_bias = compute_output_bias(train_df)

    # Optionally get path to pretrained weights
    pretrained_path = cfg['PATHS']['PRETRAINED_WEIGHTS'] if cfg['TRAIN']['USE_PRETRAINED'] else None

    # Train the model
    model = train_classifier(model_def, train_set, val_set, hparams, output_bias=output_bias, class_weight=class_weight,
                             pretrained_path=pretrained_path, save_weights=save_weights)

    log_test_results(model, test_set, test_df)

    wandb.finish()


def configure_hyperparameter_sweep() -> None:
    """
    Translates experiment configuration into a wandb sweep configuration
    """

    sweep_cfg = {
        'method': 'bayes',
        'metric': {
            'goal': 'maximize',
            'name': 'epoch/val_auc'
        },
        'parameters': {}
    }

    # Translation from config parameters to wandb sweep configuration parameters
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
                parameter_config['distribution'] = 'log_uniform_values'
                parameter_config['min'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0]
                parameter_config['max'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1]
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_uniform':
                parameter_config['distribution'] = 'uniform'
                parameter_config['min'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0]
                parameter_config['max'] = cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1]
            sweep_cfg['parameters'][hparam_name] = parameter_config

    # Initialize sweep
    sweep_id = wandb.sweep(
        project=cfg['WANDB']['PROJECT_NAME'],
        entity=cfg['WANDB']['ENTITY'],
        sweep=sweep_cfg
    )

    return sweep_id


def train_experiment(experiment='single_train', save_weights=False, write_logs=False):
    '''
    Run a training experiment
    :param experiment: String defining which experiment to run
    :param save_weights: Flag indicating whether to save any models trained during the experiment
    :param write_logs: Flag indicating whether to write logs for training
    '''

    # Conduct the desired train experiment
    if experiment == 'single_train':
        perform_single_run(save_weights=save_weights, write_logs=write_logs)
    elif experiment == 'hparam_search':
        sweep_id = configure_hyperparameter_sweep()
        wandb.agent(sweep_id, function=perform_single_run, count=cfg['TRAIN']['HPARAM_SEARCH']['N_EVALS'])
    elif experiment == 'cross_validation':
        pass
    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT_TYPE field of config.yml.")
    return


if __name__ == '__main__':
    train_experiment(cfg['TRAIN']['EXPERIMENT_TYPE'], write_logs=True, save_weights=True)
