import os
import datetime
import numpy as np
from math import ceil
from typing import Dict, Optional, Callable

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
from src.train_utils import get_train_val_test_artifact, get_datasets, generate_classification_test_results, \
    initialize_wandb_run, get_fold_artifact, get_n_folds, WandbGradcamEvalCallback

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
        model_def: Callable,
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
    callbacks = define_callbacks(cfg['TRAIN']['LOG_FREQ'])
    callbacks.append(
        WandbGradcamEvalCallback(
            val_set=val_set,
            data_table_columns=['idx', 'image', 'label'],
            pred_table_columns=['epoch', 'idx', 'image', 'label', 'probs', 'pred']
        )
    )

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


def define_callbacks(log_freq: int):
    '''
    Defines a list of Keras callbacks to be applied to model training loop
    :param log_freq: integer that represents after how many batches metrics are logged on wandb
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

    callbacks = [early_stopping, reduce_lr, ClearMemory(), WandbMetricsLogger(log_freq=log_freq)]

    return callbacks


def perform_single_run(
        save_weights: bool = False,
        group_id: Optional[str] = None,
        fold_id: Optional[int] = None,
) -> None:
    """
    Used to perform a single training run. Used in a variety of contexts - single training, cross-val, hyper-param search
    :param save_weights: Flag indicating whether to save the model's weights
    :param group_id: Optional string indicates a group id for the training run, used only for cross-validation
    :param fold_id: Optional fold id which specifies fold for validation, used only for cross-validation
    :return: Dictionary of test set performance metrics
    """

    # Hardware configuration
    # If configuration is set, the model will train with a specified memory limit
    if cfg['TRAIN']['USE_MEMORY_LIMIT']:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(cfg['TRAIN']['MEMORY_LIMIT'])])

    # Initialize wandb run based on experiment type
    run = initialize_wandb_run(project_name=cfg['WANDB']['PROJECT_NAME'], entity_name=cfg['WANDB']['ENTITY'],
                               experiment_type=cfg['TRAIN']['EXPERIMENT_TYPE'], group_id=group_id)

    # Code to get default hyperparameters from config unless overridden by sweep
    model_name = cfg['TRAIN']['MODEL_DEF'].upper()
    run.config.setdefaults(cfg['HPARAMS'][model_name])
    hparams = dict(wandb.config)

    # Store additional configuration information in wandb run
    wandb.config.update({
        'HPARAMS': hparams,
        'TRAIN': cfg['TRAIN'],
        'DATA': {
            'IMG_DIM': cfg['DATA']['IMG_DIM']
        }
    })
    if fold_id is not None:
        wandb.config.update({'FOLD_ID': fold_id})

    model_def, preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])

    experiment = cfg['TRAIN']['EXPERIMENT_TYPE']
    if experiment == 'cross_validation':
        train_df, val_df, test_df, frames_dir = get_fold_artifact(run,
                                                                  cfg['WANDB']['K_FOLD_CROSS_VAL_ARTIFACT_VERSION'],
                                                                  fold_id)
    else:
        # Get TrainValTest artifact from wandb
        train_df, val_df, test_df, frames_dir = get_train_val_test_artifact(run,
                                                                            cfg['WANDB']
                                                                            ['TRAIN_VAL_TEST_ARTIFACT_VERSION'])

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

    if test_set is not None:
        generate_classification_test_results(model, test_set, test_df, 'Class', cfg['DATA']['CLASSES'])

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


def train_experiment(experiment='single_train', save_weights=False):
    """
    Run a training experiment
    :param experiment: String defining which experiment to run
    :param save_weights: Flag indicating whether to save any models trained during the experiment
    """

    # Conduct the desired train experiment
    if experiment == 'single_train':
        perform_single_run(save_weights=save_weights)
    elif experiment == 'hparam_search':
        sweep_id = configure_hyperparameter_sweep()
        wandb.agent(sweep_id, function=perform_single_run, count=cfg['TRAIN']['HPARAM_SEARCH']['N_EVALS'])
    elif experiment == 'cross_validation':
        val_group_id = f'kfold-{wandb.util.generate_id()}'
        n_folds = get_n_folds(project_name=cfg['WANDB']['PROJECT_NAME'], entity_name=cfg['WANDB']['ENTITY'],
                              artifact_version=cfg['WANDB']['K_FOLD_CROSS_VAL_ARTIFACT_VERSION'])
        for fold_id in range(n_folds):
            perform_single_run(group_id=val_group_id, fold_id=fold_id)

    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT_TYPE field of config.yml.")
    return


if __name__ == '__main__':
    train_experiment(cfg['TRAIN']['EXPERIMENT_TYPE'], save_weights=True)
