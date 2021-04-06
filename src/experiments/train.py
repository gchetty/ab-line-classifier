import pandas as pd
import os
import yaml
import dill
import datetime
import numpy as np
from math import ceil
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.models import *

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

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


def define_callbacks(patience):
    '''
    Defines a list of Keras callbacks to be applied to model training loop
    :param cfg: Project config object
    :return: list of Keras callbacks
    '''
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min',
                                   restore_best_weights=True)
    print(type(patience))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=cfg['TRAIN']['PATIENCE'] // 2 + 1, verbose=1,
                                  min_lr=1e-8, min_delta=0.0001)
    callbacks = [early_stopping]
    return callbacks


def partition_dataset(frame_df, val_split, test_split, save_dfs=True):
    '''
    Partition the frame_df into training, validation and test sets by patient ID
    :param frame_df: DataFrame consisting of LUS frames, their classes and patient IDs
    :param val_split: Validation split (in range [0, 1])
    :param test_split: Test split (in range [0, 1])
    :param save_dfs: Flag indicating whether to save the splits
    :return: (Training DataFrame, validation DataFrame, test DataFrame)
    '''

    all_pts = frame_df['Patient'].unique()  # Get list of patients
    relative_val_split = val_split / (1 - (test_split))
    trainval_pts, test_pts = train_test_split(all_pts, test_size=test_split)
    train_pts, val_pts = train_test_split(trainval_pts, test_size=relative_val_split)

    train_df = frame_df[frame_df['Patient'].isin(train_pts)]
    val_df = frame_df[frame_df['Patient'].isin(val_pts)]
    test_df = frame_df[frame_df['Patient'].isin(test_pts)]

    if not os.path.exists(cfg['PATHS']['PARTITIONS']):
        os.makedirs(cfg['PATHS']['PARTITIONS'])

    if save_dfs:
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_df.to_csv(cfg['PATHS']['PARTITIONS'] + 'train_set.csv')
        val_df.to_csv(cfg['PATHS']['PARTITIONS'] + 'val_set.csv')
        test_df.to_csv(cfg['PATHS']['PARTITIONS'] + 'test_set.csv')
    return train_df, val_df, test_df


def train_model(frame_df, callbacks, verbose=1):
    '''
    Train a and evaluate model on given data.
    :param cfg: Project config (from config.yml)
    :param data: dict of partitioned dataset
    :param callbacks: list of callbacks for Keras model
    :param verbose: Verbosity mode to pass to model.fit_generator()
    :return: Trained model and associated performance metrics on the test set
    '''

    train_df, val_df, test_df = partition_dataset(frame_df, cfg['DATA']['VAL_SPLIT'], cfg['DATA']['TEST_SPLIT'])
    model_def, preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])

    # Create ImageDataGenerators. For training data: randomly zoom, stretch, horizontally flip image as data augmentation.
    train_img_gen = ImageDataGenerator(zoom_range=cfg['TRAIN']['DATA_AUG']['ZOOM_RANGE'],
                                       horizontal_flip=cfg['TRAIN']['DATA_AUG']['HORIZONTAL_FLIP'],
                                       width_shift_range=cfg['TRAIN']['DATA_AUG']['WIDTH_SHIFT_RANGE'],
                                       height_shift_range=cfg['TRAIN']['DATA_AUG']['HEIGHT_SHIFT_RANGE'],
                                       shear_range=cfg['TRAIN']['DATA_AUG']['SHEAR_RANGE'],
                                       rotation_range=cfg['TRAIN']['DATA_AUG']['ROTATION_RANGE'],
                                       brightness_range=cfg['TRAIN']['DATA_AUG']['BRIGHTNESS_RANGE'],
                                       preprocessing_function=preprocessing_fn)
    val_img_gen = ImageDataGenerator(preprocessing_function=preprocessing_fn)
    test_img_gen = ImageDataGenerator(preprocessing_function=preprocessing_fn)

    # Create DataFrameIterators
    img_shape = tuple(cfg['DATA']['IMG_DIM'])
    x_col = 'Frame Path'
    y_col = 'Class Name'
    class_mode = 'categorical'
    train_generator = train_img_gen.flow_from_dataframe(dataframe=train_df, directory=cfg['PATHS']['FRAMES'],
                                                        x_col=x_col, y_col=y_col, target_size=img_shape,
                                                        batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                                        class_mode=class_mode, validate_filenames=True)
    val_generator = val_img_gen.flow_from_dataframe(dataframe=val_df, directory=cfg['PATHS']['FRAMES'],
                                                    x_col=x_col, y_col=y_col, target_size=img_shape,
                                                    batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                                    class_mode=class_mode, validate_filenames=True)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=test_df, directory=cfg['PATHS']['FRAMES'],
                                                      x_col=x_col, y_col=y_col, target_size=img_shape,
                                                      batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                                      class_mode=class_mode, validate_filenames=True, shuffle=False)

    # Save model's ordering of class indices
    dill.dump(train_generator.class_indices, open(cfg['PATHS']['CLASS_NAME_MAP'], 'wb'))

    # Apply class imbalance strategy. We have many more X-rays negative for COVID-19 than positive.
    histogram = np.bincount(np.array(train_generator.labels).astype(int))  # Get class distribution
    class_weight = get_class_weights(histogram)

    # Define performance metrics
    n_classes = len(cfg['DATA']['CLASSES'])
    threshold = 1.0 / n_classes # Binary classification threshold for a class
    metrics = ['accuracy', AUC(name='auc'), F1Score(name='f1score', num_classes=n_classes)]
    metrics += [Precision(name='precision_' + c, thresholds=threshold, class_id=train_generator.class_indices[c]) for c in train_generator.class_indices]
    metrics += [Recall(name='recall_' + c, thresholds=threshold, class_id=train_generator.class_indices[c]) for c in train_generator.class_indices]

    print('Training distribution: ',
          ['Class ' + list(train_generator.class_indices.keys())[i] + ': ' + str(histogram[i]) + '. '
           for i in range(len(histogram))])
    input_shape = cfg['DATA']['IMG_DIM'] + [3]

    # Compute output bias
    histogram = np.bincount(train_df['Class'].astype(int))
    output_bias = np.log([histogram[i] / (np.sum(histogram) - histogram[i]) for i in range(histogram.shape[0])])

    model = model_def(cfg['NN'][cfg['TRAIN']['MODEL_DEF'].upper()], input_shape, metrics, cfg['TRAIN']['N_CLASSES'],
                      mixed_precision=cfg['TRAIN']['MIXED_PRECISION'], output_bias=output_bias)

    # Train the model.
    steps_per_epoch = ceil(train_generator.n / train_generator.batch_size)
    val_steps = ceil(val_generator.n / val_generator.batch_size)
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=cfg['TRAIN']['EPOCHS'],
                        validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks,
                        verbose=verbose, class_weight=class_weight)

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(test_generator, verbose=1)
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        print(metric, ' = ', value)
        test_summary_str.append([metric, str(value)])
    return model, test_metrics, test_generator

def log_test_results(cfg, model, test_generator, test_metrics, log_dir):
    '''
    Visualize performance of a trained model on the test set. Optionally save the model.
    :param cfg: Project config
    :param model: A trained Keras model
    :param test_generator: A Keras generator for the test set
    :param test_metrics: Dict of test set performance metrics
    :param log_dir: Path to write TensorBoard logs
    '''

    # Visualization of test results
    test_predictions = model.predict(test_generator, verbose=0)
    test_labels = test_generator.labels
    plt = plot_roc(test_labels, test_predictions, list(test_generator.class_indices.keys()), dir_path=cfg['PATHS']['IMAGES'])
    roc_img = plot_to_tensor()
    plt = plot_confusion_matrix(test_labels, test_predictions, list(test_generator.class_indices.keys()), dir_path=cfg['PATHS']['IMAGES'])
    cm_img = plot_to_tensor()

    # Log test set results and plots in TensorBoard
    writer = tf_summary.create_file_writer(logdir=log_dir)

    # Create table of test set metrics
    test_summary_str = [['**Metric**','**Value**']]
    for metric in test_metrics:
        metric_values = test_metrics[metric]
        test_summary_str.append([metric, str(metric_values)])

    # Create table of model and train hyperparameters used in this experiment
    hparam_summary_str = [['**Variable**', '**Value**']]
    for key in cfg['TRAIN']:
        hparam_summary_str.append([key, str(cfg['TRAIN'][key])])
    for key in cfg['NN'][cfg['TRAIN']['MODEL_DEF'].upper()]:
        hparam_summary_str.append([key, str(cfg['NN'][cfg['TRAIN']['MODEL_DEF'].upper()][key])])

    # Write to TensorBoard logs
    with writer.as_default():
        tf_summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
        tf_summary.text(name='Run hyperparameters', data=tf.convert_to_tensor(hparam_summary_str), step=0)
        tf_summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
        tf_summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)
    return

def train_experiment(save_weights=True, write_logs=True):
    '''
    Defines and trains COVID US model according to selected experiment type. Prints and logs relevant metrics.
    :param experiment: The type of training experiment. Choices are currently {'single_train'}
    :param save_weights: A flag indicating whether to save the model weights
    :param write_logs: A flag indicating whether to write TensorBoard logs
    :return: A dictionary of metrics on the test set
    '''
    
    # Enable mixed precision if desired
    if cfg['TRAIN']['MIXED_PRECISION']:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # Set logs directory
    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = cfg['PATHS']['LOGS'] + "training/" + cur_date if write_logs else None
    if not os.path.exists(cfg['PATHS']['LOGS'] + "training/"):
        os.makedirs(cfg['PATHS']['LOGS'] + "training/")
    if sys.platform.startswith('win'):
        log_dir = log_dir.replace('/', '\\')    # On Windows, path separators must be '\\' to work with TensorBoard

    # Set training callbacks.
    callbacks = define_callbacks(cfg)

    if write_logs:
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)
    model, test_metrics, test_generator = train_model(frame_df, callbacks, verbose=1)
    if write_logs:
        log_test_results(cfg, model, test_generator, test_metrics, log_dir)
    
    if save_weights:
        model_path = cfg['PATHS']['MODEL_WEIGHTS'] + 'model' + cur_date + '.h5'
        save_model(model, model_path)  # Save the model's weights
    
    return


if __name__=='__main__':

    frame_df = pd.read_csv(cfg['PATHS']['FRAME_TABLE'])
    callbacks = define_callbacks(cfg['TRAIN']['PATIENCE'])
    model, test_metrics, test_generator = train_experiment(frame_df, callbacks)
