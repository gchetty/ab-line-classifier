import os

from typing import Tuple, Dict, Callable, Union

import wandb
import pandas as pd
import tensorflow as tf

from src.data.preprocessor import Preprocessor

def get_train_val_test_artifact(
    run: wandb.sdk.wandb_run,
    artifact_version: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Get training, validation and test DataFrames from wandb artifact registry
    :param run: wandb run
    :param artifact_version: version of artifact stored in wandb
    :return: (Training DataFrame, validation DataFrame, test DataFrame, directory str for frames)
    """

    # uses the latest version of Images artifact if no version is specified
    train_val_test_version = artifact_version if artifact_version else 'latest'

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


def get_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    frames_key: str,
    target_key: str,
    frames_dir: str,
    preprocessing_fn: Callable,
    preprocessing_class: Union[Preprocessor],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Get training, validation and test datasets from DataFrames
    :param train_df: DataFrame of training data
    :param val_df: DataFrame of validation data
    :param test_df: DataFrame of testing data
    :param frames_key: key in DataFrame that has the paths of the frames
    :param target_key: key in DataFrame that corresponds to the model target (y)
    :param frames_dir: path to directory that has the frames
    :param preprocessing_fn: function that defines preprocessing steps
    :param preprocessing_class: class that is used to perform preprocessing function
    :return: (Training DataFrame, validation DataFrame, test DataFrame, directory str for frames)
    """

    # Create TF datasets for training, validation and test sets
    train_set = tf.data.Dataset.from_tensor_slices(
        ([os.path.join(frames_dir, f) for f in train_df[frames_key].tolist()], train_df[target_key]))
    val_set = tf.data.Dataset.from_tensor_slices(
        ([os.path.join(frames_dir, f) for f in val_df[frames_key].tolist()], val_df[target_key]))
    test_set = tf.data.Dataset.from_tensor_slices(
        ([os.path.join(frames_dir, f) for f in test_df[frames_key].tolist()], test_df[target_key]))

    # Set up preprocessing transformations to apply to each item in dataset
    preprocessor = preprocessing_class(preprocessing_fn)
    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    return train_set, val_set, test_set
