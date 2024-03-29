import os
import logging
from typing import Tuple, Dict, Callable, Union, List, Optional

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf

from wandb.keras import WandbEvalCallback

from src.data.preprocessor import Preprocessor
from src.data.artifact_logging import group_train_test_split

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


def get_train_val_test_artifact(
    run: wandb.sdk.wandb_run,
    artifact_version: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Get training, validation and test DataFrames from wandb TrainValTest artifact
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

    frames_dir = f"{images_artifact.download()}/frames"
    train_val_test_frames_path = f"{train_val_test_artifact.download()}/frames"

    train_df = pd.read_csv(f"{train_val_test_frames_path}/train.csv")
    val_df = pd.read_csv(f"{train_val_test_frames_path}/val.csv")
    test_df = pd.read_csv(f"{train_val_test_frames_path}/test.csv")

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
    :return: (training DataFrame, validation DataFrame, test DataFrame, directory str for frames)
    """

    # Create TF datasets for training, validation and test sets
    train_set = tf.data.Dataset.from_tensor_slices(
        ([os.path.join(frames_dir, f) for f in train_df[frames_key].tolist()], train_df[target_key]))
    val_set = tf.data.Dataset.from_tensor_slices(
        ([os.path.join(frames_dir, f) for f in val_df[frames_key].tolist()], val_df[target_key]))
    if test_df is not None:
        test_set = tf.data.Dataset.from_tensor_slices(
            ([os.path.join(frames_dir, f) for f in test_df[frames_key].tolist()], test_df[target_key]))
    else:
        test_set = None

    # Set up preprocessing transformations to apply to each item in dataset
    preprocessor = preprocessing_class(preprocessing_fn)
    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    if test_set is not None:
        test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    return train_set, val_set, test_set

def generate_classification_test_results(
        model: tf.keras.Model,
        test_set: tf.data.Dataset,
        test_df: pd.DataFrame,
        target_key: str,
        classes: List[str]
) -> None:
    """
    Visualize performance of a trained classification model on the test set
    :param model: A trained TensorFlow model
    :param test_set: Test set of LUS frames
    :param test_df: A pd.DataFrame containing frame paths and labels for the test set
    :param target_key: key in DataFrame corresponding to model's target (y)
    :param classes: A list of classes
    """

    # Run the model on the test set and og the resulting performance metrics in wandb
    test_results = model.evaluate(test_set, verbose=1)
    test_metrics = {}
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[f'test/{metric}'] = value
    wandb.log(test_metrics)

    # Logging roc and confusion matrix visualizations of results
    test_probabilities = model.predict(test_set, verbose=0)
    test_labels = test_df[target_key]
    roc = wandb.plot.roc_curve(y_true=test_labels, y_probas=test_probabilities, labels=classes)
    wandb.log({'test/roc': roc})
    test_predictions = np.argmax(test_probabilities, axis=1)
    cm = wandb.plot.confusion_matrix(y_true=test_labels, preds=test_predictions, class_names=classes)
    wandb.log({'test/confusion_matrix': cm})

def initialize_wandb_run(
        project_name: str,
        entity_name: str,
        experiment_type: str,
        sweep_id: Optional[str] = None
) -> wandb.sdk.wandb_run:
    """
    Generates a wandb run type based on the experiment. This allows the same training function to be used for different
    types of experiments - training, k-fold cross-validation, hyperparameter sweeps e.t.c.
    :param project_name: name of wandb project
    :param entity_name: name of wandb entity
    :param experiment_type: defines what type of experiment is being run
    :param sweep_id: optional sweep_id argument to be used for cross-validation group id creation
    """

    # mapping to config experiment names to wandb job type names
    experiment_job_type_mapping = {
        'single_train': 'train',
        'cross_validation': 'k_fold_cross_val',
        'hparam_search': 'hparam_search'
    }

    # generate a group id for k-fold experiments only
    group_id = f'kfold-{sweep_id}' if experiment_type == 'cross_validation' else None

    job_type = experiment_job_type_mapping[experiment_type]

    # Initialize WandB run
    run = wandb.init(
        project=project_name,
        entity=entity_name,
        job_type=job_type,
        group=group_id
    )

    return run


def get_n_folds(
        project_name: str,
        entity_name: str,
        artifact_version: str,
) -> int:
    """
    Gets the number of folds in the KFoldCrossValidation artifact
    :param project_name: name of wandb project
    :param entity_name: name of wandb entity
    :param artifact_version: version of KFoldCrossValidation artifact stored in wandb
    :return: number of folds in the KFoldCrossValidation artifact
    """

    run = wandb.init(
        project=project_name,
        entity=entity_name,
        job_type="fetch_n_folds",
    )

    # uses the latest version of Images artifact if no version is specified
    k_folds_version = artifact_version if artifact_version else 'latest'

    # downloads previously logged artifacts
    k_folds_artifact = run.use_artifact(f'KFoldCrossValidation:{k_folds_version}')
    n_folds = k_folds_artifact.metadata['n_folds']

    wandb.finish()

    return n_folds

def get_fold_artifact(
        run: wandb.sdk.wandb_run,
        artifact_version: str,
        fold_id: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Get training, validation, test DataFrames and frames path from wandb KFoldCrossValidation artifact for given fold
    :param run: wandb run
    :param artifact_version: version of artifact stored in wandb
    :param fold_id: id of validation fold
    :return: (training DataFrame, validation DataFrame, test DataFrame, path to frames)
    """

    k_folds_version = artifact_version if artifact_version else 'latest'

    # downloads previously logged artifacts
    k_folds_artifact = run.use_artifact(f'KFoldCrossValidation:{k_folds_version}')
    model_dev_artifact_version = k_folds_artifact.metadata['model_dev_artifact_version']
    model_dev_artifact = run.use_artifact(f'ModelDev:{model_dev_artifact_version}')
    images_artifact_version = model_dev_artifact.metadata['images_artifact_version']
    images_artifact = run.use_artifact(f'Images:{images_artifact_version}')

    frames_dir = f"{images_artifact.download()}/frames"
    k_folds_dir = f"{k_folds_artifact.download()}"

    random_seed = k_folds_artifact.metadata['random_seed']
    val_split = k_folds_artifact.metadata['val_split']

    # Create a train_df from all folds except validation fold
    train_val_dfs = []
    test_df = None
    for filename in os.listdir(k_folds_dir):
        cur_fold_id = int(filename.split('_')[-1])
        fold_frames_path = os.path.join(k_folds_dir, filename, 'frames.csv')
        if cur_fold_id == fold_id:
            test_df = pd.read_csv(fold_frames_path)
        else:
            train_val_dfs.append(pd.read_csv(fold_frames_path))
    train_val_df = pd.concat(train_val_dfs)

    train_df, val_df = group_train_test_split(train_val_df, test_size=val_split, group_key='patient_id',
                                              target_key='Class', random_seed=random_seed)

    return train_df, val_df, test_df, frames_dir

# Evaluation callback generates prediction visualizations at the end of each epoch
class WandbGradcamEvalCallback(WandbEvalCallback):
    def __init__(self, val_set, data_table_columns, pred_table_columns):
        super().__init__(data_table_columns, pred_table_columns)
        self.val_set = val_set

    def add_ground_truth(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Add validation ground truth data to the callback's data table. Runs at training start.
        :param logs: optional wandb logs
        """
        for idx, (image, label) in enumerate(self.val_set.unbatch()):
            self.data_table.add_data(
                idx,
                wandb.Image(image),
                np.argmax(label, axis=-1)
            )

    def add_model_predictions(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Adds prediction data to the callback's prediction table. Run's at epoch end.
        :param logs: optional wandb logs
        """

        # Gets predictions using the latest model weights on validation data
        probs, preds = self._inference()

        # Gets the reference of data table previously logged (saves memory)
        data_table_ref = self.data_table_ref
        table_idxs = data_table_ref.get_index()

        for idx in table_idxs:
            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                data_table_ref.data[idx][2],
                probs[idx],
                preds[idx]
            )

    def _inference(self) -> Tuple[List, List]:
        """
        Uses the latest model to generate class predictions and their associated probabilities on validation data.
        """
        preds = []
        probs = []

        for image, label in self.val_set.unbatch():
            prob = np.array(self.model(tf.expand_dims(image, axis=0)))
            pred = np.argmax(prob, axis=-1)[0]
            preds.append(pred)
            probs.append(np.squeeze(prob))

        return probs, preds
