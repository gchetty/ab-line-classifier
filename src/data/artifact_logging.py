import wandb
import yaml
import os
import math
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def log_artifact(
    project_name: str,
    entity_name: str,
    paths: Dict[str, str],
    artifact_name: str,
    artifact_type: str,
    artifact_description: str,
    artifact_metadata: str
) -> None:
    """
    generic function to log an artifact in wandb
    :param project_name: name of wandb project
    :param entity_name: name of wandb entity
    :param paths: dictionary containing local path (directory) as key and path in wandb artifact as value
    :param artifact_name: name of dataset/artifact (in wandb)
    :param artifact_type: specifies what type of artifact it is on wandb - e.g. dataset
    :param artifact_description: description of artifact that can be seen in wandb
    :param artifact_metadata: metadata of artifact that can be seen in wandb
    """

    with wandb.init(project=project_name, entity=entity_name,
                    job_type="log-artifact") as run:

        # Creates artifact
        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type,
            description=artifact_description,
            metadata=artifact_metadata
        )

        # Adds every path that is to be added to the artifact
        for local_path, artifact_path in paths.items():
            artifact.add_dir(local_path, name=artifact_path)

        # Save the artifact to W&B.
        run.log_artifact(artifact)


def log_images_artifact(
    cfg: Dict[str, str],
) -> None:
    """
    function to log Deep Breathe images artifact in wandb
    :param cfg: configuration info for project containing various wandb project info, artifact paths, and metadata
    """

    with wandb.init(project=cfg['WANDB']['PROJECT_NAME'], entity=cfg['WANDB']['ENTITY'],
                    job_type="log-artifact") as run:

        # Creates artifact
        artifact = wandb.Artifact(
            'Images',
            type='dataset',
            description='Holds images, images table, clips table, and sql query.',
            metadata={
                'automask_version': cfg['DATA']['AUTOMASK']['VERSION'],
                'automask_output_format': cfg['DATA']['AUTOMASK']['OUTPUT_FORMAT'],
                'automask_edge_preserve': cfg['DATA']['AUTOMASK']['EDGE_PRESERVE'],
                'automask_save_cropped_roi': cfg['DATA']['AUTOMASK']['SAVE_CROPPED_ROI']
            }
        )

        artifact.add_dir(cfg['PATHS']['FRAMES'], name='images/')
        artifact.add_file(cfg['PATHS']['QUERY_TABLE'], name='clips_table.csv')
        artifact.add_file(cfg['PATHS']['FRAME_TABLE'], name='images.csv')
        artifact.add_file(cfg['PATHS']['DATABASE_QUERY'], name='clips_query.sql')

        # Save the artifact to W & B.
        run.log_artifact(artifact)

def log_model_dev_holdout_artifacts(
    cfg: Dict[str, str],
) -> None:
    """
    function to log Deep Breathe model development and holdout artifacts in wandb
    :param cfg: configuration info for project containing various wandb project info, artifact paths, and metadata
    """

    with wandb.init(project=cfg['WANDB']['PROJECT_NAME'], entity=cfg['WANDB']['ENTITY'],
                    job_type="log-artifact") as run:

        # uses the latest version of Images artifact if no version is specified
        images_version = cfg['WANDB']['IMAGES_ARTIFACT_VERSION'] if cfg['WANDB']['IMAGES_ARTIFACT_VERSION'] else 'latest'

        # downloads previously logged Images artifact
        images_artifact = run.use_artifact(f'Images:{images_version}')
        images_artifact_clips_table = images_artifact.get_path('clips_table.csv')
        images_artifact_images_table = images_artifact.get_path('images.csv')
        images_artifact_clips_table = images_artifact_clips_table.download()
        images_artifact_images_table = images_artifact_images_table.download()

        # clips table DataFrame from Images artifact
        clips_table_df = pd.read_csv(images_artifact_clips_table)
        # images table DataFrame from Images artifact
        images_table_df = pd.read_csv(images_artifact_images_table)

        # math to convert percentage holdout to a number of folds
        n_splits = math.floor(1.0/cfg['DATA']['HOLDOUT_ARTIFACT_SPLIT'])

        # patient_id_list need to ensure that patient_id is only found in train and validation group
        patient_id_list = np.array(images_table_df.patient_id.values)
        y_labels = images_table_df.Class.values

        # splits data into k folds and takes the indices of the first fold to correspond with a single train-test split
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg['WANDB']['ARTIFACT_SEED'])
        model_dev_index, holdout_index = next(sgkf.split(images_table_df, y_labels, groups=patient_id_list))

        # subsets the images DataFrame based on the split indices generated
        model_dev_images_df = images_table_df.iloc[model_dev_index]
        holdout_images_df = images_table_df.iloc[holdout_index]

        # subsets of clips table for each artifact type
        model_dev_clips_df = generate_clips_table_subset(clips_table_df, model_dev_images_df)
        holdout_clips_df = generate_clips_table_subset(clips_table_df, holdout_images_df)

        # save data for artifacts that are generated
        model_dev_images_df.to_csv(cfg['PATHS']['MODEL_DEV_IMAGES_PATH'], index=False)
        model_dev_clips_df.to_csv(cfg['PATHS']['MODEL_DEV_CLIPS_PATH'], index=False)
        holdout_images_df.to_csv(cfg['PATHS']['HOLDOUT_IMAGES_PATH'], index=False)
        holdout_clips_df.to_csv(cfg['PATHS']['HOLDOUT_CLIPS_PATH'], index=False)

        model_dev_artifact = create_model_dev_holdout_artifact(
            cfg=cfg,
            images_artifact_version=images_artifact.version,
            artifact_name='ModelDev',
            artifact_description='Images table and clips table for model research and development.',
            images_path=cfg['PATHS']['MODEL_DEV_IMAGES_PATH'],
            clips_path=cfg['PATHS']['MODEL_DEV_CLIPS_PATH']
        )

        holdout_artifact = create_model_dev_holdout_artifact(
            cfg=cfg,
            images_artifact_version=images_artifact.version,
            artifact_name='Holdout',
            artifact_description='Images table and clips table held out for final model validation.',
            images_path=cfg['PATHS']['HOLDOUT_IMAGES_PATH'],
            clips_path=cfg['PATHS']['HOLDOUT_CLIPS_PATH']
        )

        run.log_artifact(model_dev_artifact)
        run.log_artifact(holdout_artifact)

def create_model_dev_holdout_artifact(
    cfg: Dict[str, str],
    images_artifact_version: str,
    artifact_name: str,
    artifact_description: str,
    images_path: str,
    clips_path: str
) -> wandb.Artifact:
    """
    function used to create either an individual Deep Breathe model development or holdout artifact in wandb
    :param cfg: configuration info for project containing various wandb project info, artifact paths, and metadata
    :param images_artifact_version: version identifier of the Images artifact used to generate this artifact
    :param artifact_name: name of dataset/artifact (in wandb)
    :param artifact_description: description of artifact that can be seen in wandb
    :param images_path: local path to images table
    :param clips_path: local path to clips table

    """

    # Creates artifact
    artifact = wandb.Artifact(
        artifact_name,
        type='dataset',
        description=artifact_description,
        metadata={
            'images_artifact_version': images_artifact_version,
            'holdout_split': cfg['DATA']['HOLDOUT_ARTIFACT_SPLIT'],
            'random_seed': cfg['WANDB']['ARTIFACT_SEED']
        }
    )

    # Adds relevant files for this type of artifact
    artifact.add_file(images_path, name='images.csv')
    artifact.add_file(clips_path, name='clips_table.csv')

    return artifact


def generate_clips_table_subset(
    original_clips_table: pd.DataFrame,
    images_table_subset: pd.DataFrame
) -> pd.DataFrame:
    """
    function used to generate a subset of the clips table corresponding to the subset of the images table
    :param original_clips_table: DataFrame containing clips table before a subset of clip ids is taken
    :param images_table_subset: DataFrame that has clip ids that are a subset of the over clips table
    """

    # generates a subset of clip table by matching the first instance of clip id found in the images table subset
    images_unique_id = images_table_subset.drop_duplicates(subset='id')
    clips_table_subset = pd.merge(original_clips_table, images_unique_id, how='inner', on='id',
                                  suffixes=('', '_discard'))

    # discard unnecessary or duplicate columns that are not originally found in clips table
    clips_table_subset.drop(['patient_id_discard', 'Class Name', 'Frame Path', 'Class'], axis=1, inplace=True)

    return clips_table_subset

if __name__ == "__main__":
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    log_images_artifact(cfg)
    log_model_dev_holdout_artifacts(cfg)

