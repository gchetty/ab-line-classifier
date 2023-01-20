import glob
import logging
import os
from typing import List

import cv2
import mysql.connector
import pandas as pd
import yaml
from tqdm import tqdm

from src.data.dataset_creation import DatasetCreation

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


# Dataset creation for retrospective validation
class ABLineDatasetCreation(DatasetCreation):
    def __init__(self, cfg, database_cfg):
        super().__init__(cfg)
        self.database_cfg = database_cfg

    def mp4_to_images(self, mp4_path: str) -> List[str]:
        """
        Converts masked ultrasound mp4 video to a series of images and saves the images in the same directory.
        :param mp4_path: File name of the mp4 file to convert to series of images.
        """
        vc = cv2.VideoCapture(mp4_path)
        vid_dir, mp4_filename = os.path.split(mp4_path)  # Get folder and filename of mp4 file respectively
        mp4_filename = mp4_filename.split('.')[0]  # Strip file extension

        if not os.path.exists(self.cfg['PATHS']['FRAMES']):
            os.makedirs(self.cfg['PATHS']['FRAMES'])

        idx = 0
        image_paths = []

        if not os.path.isdir(self.cfg['PATHS']['FRAMES']):
            os.mkdir(self.cfg['PATHS']['FRAMES'])

        while True:
            ret, frame = vc.read()
            if not ret:
                break  # End of frames reached
            image_path = mp4_filename + '_' + str(idx) + '.jpg'
            image_paths.append(image_path)
            cv2.imwrite(self.cfg['PATHS']['FRAMES'] + image_path, frame)  # Save all the images out
            idx += 1
        return image_paths

    def build_dataset(self) -> None:
        """
        Create a dataset of frames, including their patient ID (if not real-time data) and class
        Dataset building is different whether dataset being created is from clips exported from the WaveBase device
        """

        query_df = pd.read_csv(self.cfg['PATHS']['CLIPS_TABLE'])
        clip_dfs = []

        for index, row in tqdm(query_df.iterrows()):
            for mp4_file in glob.glob(row['Path'] + '/' + row['filename'] + '.mp4'):
                image_paths = self.mp4_to_images(mp4_file)  # Convert mp4 encounter file to image files
                if self.cfg['DATA']['REAL_TIME_DATA']:  # Real-time clips aren't associated with patient IDs
                    clip_df = pd.DataFrame({'Frame Path': image_paths, 'Class': row['class'],
                                            'Class Name': cfg['DATA']['CLASSES'][row['class']]})
                else:
                    # Note that the id is key to linking the clips and image tables in dataset artifacts
                    clip_df = pd.DataFrame(
                        {'Frame Path': image_paths, 'patient_id': row['patient_id'], 'Class': row['class'],
                         'Class Name': cfg['DATA']['CLASSES'][row['class']], 'id': row['id']})
                clip_dfs.append(clip_df)

        all_clips_df = pd.concat(clip_dfs, axis=0, ignore_index=True)
        all_clips_df.to_csv(cfg['PATHS']['FRAME_TABLE'], index=False)
        return

    def query_to_df(self) -> pd.DataFrame:
        """
        Extracts out pertinent information from database query and builds a dataframe linking filenames, patients,
        and class
        """

        COLUMNS_WANTED = ['patient_id', 'a_or_b_lines', 'id']

        # Get database configs
        user = self.database_cfg['USERNAME']
        password = self.database_cfg['PASSWORD']
        host = self.database_cfg['HOST']
        db = self.database_cfg['DATABASE']

        # Establish connection to database
        conn = mysql.connector.connect(user=user, password=password, host=host, database=db)

        if conn.is_connected():
            logging.info("Connected to database")

            # Open sql file and read into DataFrame
            with open(self.cfg['PATHS']['DATABASE_QUERY'], 'r') as query_file:
                df = pd.read_sql(query_file.read(), conn)

            logging.info(df.head())

        else:
            logging.error("Couldn't connect to database")

        df.to_csv(self.cfg['PATHS']['QUERY_TABLE'], index=False)

        # Remove all muggle clips
        df = df[df.frame_homogeneity.isnull()]

        # Remove Non-A/Non-B line clips
        df = df[df.a_or_b_lines != 'non_a_non_b']

        # Removes clips with unlabelled parenchymal findings
        df = df[df.a_or_b_lines.notnull()]

        # Create filename
        df['filename'] = df['exam_id'] + "_" + df['patient_id'] + "_VID" + df["vid_id"].map(str)

        # Remove filenames that are NaNs because one of the fields used to create the filename is null
        df = df[df.filename.notnull()]

        # Create column of class category to each clip. 
        # Modifiable for binary or multi-class labelling
        df['class'] = df.apply(lambda row: 0 if row.a_or_b_lines == 'a_lines' else
        (1 if row.a_or_b_lines == 'b_lines_<_3' else
         (1 if row.a_or_b_lines == 'b_lines-_moderate_(<50%_pleural_line)' else
          (1 if row.a_or_b_lines == 'b_lines-_severe_(>50%_pleural_line)' else
           2 if row.a_or_b_lines == 'non_a_non_b' else
           -1))), axis=1)

        # Relabel all b-line severities as a single class for A- vs. B-line classifier
        df['a_or_b_lines'] = df['a_or_b_lines'].replace(
            {'b_lines_<_3': 'b_lines', 'b_lines-_moderate_(<50%_pleural_line)': 'b_lines',
             'b_lines-_severe_(>50%_pleural_line)': 'b_lines'})

        df['Path'] = df.apply(lambda row: self.cfg['PATHS']['MASKED_CLIPS'] + row.filename, axis=1)

        df['s3_path'] = df.apply(lambda row: row.s3_path, axis=1)

        # Finalize dataframe
        df = df[['filename'] + COLUMNS_WANTED + ['class'] + ['Path'] + ['s3_path']]

        # Save df - append this csv to the previous csv 'clips_by_patient_2.csv'
        df.to_csv(self.cfg['PATHS']['CLIPS_TABLE'], index=False)

        return df


if __name__ == '__main__':
    cfg = yaml.full_load(open(f"{os.getcwd()}/config.yml", 'r'))
    database_cfg = yaml.full_load(open(f"{os.getcwd()}/database_config.yml", 'r'))
    dataset_creation = ABLineDatasetCreation(cfg, database_cfg)
    dataset_creation.create_dataset()
