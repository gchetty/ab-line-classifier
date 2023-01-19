import logging
import os
import urllib
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import wget
import yaml
from tqdm import tqdm

from src.data.auto_masking import UnetSegmentation

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class DatasetCreation(ABC):
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        
    # takes the dataset from sql query csv to a full dataset artifact
    def create_dataset(self) -> None:
        logging.info("Running query to df process...")
        self.query_to_df()
        logging.info("Running database pull process...")
        self.database_pull()
        logging.info("Running auto-masking process...")
        self.auto_mask()
        logging.info("Performing final dataset build...")
        self.build_dataset()

    @abstractmethod
    def query_to_df(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def build_dataset(self) -> None:
        pass

    def database_pull(self) -> None:
        '''
        Pull raw clips from AWS database using a generated query in csv format
        '''
        output_folder = self.cfg['PATHS']['RAW_CLIPS']
        df = pd.read_csv(self.cfg['PATHS']['CLIPS_TABLE'])

        if not os.path.isdir(self.cfg['PATHS']['RAW_CLIPS']):
            os.mkdir(self.cfg['PATHS']['RAW_CLIPS'])

        logging.info('Getting AWS links...')

        # Dataframe of all clip links
        links = df.s3_path

        logging.info('Fetching clips from AWS...')

        # dictionary to store the different warnings/errors
        warning_counts = {}

        # Download clips and save to disk
        for link in tqdm(links):
            logging.info(link)
            firstpos = link.rfind("/")
            lastpos = link.rfind("-")
            filename = link[firstpos+1:lastpos] + '.mp4'

            try:
                wget.download(link, output_folder + filename)
                
            except urllib.error.HTTPError as e:
                if e in warning_counts:
                    warning_counts[e] += 1
                else:
                    warning_counts[e] = 1
        
        logging.info('Fetched clips successfully!')
        for k, v in warning_counts.items():
            logging.warning(f"{k} occured {v} times")

    def auto_mask(self) -> None:
        unet_seg = UnetSegmentation()
        unet_seg.predict(
            input_paths=self.cfg['PATHS']['RAW_CLIPS'], 
            output_path=self.cfg['PATHS']['MASKED_CLIPS'], 
            model_path=self.cfg['PATHS']['AUTOMASK_MODEL_PATH'], 
            output_format=self.cfg['DATA']['AUTOMASK']['OUTPUT_FORMAT'], 
            edge_preserve=self.cfg['DATA']['AUTOMASK']['EDGE_PRESERVE'],
            save_cropped_roi=self.cfg['DATA']['AUTOMASK']['SAVE_CROPPED_ROI']
        )