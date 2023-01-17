import yaml
import os
import pandas as pd
from tqdm import tqdm
import wget
import urllib
import logging

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def data_pull():
    '''
    Pull raw clips from AWS database using a generated query in csv format
    '''
    output_folder = cfg['PATHS']['RAW_CLIPS']
    df = pd.read_csv(cfg['PATHS']['CLIPS_TABLE'])

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

    return

if __name__ == '__main__':
    data_pull()