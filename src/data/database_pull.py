import yaml
import os
import pandas as pd
from tqdm import tqdm
import wget

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def data_pull():
    '''
    Pull raw clips from AWS database using a generated query in csv format
    '''
    output_folder = cfg['PATHS']['RAW_CLIPS']
    df = pd.read_csv(cfg['PATHS']['DATABASE_QUERY'])

    print('Getting AWS links...')

    # Dataframe of all clip links
    links = df.s3_path

    print('Fetching clips from AWS...')

    # Download clips and save to disk
    for link in tqdm(links):
        print(link)
        firstpos = link.rfind("/")
        lastpos = len(link)
        filename = link[firstpos+1:lastpos]

        wget.download(link, output_folder + filename)

    print('Fetched clips successfully!')

    return

if __name__ == '__main__':
    data_pull()