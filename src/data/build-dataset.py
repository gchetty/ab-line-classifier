import pandas as pd
import yaml
import os
import cv2
import glob
from tqdm import tqdm

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def to_greyscale(img):
    '''
    Converts RBG image to greyscale.
    :param img: RBG image
    :return: Image to convert to greyscale
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def mp4_to_images(mp4_path):
    '''
    Converts masked ultrasound mp4 video to a series of images and saves the images in the same directory.
    :param mp4_path: File name of the mp4 file to convert to series of images.
    '''
    vc = cv2.VideoCapture(mp4_path)
    vid_dir, mp4_filename = os.path.split(mp4_path)      # Get folder and filename of mp4 file respectively
    mp4_filename = mp4_filename.split('.')[0]       # Strip file extension

    idx = 0
    max_area = 0
    max_area_id = 0
    image_paths = []

    if not os.path.isdir(cfg['PATHS']['FRAMES']):
        os.mkdir(cfg['PATHS']['FRAMES'])

    while (True):
        ret, frame = vc.read()
        if not ret:
            break   # End of frames reached
        image_path = mp4_filename + '_' + str(idx) + '.jpg'
        image_paths.append(image_path)
        cv2.imwrite(cfg['PATHS']['FRAMES'] + image_path, frame)  # Save all the images out
        idx += 1
    return image_paths

def create_image_dataset(query_df_path):
    '''
    Create a dataset of frames, including their patient ID and class
    :param query_df_path: File name of the CSV file containing the database query results for clips
    '''

    query_df = pd.read_csv(query_df_path)
    clip_dfs = []

    for index, row in tqdm(query_df.iterrows()):
        for mp4_file in glob.glob(row['Path'] + '.mp4'):
            image_paths = mp4_to_images(mp4_file)  # Convert mp4 encounter file to image files
            clip_df = pd.DataFrame({'Frame Path': image_paths, 'Patient': row['patient_id'], 'Class': row['class'],
                                    'Class Name': cfg['DATA']['CLASSES'][row['class']]})
            clip_dfs.append(clip_df)
    all_clips_df = pd.concat(clip_dfs, axis=0, ignore_index=True)
    all_clips_df.to_csv(cfg['PATHS']['FRAME_TABLE'], index=False)
    return


def create_rt_image_dataset(query_df_path):
    '''
    Create a dataset of frames, including their class, from the real-time validation clip dataframe
    :param query_df_path: File name of the CSV file containing the real-time validation clip info
    '''

    query_df = pd.read_csv(query_df_path)
    clip_dfs = []

    for index, row in tqdm(query_df.iterrows()):
        for mp4_file in glob.glob(row['Path']):
            image_paths = mp4_to_images(mp4_file)  # Convert mp4 encounter file to image files
            clip_df = pd.DataFrame({'Frame Path': image_paths, 'Class': row['class'],
                                    'Class Name': cfg['DATA']['CLASSES'][row['class']]})
            clip_dfs.append(clip_df)
    all_clips_df = pd.concat(clip_dfs, axis=0, ignore_index=True)
    all_clips_df.to_csv(cfg['PATHS']['FRAME_TABLE'], index=False)
    return


if __name__=='__main__':
    create_rt_image_dataset(cfg['PATHS']['CLIPS_TABLE'])






