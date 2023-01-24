import pandas as pd
import yaml
import os
import cv2
import glob
from tqdm import tqdm

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

lb_annot = cfg['PATHS']['RT_LABELBOX_ANNOTATIONS']
b_lines_3_class = cfg['DATA']['RT_B_LINES_3_CLASS']

def get_rt_masked_clip_paths():
    '''
    Combines paths to all real-time validation masked clips into a single dataframe
    '''

    # Point to acquired data
    rootdir = cfg['PATHS']['RT_ROOT_DIR']
    dated_dirs = next(os.walk(rootdir))[1]
    clips_dir = 'masked_recordings'
    data = []

    # Loop over all directories containing csvs
    for dated_dir in dated_dirs:
        for root, dir, files in os.walk(os.path.join(rootdir, dated_dir, clips_dir)):
            # Keep only predicted probs csvs
            clips = [file for file in files]
            for clip in clips:
                clip_id = clip.split('.')[0]
                path_name = rootdir + dated_dir + '/' + clips_dir + '/' + clip_id
                data.append([int(clip_id), path_name])

    path_df = pd.DataFrame(data, columns=['filename', 'Path'])

    return path_df


def create_rt_ABline_dataframe(lb_annot, b_lines_3_class, preprocessed=False):
    '''
    Extracts pertinent information from Labelbox expert annotations and builds a dataframe linking clips, class, and their path

    :param lb_annot: path to Labelbox annotations
    :param b_lines_3_class: class label of < 3 B line clips ('a_lines' or 'b_lines')
    :param preprocessed: whether the Labelbox annotations have already been pre-processed
                         If True, lb_annot is a path to a csv with two columns 'filename' and 'a_or_b_lines'.
                         The former contains integer video IDs, the latter contains annotation labels (non-binary).
                         If False, lb_annot is a path to the raw Labelbox output, stored as an Excel workbook
    '''

    if not preprocessed:
        df = pd.read_excel(lb_annot)
        df['filename'] = df.apply(lambda row: int(row['External ID'][:10]), axis=1)
        df = df[['filename', 'a_or_b_lines']]
    else:
        df = pd.read_csv(lb_annot)

    b_lines_3_dict = {'b_lines': 1, 'a_lines': 0}

    # Create column of class category to each clip.
    # Modifiable for binary or multi-class labelling
    df['class'] = df.apply(lambda row: 0 if row.a_or_b_lines == 'a_lines' else
                           (b_lines_3_dict[b_lines_3_class] if row.a_or_b_lines == 'b_lines_3' else
                            (1 if row.a_or_b_lines == 'b_lines_moderate_50_pleural_line' else
                             (1 if row.a_or_b_lines == 'b_lines_severe_50_pleural_line' else
                               0 if row.a_or_b_lines == 'non_a_non_b' else
                                -1))), axis=1)

    # Relabel all b-line severities as a single class for A- vs. B-line classifier
    df['a_or_b_lines'] = df['a_or_b_lines'].replace(
        {'b_lines_3': b_lines_3_class, 'b_lines_moderate_50_pleural_line': 'b_lines',
         'b_lines_severe_50_pleural_line': 'b_lines'})

    # Add path to masked clips
    path_df = get_rt_masked_clip_paths()
    df = df.merge(path_df, how='outer', on='filename')
    return df

def mp4_to_images(mp4_path):
    """
    Converts masked ultrasound mp4 video to a series of images and saves the images in the same directory.
    :param mp4_path: File name of the mp4 file to convert to series of images.
    """
    vc = cv2.VideoCapture(mp4_path)
    vid_dir, mp4_filename = os.path.split(mp4_path)  # Get folder and filename of mp4 file respectively
    mp4_filename = mp4_filename.split('.')[0]  # Strip file extension

    if not os.path.exists(cfg['PATHS']['FRAMES']):
        os.makedirs(cfg['PATHS']['FRAMES'])

    idx = 0
    image_paths = []

    if not os.path.isdir(cfg['PATHS']['FRAMES']):
        os.mkdir(cfg['PATHS']['FRAMES'])

    while True:
        ret, frame = vc.read()
        if not ret:
            break  # End of frames reached
        image_path = mp4_filename + '_' + str(idx) + '.jpg'
        image_paths.append(image_path)
        cv2.imwrite(cfg['PATHS']['FRAMES'] + image_path, frame)  # Save all the images out
        idx += 1
    return image_paths


def create_rt_image_dataset(query_df_path):
    """
    Create a dataset of frames for real_time data, including their class
    :param query_df_path: File name of the CSV file containing the database query results for clips
    """

    query_df = pd.read_csv(query_df_path)
    clip_dfs = []

    for index, row in tqdm(query_df.iterrows()):
        for mp4_file in glob.glob(row['Path'] + '/' + row['filename'] + '.mp4'):
            image_paths = mp4_to_images(mp4_file)  # Convert mp4 encounter file to image files
            # Real-time clips aren't associated with patient IDs
            clip_df = pd.DataFrame({'Frame Path': image_paths, 'Class': row['class'],
                                    'Class Name': cfg['DATA']['CLASSES'][row['class']]})

            clip_dfs.append(clip_df)
    all_clips_df = pd.concat(clip_dfs, axis=0, ignore_index=True)
    all_clips_df.to_csv(cfg['PATHS']['FRAME_TABLE'], index=False)
    return


if __name__ == '__main__':
    create_rt_ABline_dataframe(lb_annot, b_lines_3_class)
    create_rt_image_dataset(cfg['PATHS']['CLIPS_TABLE'])

