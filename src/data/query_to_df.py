import yaml
import os
import pandas as pd

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

COLUMNS_WANTED = ['patient_id', 'a_or_b_lines']

database_query = cfg['PATHS']['DATABASE_QUERY']
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


def create_rt_ABline_dataframe(lb_annot,b_lines_3_class, preprocessed=False):
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


def create_ABline_dataframe(database_query):
    '''
    Extracts out pertinent information from database query csv and builds a dataframe linking filenames, patients, and class
    :database_filename: filepath to database query csv
    '''
    df = pd.read_csv(database_query)

    # Remove all muggle clips
    df = df[df.frame_homogeneity.isnull()]

    # Remove Non-A/Non-B line clips
    df = df[df.a_or_b_lines != 'non_a_non_b']

    # Removes clips with unlabelled parenchymal findings
    df = df[df.a_or_b_lines.notnull()]

    # Create filename
    df['filename'] = df['exam_id'] + "_" + df['patient_id'] + "_VID" + df["vid_id"].map(str)

    # Create column of class category to each clip. 
    # Modifiable for binary or multi-class labelling
    df['class'] = df.apply(lambda row: 0 if row.a_or_b_lines == 'a_lines' else
                           (1 if row.a_or_b_lines == 'b_lines_<_3' else
                            (1 if row.a_or_b_lines == 'b_lines-_moderate_(<50%_pleural_line)' else
                             (1 if row.a_or_b_lines == 'b_lines-_severe_(>50%_pleural_line)' else
                               2 if row.a_or_b_lines == 'non_a_non_b' else
                                -1))), axis=1)

    # Relabel all b-line severities as a single class for A- vs. B-line classifier
    df['a_or_b_lines'] = df['a_or_b_lines'].replace({'b_lines_<_3': 'b_lines', 'b_lines-_moderate_(<50%_pleural_line)': 'b_lines', 'b_lines-_severe_(>50%_pleural_line)': 'b_lines'})

    df['Path'] = df.apply(lambda row: cfg['PATHS']['MASKED_CLIPS'] + row.filename, axis=1)

    df['s3_path'] = df.apply(lambda row: row.s3_path, axis=1)
    
    # Finalize dataframe
    df = df[['filename'] + COLUMNS_WANTED + ['class'] + ['Path'] + ['s3_path']]

    # Save df - append this csv to the previous csv 'clips_by_patient_2.csv'
    df.to_csv(cfg['PATHS']['CLIPS_TABLE'], index=False)

    return df


if __name__ == "__main__":
    create_rt_ABline_dataframe(lb_annot, b_lines_3_class)
    #create_ABline_dataframe(database_query)