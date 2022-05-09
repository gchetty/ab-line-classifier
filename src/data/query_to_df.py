import yaml
import os
import pandas as pd

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

COLUMNS_WANTED = ['patient_id', 'a_or_b_lines']

database_query = cfg['PATHS']['DATABASE_QUERY']

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

#print(create_ABline_dataframe("parenchymal_clips.csv"))

if __name__ == "__main__":
    create_ABline_dataframe(database_query)