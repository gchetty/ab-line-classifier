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

    df = df[df.a_or_b_lines != 'non_a,_non_b']

    # Create filename
    df['filename'] = df['exam_id'] + "_" + df['patient_id'] + "_" + df["VID_id"]

    # Create column of class category to each clip. 
    # Modifiable for binary or multi-class labelling
    df['class'] = df.apply(lambda row: 0 if row.a_or_b_lines == 'a_lines' else
                           (1 if row.a_or_b_lines == 'b_lines_<_3' else
                            (1 if row.a_or_b_lines == 'b_lines-_moderate_(<50%_pleural_line)' else
                             (1 if row.a_or_b_lines == 'b_lines-_severe_(>50%_pleural_line)' else
                               2 if row.a_or_b_lines == 'non_a,_non_b' else
                                -1))), axis=1)

    df['a_or_b_lines'] = df['a_or_b_lines'].replace({'b_lines_<_3': 'b_lines', 'b_lines-_moderate_(<50%_pleural_line)': 'b_lines', 'b_lines-_severe_(>50%_pleural_line)': 'b_lines'})

    df['Path'] = df.apply(lambda row: '/home/derekwu/git repos/ab-line-classifer/data/masked_clips/' + row.filename, axis=1)
    
    # Finalize dataframe
    df = df[['filename'] + COLUMNS_WANTED + ['class'] + ['Path']]

    # Save df - append this csv to the previous csv 'clips_by_patient_2.csv'
    df.to_csv(cfg['PATHS']['CLIPS_TABLE'], index=False)

    return df

#print(create_ABline_dataframe("parenchymal_clips.csv"))

if __name__ == "__main__":
    create_ABline_dataframe(database_query)