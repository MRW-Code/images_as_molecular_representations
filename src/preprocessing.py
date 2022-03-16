import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re

def split_cc_dataset(df, val_pct):
    # get list of unique pairs
    df_unique = df.iloc[::2]

    # perform test/val split on unique pairs
    _, valid_df = train_test_split(df_unique, test_size=val_pct)

    # find index of both images in the valid pairs
    valid_idx = []
    for value in valid_df.index.values:
        if value % 2 == 0:
            valid_idx.append(value)
            valid_idx.append(value + 1)
        else:
            valid_idx.append(value)
            valid_idx.append(value - 1)

    # assign correct 'is_valid' label
    df['is_valid'] = None
    for row, _ in enumerate(df.index):
        if row in valid_idx:
            df.loc[row, 'is_valid'] = 1
        else:
            df.loc[row, 'is_valid'] = 0
    return df

def get_sol_labels(f):
    database = pd.read_csv('./data/solubility/raw_water_sol_set.csv')
    idx = re.findall(r'.*\/(.*).png', f)[0]
    label = database['logS'][database['CompoundID'] == idx].values[0]
    return label

def get_sol_df():
    image_dir = './data/solubility/images'
    paths = [f'{image_dir}/{x}' for x in os.listdir(image_dir)]
    labels = [get_sol_labels(f) for f in paths]
    model_df = pd.DataFrame({'fname' : paths,
                             'label' : labels})
    return model_df

def get_split_sol_dataset(val_pct, test_pct):
    df = get_sol_df()
    train, val = train_test_split(df, test_size=val_pct)
    val, test = train_test_split(df, test_size=test_pct)
    train.loc[:, 'is_valid'] = 0
    val.loc[:, 'is_valid'] = 1
    test.loc[:, 'is_valid'] = 'test'
    model_df = pd.concat([train, test, val], axis=0)
    return model_df
