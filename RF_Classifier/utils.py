import pandas as pd
import numpy as np
from tqdm import tqdm


def impute_missing(df, attributes=None):
    """ Pad with '0' if column is empty.
    Fill with a value if there is a single non-empty cell.
    Interpolate in other cases.
    :param DataFrame df:
    :param attributes: columns to edit
    :return: DataFrame
    """
    attributes = attributes
    if attributes is None:
        attributes = df.columns
    df_clean = df.copy()
    for att in attributes:
        if df_clean[att].isnull().sum() == len(df_clean):
            df_clean[att] = df_clean[att].fillna(0)
        elif df_clean[att].isnull().sum() == len(df_clean) - 1:
            df_clean[att] = df_clean[att].ffill().bfill()
        else:
            df_clean[att] = df_clean[att].interpolate(method='nearest', limit_direction='both')
            df_clean[att] = df_clean[att].ffill().bfill()
    return df_clean


def transform_stats(df, cols, save=False):
    """ Return a row of stats for each patient in df
    :param DataFrame df: after imputing missing data
    :param list cols: parameters of interest
    :param bool save: save result as a file named 'train_transformed.csv'
    :return: DataFrame
    """
    rows = []
    for patient_id, group in tqdm(df.groupby('PatientID')):
        stats = group.describe()
        row = []
        for col in cols:
            row += list(stats[col].rename(index=lambda s: col+'_'+s))
        row.append(patient_id)
        rows.append(row)
    # Create the header row:
    header = []
    suffixes = ['_count', '_mean', '_std', '_min', '_25%', '_50%', '_75%', '_max']
    for col in cols:
        header += [col+s for s in suffixes]
    header.append('PatientID')
    # Result DataFrame:
    res = pd.DataFrame(rows).fillna(0)
    res.columns = header
    if save:
        res.to_csv('RF_Data/train_transformed.csv')
    return res


