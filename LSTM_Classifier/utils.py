import numpy as np
import pandas as pd
from tqdm import tqdm


def impute_missing(df, attributes=None):
    """ Linear interpolation
    :param  df
    :param attributes: columns to edit
    :return: DataFrame
    """
    attributes = attributes
    if attributes is None:
        attributes = df.columns
    df = df[attributes].interpolate(limit_direction='both', method='linear')
    return df


def transform_trim(df, cols, r=20, save=False, mean_path='train'):
    """ Trim rows (tail of size r).
    :param DataFrame df: after imputing missing data
    :param list cols: list of data features
    :param int r: length of requested tail
    :param bool save: save result as a file named 'train_transformed.csv'
    :param string mean_path: means of features, used to fill NaN values.
    if 'train', calculates means here (from the given df). O.w., loads means DataFrame (from the CSV file in path).
    :return: DataFrame of concatenated set
    """
    df_list = []
    for patient_id, group in tqdm(df.groupby('PatientID')):
        tail = group[cols].tail(r).reset_index(drop=True)
        res = tail.copy()
        if len(tail.index) < r:
            m = r - len(tail.index)  # number of missing rows
            nan_rows = pd.DataFrame(np.nan, index=range(m), columns=cols)
            res = pd.concat([nan_rows, tail]).reset_index(drop=True)
            res = impute_missing(res, cols)
        res['PatientID'] = np.full(r, patient_id, dtype=int)
        df_list.append(res)
    res = pd.concat(df_list)
    if mean_path == 'train':
        res.fillna(res.mean(), inplace=True)
    else:
        means = pd.read_csv(mean_path, index_col=0)
        res.fillna(means, inplace=True)
    if save:
        res.to_csv('LSTM_Data/train_transformed.csv')
    return res


def get_train_stats(df, save=False):
    """
    Return some useful stats (i.e. mean & std).
    :param df: transformed training DataFrame
    :param save: bool save: save result files as named below.
    :return: DataFrame
    """
    stats = df.describe()
    if save:
        stats.loc['mean'].to_csv('train_mean_imputed_BF.csv')  # pd.read_csv('train_mean_imputed_BF.csv', index_col=0)
        stats.loc['std'].to_csv('train_std_imputed_BF.csv')  # pd.read_csv('train_std_imputed_BF.csv', index_col=0)
    return stats


def zip_sequences(df, labels, cols):
    """ Zip every data sample to tuple: (features_array, label, patient_id).
    :param df: ready to use DataFrame
    :param labels: DateFrame of labels, where index = patient_id
    :param cols: list of data features
    :return: list of tuples
    """
    sequences = []
    for patient_id, group in df.groupby('PatientID'):
        sequence_features = group[cols]
        label = labels.loc[patient_id]
        sequences.append((sequence_features, label, patient_id))
    return sequences







