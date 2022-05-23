import os
import tarfile
import shutil
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
import LSTM_Classifier.utils as lstm_utils
import RF_Classifier.utils as rf_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(tar_file='data.tar'):
    """ Extract 'data.tar' file.
    :param string tar_file: path of '.tar' file
    :return: None
    """
    tar = tarfile.open(tar_file)
    tar.extractall('./')
    tar.close()
    os.rename(src='data', dst='raw')
    os.mkdir('Data')
    shutil.move('raw', 'Data/raw')


def slice_data(src_dir, dst_dir):
    """ Slice each train PSV file according to the HW instructions.
    Drop 'SepsisLabel' column.
    Impute missing values.
    Save the labels into CSV file named labels.csv
    :param string src_dir: path of source directory
    :param string dst_dir: path of destination directory
    :return: DataFrame
    """
    labels = {}
    for file in tqdm(os.listdir(src_dir), desc='Progress'):
        df = pd.read_csv(src_dir+'/'+file, delimiter='|')
        patient_id = int(file[8:-4])
        try:
            positive = df[df.SepsisLabel == 1].index[0]
            df = df.loc[0:positive].drop('SepsisLabel', axis=1)
            labels[patient_id] = 1
        except IndexError:
            df = df.drop('SepsisLabel', axis=1)
            labels[patient_id] = 0
        df.to_csv(dst_dir+'/'+file, sep='|', index=False)
    labels = pd.DataFrame.from_dict(labels, orient='index').sort_index().rename(columns={0: 'label'})
    labels.to_csv(dst_dir+'/labels.csv')
    return labels


def concat_data(src_dir, dst_dir, name=None, impute=None):
    """
    Concatenate all the sliced PSV files into a single file (and DataFrame) and Add 'PatientID' column.
    :param string src_dir: path of source directory (with PSV files)
    :param string dst_dir: path of destination directory
    :param string name: 'train' or 'test' or other
    :param impute: None or 'lstm' or 'rf'
    :return: DataFrame of concatenated files
    """
    # Receives ID of a patient from t (train or test) and returns its path:
    df_list = []
    for file in tqdm(os.listdir(src_dir), mininterval=1):
        df = pd.read_csv(src_dir+'/'+file, sep='|')
        if isinstance(impute, str):
            if impute == 'lstm':
                df = lstm_utils.impute_missing(df, attributes=df.columns)
            if impute == 'rf':
                df = rf_utils.impute_missing(df, attributes=df.columns)
        patient_id = int(file[8:-4])
        df['PatientID'] = np.full(len(df.index), patient_id)
        df_list.append(df)
    df = pd.concat(df_list)
    if isinstance(name, str):
        df.to_csv(dst_dir+'/'+name+'_concatenated.csv')
    return df


def scale_train_data(df, method, path):
    """ MinMaxScaler or StandardScaler()
    :param DataFrame df: Unscaled train DataFrame
    :param string method: 'minmax' or 'standard'
    :param string path: to save scaler.gz
    :return: DataFrame
    """
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, path+'/scaler.gz')  # joblib.load('scaler.gz')
    scaled = scaler.transform(df)
    result = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return result
