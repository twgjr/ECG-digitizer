import os
import json
import numpy as np
import wfdb
import torch
import pandas as pd
from PIL import Image



def ecg_rel_path_base(ecg_id):
    dir_num = int(ecg_id / 1000)
    base_path = os.path.join("..", 'data', 'ptb_xl', 'records500', f'{dir_num:02}000')
    return base_path


def gen_splits(img_set_df, train_folds=range(1,9), val_folds=[9], test_folds=[10]):
    train = img_set_df[img_set_df['strat_fold'].isin(train_folds)]['ecg_id']
    val = img_set_df[img_set_df['strat_fold'].isin(val_folds)]['ecg_id']
    test = img_set_df[img_set_df['strat_fold'].isin(test_folds)]['ecg_id']

    return train, val, test


def get_X(ecg_id_df):
    # get the dataframe containing all the images as grayscale numpy arrays
    X_img = ecg_id_df.apply(lambda ecg_id: img_to_np(get_image(ecg_id))) 

    assert ecg_id_df.index.equals(X_img.index), "Indices are not aligned!"
    
    # join the dataframes
    X = pd.DataFrame({'ecg_id':ecg_id_df, 'img':X_img}).reset_index(drop=True)

    # filter out any images that are not the expected 425x550
    X = X[X['img'].apply(
        lambda img: isinstance(img, np.ndarray) and img.shape == (1,425,550)
    )].reset_index(drop=True)
    return X


def img_to_np(gs_img: Image):
    np_array = np.array(gs_img)  # (H, W)
    np_array = np.expand_dims(np_array, axis=0)  # Add channel dim -> (1, H, W)
    return np_array


def get_image(ecg_id, is_gray=True):
    print(f"ecg_id: {ecg_id}", end="\r")

    img_path = os.path.join(ecg_rel_path_base(ecg_id), f'{ecg_id:05}_hr-0.png')

    if(is_gray):
        return Image.open(img_path).convert('L')
    else:
        return Image.open(img_path).convert('RGB')


def get_outcomes(X_split, target_lead_name=None):
    unique_ecgs = X_split['ecg_id'].unique()
    
    records = []
    
    for ecg_id in unique_ecgs:
        print(f"Progress ecg_id: {ecg_id}", end="\r")
        record_df = get_y_single(ecg_id).to_dataframe()
        
        if target_lead_name is None:
            # Process all leads as before
            columns = []
            for lead_name in record_df.columns:
                signal_np = record_df[lead_name].values
                columns.append(signal_np)
            records.append(np.hstack(columns))
        else:
            # Process only the specified lead
            if target_lead_name in record_df.columns:
                signal_np = record_df[target_lead_name].values
                records.append(signal_np)
            else:
                print(f"Warning: Lead '{target_lead_name}' not found for ecg_id: {ecg_id}")
    
    return np.vstack(records)


def get_y_single(ecg_id: int) -> pd.DataFrame:
    abs_path = os.path.join(ecg_rel_path_base(ecg_id), f"{ecg_id:05}_hr")
    # relative_path = os.path.join(ecg_rel_path_base(ecg_id), f"{ecg_id:05}_hr")
    # abs_path = os.path.abspath(relative_path) 
    record = wfdb.rdrecord(abs_path)
    return record


class ECGDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y):
        if not (torch.is_tensor(x)):
            raise TypeError(f"Expected self.x to be a tensor, but got {type(x)}")
        if not (torch.is_tensor(y)):
            raise TypeError(f"Expected self.t to be a tensor, but got {type(y)}")
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        idx = idx % len(self.x)  # Ensure circular indexing
        sample, target = self.x[idx], self.y[idx]

        return sample, target