"""
Loads the ECG dataset of signals into memory according to the index "train.csv".
Originally data was stored in "train" folder. "train" is a misnomer, since
it was practically the entire dataset available for training.

First the dataset is loaded as a numpy array shaped: (num_samples, num_leads, num_timesteps)

num_samples is 977, num_leads is 12, and num_timesteps is 2500.

This is sparse since 11 of 12 leads have 3/4 of the timesteps as NaN.

The 12th lead has data for the full 2500 timesteps.

Will not be imputing the NaN values. NaN values will be dropped and lead II (the
12th lead) will be split into 4 equal segments of 625 timesteps each.
This will align all the leads to have 625 timesteps.
"""

import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ECGDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.data = None

    def load_data(self):
        # Load the CSV file
        print(f"Loading CSV file from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        # Initialize an array to hold the ECG data
        num_leads = 12
        num_samples = len(df)
        num_segments = 15
        num_timesteps = 2500
        num_cols = 4
        num_segment_timesteps = num_timesteps // num_cols  # 625 timesteps per segment
        print(f"Initializing data array with shape: ({num_samples * num_segments}, {num_segment_timesteps})")
        self.data = np.full((num_samples * num_segments, num_segment_timesteps), np.nan)
        print(f"Initialized data array with shape: {self.data.shape} with # NaN values: {np.isnan(self.data).sum()}")

        # Load each sample's ECG data into the array
        for i, row in tqdm(df.reset_index(drop=True).iterrows(), total=len(df), desc="Loading ECG data"):
            data_index = i * num_segments
            sample_id = row["id"]
            file_path = os.path.join(self.data_dir, str(sample_id), f"{sample_id}_250Hz.csv")
            lead_df = pd.read_csv(
                file_path
            )  # I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6 cols x 2500 rows

            lead_II_df = lead_df["II"]
            other_leads_df = lead_df.drop(columns=["II"])
            # delete lead_df to free up memory since we have lead_II_df and other_leads_df
            del lead_df

            # print(f"Processing sample {i}/{num_samples}, file: {file_path}, lead_II_df shape: {lead_II_df.shape}")

            other_leads = np.empty((num_leads-1, num_segment_timesteps)) # shape (11, 625)

            for lead_idx, lead_name in enumerate(other_leads_df.columns):
                # print(f"Processing lead: {lead_name}")
                lead_data = other_leads_df[lead_name].dropna().values
                # print(f"Lead {lead_name} data length after dropping NaNs: {len(lead_data)}")
                # print(f"other_leads shape for {lead_name}: {other_leads.shape}")
                other_leads[lead_idx] = lead_data

            # add other leads to to the full data array
            other_slice_start = data_index
            other_slice_end = data_index + num_segments - num_cols
            # print(f"Placing other leads {other_leads.shape} data into self.data[{other_slice_start}:{other_slice_end},:]")
            self.data[other_slice_start:other_slice_end,:] = other_leads
            
            # add 4 lead II segments to 12, 13, 14, 15 positions
            II_slice_start = data_index + num_segments - num_cols
            II_slice_end = data_index + num_segments
            # print(f"Placing lead II data into self.data[{II_slice_start}:{II_slice_end},:]")
            self.data[II_slice_start:II_slice_end,:] = np.array_split(
                lead_II_df.values, num_cols
            )

            # print(f"Finished processing sample {i}/{num_samples}")
            # if i >= 1:
                # break

        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_splits(csv_path, data_dir, val_size=0.1, test_size=0.1, random_state=42):
    dataset = ECGDataset(csv_path, data_dir)
    dataset.load_data()

    idx = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(idx, test_size=val_size + test_size, random_state=random_state)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (val_size + test_size), random_state=random_state)

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


if __name__ == "__main__":
    dataset = ECGDataset(csv_path="data/train.csv", data_dir="data/train")
    data = dataset.load_data()
    print(data.shape)  # Should print (977*15, 625)
    # print the number of NaN values in the dataset
    print(f"Number of NaN values in the dataset: {np.isnan(data).sum()}")