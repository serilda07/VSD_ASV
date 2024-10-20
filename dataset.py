import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class ASVspoof2021(Dataset):
    def __init__(self, path_to_features, path_to_protocol, part='train'):
        self.path_to_features = path_to_features
        self.part = part
        
        # Load protocol
        protocol_file = os.path.join(path_to_protocol, f'ASVspoof2021.LA.cm.{self.part}.trl.txt')
        self.audio_info = self.load_protocol(protocol_file)

    def load_protocol(self, protocol_file):
        with open(protocol_file, 'r') as f:
            return [line.strip().split() for line in f.readlines()]

    def get_lfcc_feature(self, idx):
        # Load the LFCC feature corresponding to the given index
        filename = self.protocol_file[idx].split()[1]  # Assuming protocol line format contains filename
        lfcc_path = f"{self.path_to_features}/{filename}_lfcc.csv"  # Assuming .npy format for LFCC features
        lfcc_feat = np.load(lfcc_path)
        return lfcc_feat

    def get_mfcc_feature(self, idx):
        # Load the MFCC feature corresponding to the given index
        filename = self.protocol_file[idx].split()[1]  # Assuming protocol line format contains filename
        mfcc_path = f"{self.path_to_features}/{filename}_mfcc.csv"  # Assuming .npy format for MFCC features
        mfcc_feat = np.load(mfcc_path)
        return mfcc_feat

    def __len__(self):
        return len(self.protocol_data)

    def __getitem__(self, idx):
        # Get the filename and label from the protocol
        filename = self.audio_info[idx][1]
        label = int(self.audio_info[idx][2])  # Assuming label is in the third column

        # Load features from CSV files
        lfcc_feat = pd.read_csv(os.path.join(self.path_to_features, f'{filename}_LFCC.csv'), header=None).values
        mfcc_feat = pd.read_csv(os.path.join(self.path_to_features, f'{filename}_MFCC.csv'), header=None).values

        # Convert to tensors
        lfcc_feat = torch.tensor(lfcc_feat, dtype=torch.float32)
        mfcc_feat = torch.tensor(mfcc_feat, dtype=torch.float32)

        return lfcc_feat, mfcc_feat, filename, label


   
