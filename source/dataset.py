import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from data_utils import clean_dataframe


class DDRDataset(Dataset):
    def __init__(
        self,
        csv_file,
        scaler=None
    ):
        df = pd.read_csv(csv_file, low_memory=False)

        drop_cols = [
            "Folder",
            "Title",
            "Artist",
            "Difficulty",
        ]

        cleaned_df = df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            errors="ignore"
        )

        self.y = cleaned_df["Rating"].values.astype("float32")

        X = cleaned_df.drop(columns=["Rating"], errors="ignore")

        if scaler:
            self.scaler = scaler
            self.X = torch.tensor(
                scaler.transform(X),
                dtype=torch.float32
            )

        else:
            self.scaler = StandardScaler()
            self.X = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float32)
        
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

        print(f"Final feature count: {self.X.shape[1]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]