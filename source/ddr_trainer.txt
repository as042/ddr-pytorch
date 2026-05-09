import pickle
import pprint

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json

def clean_dataframe(df):
    """
    Takes raw DDR dataframe and returns cleaned version
    """

    original_columns = set(df.columns)

    # =====================
    # DROP COLUMNS
    # =====================
    drop_cols = [
        # redundant or useless
        "Chart Start", "Chart End",
        "Display BPM (min)", "Display BPM (max)",
        "Median Notes/Second", "Mode Notes/Second",
        "Median Time Between Notes", "Mode Time Between Notes",

        # difficulty
        "Estimate of relative difficulty",
        "Alternate estimate of relative difficulty",
        "Real-world scoring difficulty (1.0-19.7) (Sanbai Ice Cream v10r1 2025-05-25)",
        "Difficulty within folder (.00-.95) (Sanbai Ice Cream v10r1 2025-05-25)",
        "PPR X Spice Rating (2025-07-24)",

        # groove radar
        "STREAM (official)", "VOLTAGE (official)", "AIR (official)",
        "FREEZE (official)", "CHAOS (official)",
        "GROOVE RADAR Total (official)",
        "STREAM (official)/Total%", "VOLTAGE (official)/Total%",
        "AIR (official)/Total%", "FREEZE (official)/Total%",
        "CHAOS (official)/Total%",
        "STREAM (estimated)", "VOLTAGE (estimated)", "AIR (estimated)",
        "FREEZE (estimated)", "CHAOS (estimated)",
        "GROOVE RADAR Total (estimated)",
        "STREAM (estimated)/Total%", "VOLTAGE (estimated)/Total%",
        "AIR (estimated)/Total%", "FREEZE (estimated)/Total%",
        "CHAOS (estimated)/Total%",
        "STREAM (official/estimated) diff",
        "VOLTAGE (official/estimated) diff",
        "AIR (official/estimated) diff",
        "FREEZE (official/estimated) diff",
        "CHAOS (official/estimated) diff",
        "GROOVE RADAR Total (official/estimated) diff",

        # scoring junk
        "Required Judgment",
        "Combo/Step",
        "Combo/Second",
        "Max Combo (Steps+Shock Arrows)",
        "Score per Marvelous/O.K.",
        "Score per Perfect",
        "Score per Great",
        "Score per Good",
        "EX Score/Step",
        "EX Score/Second",
        "Holds Ignored by Score and EX Score",
        "Holds Ignored by Score and EX Score/Holds%",
        "Holds Ignored by Score and EX Score/Second",
        "Max EX Score (see note)",
        "Score from Steps",
        "Score from Holds",
        "Score from Shock Arrows",
        "Score from O.K.s",
        "O.K.s (Lines With Holds+Shock Arrows)",
        "O.K.s/Step",
        "O.K.s/Second",
        "AA Min Mean Precision", "AA Min Required Judgment", "AA Max Misses",
        "AAA Min Mean Precision", "AAA Min Required Judgment", "AAA Max Misses",

        # non-numerical
        "Most Frequent Quantizations",
        "Finest Quantization",

        # has non-numerical samples
        "Left/Right Bias",
        "Down/Up Bias",
        "Closest Jumps",
        "Brackets/Jumps%",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # =====================
    # REMOVE REDUNDANT FEATURES
    # =====================
    redundant_cols = [
        "Notes/Second",
        "Steps/Second",
    ]

    df = df.drop(columns=[c for c in redundant_cols if c in df.columns], errors="ignore")

    # # =====================
    # # KEEP ONLY NUMERIC
    # # =====================
    # df = df.select_dtypes(include=["number"])

    # # =====================
    # # FILL MISSING
    # # =====================
    # df = df.fillna(0)

    # =====================
    # CONVERT PERCENT STRINGS TO FLOATS
    # =====================
    for col in df.columns:
        # Try converting percentage strings like "12.5%" → 0.125
        if df[col].astype(str).str.contains("%").any() and col != "Title" and col != "Artist":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("", "0")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    # =====================
    # DEBUG INFO
    # =====================
    removed = original_columns - set(df.columns)
    print(f"Removed {len(removed)} columns")

    return df


# =====================
# Dataset
# =====================
class DDRDataset(Dataset):
    def __init__(self, csv_file, save_clean_csv=False, clean_csv_path="cleaned_ddr_data.csv"):
        df = pd.read_csv(csv_file, low_memory=False)

        # Detect if already cleaned
        if "Title" not in df.columns and "Artist" not in df.columns:
            print("Detected pre-cleaned dataset. Skipping heavy cleaning.")
            cleaned_df = df.copy()
        else:
            cleaned_df = clean_dataframe(df)

        # Save cleaned version if requested
        if save_clean_csv:
            cleaned_df.to_csv(clean_csv_path, index=False)
            print(f"Saved cleaned dataset to {clean_csv_path}")

        # =====================
        # REMOVE IDENTIFIERS
        # =====================
        drop_cols = [
            "Folder", "Title", "Artist", "Difficulty",
        ]

        cleaned_df = cleaned_df.drop(columns=[c for c in drop_cols if c in cleaned_df.columns], errors="ignore")

        # =====================
        # TARGET
        # =====================
        self.y = cleaned_df["Rating"].values.astype("float32")

        # =====================
        # FEATURES
        # =====================
        X = cleaned_df.drop(columns=["Rating"], errors="ignore")

        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

        print(f"Final feature count: {self.X.shape[1]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================
# Model
# =====================
class DDRModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# =====================
# Training
# =====================
def train(model, dataset, epochs):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in loader:
            preds = model(X_batch)

            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


def save_model(model, scaler, path="ddr_model.pt", txt_format=False):
    if txt_format:
        state_dict = model.state_dict()

        readable_state = {}
        for k, v in state_dict.items():
            readable_state[k] = v.detach().cpu().tolist()

        debug_data = {
            "model_state_dict": readable_state,

            "scaler": {
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "var_": scaler.var_.tolist()
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2)

    else:
        package = {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
        }

        torch.save(package, path)

    print(f"Saved model to {path}")


def load_model(model_class, input_dim, path="ddr_model.pth"):
    package = torch.load(path, map_location="cpu", weights_only=False)

    model = model_class(input_dim)
    model.load_state_dict(package["model_state_dict"])
    scaler = package["scaler"]

    print(f"Loaded model from {path}")

    return model, scaler


def predict(model, scaler, new_data):
    x = torch.tensor([new_data], dtype=torch.float32)
    x = torch.tensor(scaler.transform(x), dtype=torch.float32)

    with torch.no_grad():
        pred = model(x).item()

    return pred


# =====================
# Run
# =====================
if __name__ == "__main__":
    dataset = DDRDataset(
        "ddr_clean.csv",
        save_clean_csv=True,
        clean_csv_path="ddr_clean.csv"
    )
    # model = DDRModel(input_dim=dataset.X.shape[1])
    model, _ = load_model(DDRModel, dataset.X.shape[1], "ddr_model.pt")
    model = train(model, dataset, 80)

    save_model(model, dataset.scaler, path="ddr_model_e100.pt", txt_format=False)
    # save_model(model, dataset.scaler, path="test.txt", txt_format=True)

    # Example prediction
    dentity = [127,100.0,200.0,2,0,573,542,31,52,0,0,12,1,11,28,12,16,0,28,0,1,0,7.5,0.833,4.677,1.516,0.0,119.25,0.0,0.0,0.0,0.0,0.1,0.2,0.4,0.6,1.7,1.2,0.218,0.138,25.5,0.49,0.3,2.25,100,100.0,0.0,0.0,0.0,2,0.0572,0.0959,0.0,0.0,0.022099999999999998,0.0018,0.0203,0.051699999999999996,0.022099999999999998,0.029500000000000002,0.0,0.051699999999999996,0.0,0.0018,0.0,0.0157480315,0.0,0.0157480315,0.2599580713,0.4360587002,0.0,0.0,0.1006289308,0.008385744235,0.09224318658,0.2348008386,0.1006289308,0.1341719078,0.0,0.2348008386,0.0,0.008385744235,0.0,28,0.051699999999999996,0.2348008386,40,0.0738,0.3354297694,0.2269372694,1.031446541,5.576519916,49,0.0904,0.4109014675,140,148,146,139,0.01570680628,0.02617801047,0.5130890052,365,134,73,1,0,0,0,0,0,0,0,5.97556719,0.127399651,0.2019197208]
    comebacktome = [118,172.0,172.0,0,0,507,482,25,8,0,0,48,23,25,3,0,3,1,4,6,0,5,10.033,0.0,4.606,1.927,0.0,103.256,0.0,0.0,0.0,0.0,0.087,0.174,0.349,0.523,1.483,2.878,0.215,0.181,7.936,0.992,0.349,2.616,0,0.0,0.0,0.0,0.0,0,0.0519,0.0166,0.0,0.0,0.09960000000000001,0.04769999999999999,0.0519,0.0062,0.0,0.0062,0.0021,0.0083,0.0124,0.0,0.0104,0.0,0.0,0.0,0.2421166809,0.07747733788,0.0,0.0,0.4648640273,0.2227473464,0.2421166809,0.0290540017,0.0,0.0290540017,0.009684667235,0.03873866894,0.05810800341,0.0,0.04842333617,10,0.0207,0.09684667235,63,0.1307,0.6101340358,0.1991701245,0.9297280545,5.597737662,8,0.0166,0.07747733788,114,144,135,114,0.05917159763,0.100591716,0.550295858,288,159,0,60,0,0,0,0,0,0,0,6.674556213,0.0,0.2]
    print("Identity rating:", predict(model, dataset.scaler, dentity))
    print("Come Back to Me rating:", predict(model, dataset.scaler, comebacktome))