import pandas as pd


def clean_dataframe(df):
    """
    Takes raw DDR dataframe and returns cleaned version
    """

    original_columns = set(df.columns)

    drop_cols = [
        # redundant or useless
        "Chart Start", "Chart End",
        "Display BPM (min)", "Display BPM (max)",

        # difficulty; remove to avoid biasing the model
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
        "Mode Notes/Second",
        "Mode Time Between Notes"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # convert percentages so that they can be parsed as floats
    for col in df.columns:
        if df[col].astype(str).str.contains("%").any() and col != "Title" and col != "Artist":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("", "0")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    removed = original_columns - set(df.columns)
    print(f"Original: {len(original_columns)} columnns. {len(removed)} removed. {len(df.columns)} left.")

    return df