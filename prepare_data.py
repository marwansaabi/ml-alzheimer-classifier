"""Curate the raw ChEMBL export into one labelled row per molecule.

Raw data: all IC50 records for human acetylcholinesterase (CHEMBL220).
"""
import numpy as np
import pandas as pd

from common import ACTIVE_NM, CLEAN_DATA, INACTIVE_NM, RAW_DATA, scaffold, standardize


def label_record(value, relation):
    """1 = active, 0 = inactive, NaN = grey zone or uninformative censored value."""
    if relation in ("=", "<") and value <= ACTIVE_NM:
        return 1
    if relation in ("=", ">", ">>") and value >= INACTIVE_NM:
        return 0
    return np.nan


def main():
    df = pd.read_csv(RAW_DATA, sep=";")
    steps = [("raw IC50 records", len(df))]

    df = df[(df["Standard Units"] == "nM") & df["Standard Value"].gt(0) & df["Smiles"].notna()]
    df = df[df["Assay Type"] == "B"]                       # binding assays only
    steps.append(("binding assays, valid values", len(df)))

    df = df[df["Data Validity Comment"].isna() & df["Potential Duplicate"].eq(0)]
    steps.append(("not flagged by ChEMBL", len(df)))

    df = df.assign(relation=df["Standard Relation"].str.strip("'"))
    df = df[df["relation"].isin(["=", "<", ">", ">>"])]
    df["label"] = [label_record(v, r) for v, r in zip(df["Standard Value"], df["relation"])]

    df["smiles"] = df["Smiles"].map(standardize)
    df = df[df["smiles"].notna()]

    # One row per parent structure: salts and repeated measurements collapse together.
    # Exact values are summarised by their median (robust to the odd outlier in heavily
    # tested compounds); censored values are only used when no exact value exists.
    rows, n_conflict, n_grey = [], 0, 0
    for smi, g in df.groupby("smiles"):
        exact = g.loc[g["relation"] == "=", "Standard Value"]
        if len(exact):
            median = exact.median()
            label = label_record(median, "=")
        else:
            median = np.nan
            labels = set(g["label"].dropna())
            if len(labels) > 1:
                n_conflict += 1
                continue
            label = labels.pop() if labels else np.nan
        if np.isnan(label):
            n_grey += 1
            continue
        rows.append({"chembl_id": g["Molecule ChEMBL ID"].iloc[0], "smiles": smi,
                     "ic50_nm_median": median, "n_measurements": len(g),
                     "active": int(label)})

    clean = pd.DataFrame(rows)
    clean["scaffold"] = clean["smiles"].map(scaffold)
    clean.to_csv(CLEAN_DATA, index=False)

    for name, n in steps:
        print(f"{name:<32}{n:>6}")
    print(f"{'unique parent structures':<32}{df['smiles'].nunique():>6}")
    print(f"{'  dropped: conflicting labels':<32}{n_conflict:>6}")
    print(f"{'  dropped: grey zone (1-10 µM)':<32}{n_grey:>6}")
    print(f"{'final molecules':<32}{len(clean):>6}  "
          f"(active {clean['active'].sum()}, inactive {(clean['active'] == 0).sum()}, "
          f"{clean['scaffold'].nunique()} scaffolds)")


if __name__ == "__main__":
    main()
