"""Train the final model on all curated data and score a few reference compounds."""
import pandas as pd
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier

from common import CLEAN_DATA, RESULTS, SEED, bit_vectors, fingerprints, standardize

# Known AChE inhibitors, a negative control and compounds from the literature.
CANDIDATES = [
    ("Donepezil", "known AChE inhibitor", "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC"),
    ("Galantamine", "known AChE inhibitor", "CN1CC[C@@]23C=C[C@@H](C[C@@H]2OC4=C(C=CC(=C34)C1)OC)O"),
    ("Demecarium", "known AChE inhibitor",
     "[Br-].[Br-].CN(C)C(=O)Oc1cccc(c1)[N+](C)(C)CCCCCCCCCC[N+](C)(C)c2cccc(OC(=O)N(C)C)c2"),
    ("Aspirin", "negative control", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Vesatolimod", "literature (Vignaux et al.)", "CCCCOC1=NC(=C2C(=N1)N(CC(=O)N2)CC3=CC=CC(=C3)CN4CCCC4)N"),
    ("Serdemetan", "literature (Vignaux et al.)", "C1=CC=C2C(=C1)C(=CN2)CCNC3=CC=C(C=C3)NC4=CC=NC=C4"),
    ("Tacrine derivative", "literature (Vignaux et al.)", "CCN(CC)CCCNc1c2ccccc2nc3ccccc13"),
]


def main():
    train = pd.read_csv(CLEAN_DATA)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED)
    model.fit(fingerprints(train["smiles"]), train["active"])

    train_smiles = set(train["smiles"])
    train_fps = bit_vectors(train["smiles"])

    rows = []
    for name, group, raw in CANDIDATES:
        smi = standardize(raw)
        sims = DataStructs.BulkTanimotoSimilarity(bit_vectors([smi])[0], train_fps)
        nearest = max(range(len(sims)), key=sims.__getitem__)
        prob = model.predict_proba(fingerprints([smi]))[0, 1]
        rows.append({
            "compound": name, "group": group,
            "in_training_set": smi in train_smiles,
            "max_tanimoto_to_training": round(sims[nearest], 2),
            "nearest_neighbour_label": "active" if train["active"].iloc[nearest] else "inactive",
            "p_active": round(prob, 2),
            "prediction": "active" if prob >= 0.5 else "inactive",
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "screening.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
