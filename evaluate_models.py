"""Benchmark four classifiers under random and scaffold-based cross-validation."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import CLEAN_DATA, FIGURES, RESULTS, SEED, fingerprints

K_FEATURES = "all"   # the sweep below shows selection does not help under scaffold CV
K_SWEEP = [20, 50, 100, 300, 700, "all"]
METRICS = {"roc_auc": "ROC AUC", "balanced_accuracy": "Balanced accuracy",
           "f1": "F1 (actives)", "matthews_corrcoef": "MCC"}


def models(pos_weight):
    return {
        "KNN (Tanimoto)": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, scale_pos_weight=pos_weight,
                                 eval_metric="logloss", n_jobs=-1, random_state=SEED),
    }


def pipeline(model, k=K_FEATURES):
    # Feature selection sits inside the pipeline so it is refitted on each training fold.
    return Pipeline([("variance", VarianceThreshold()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("model", model)])


def splitters():
    return {"Random split": StratifiedKFold(5, shuffle=True, random_state=SEED),
            "Scaffold split": StratifiedGroupKFold(5, shuffle=True, random_state=SEED)}


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(CLEAN_DATA)
    X, y = fingerprints(df["smiles"]), df["active"].to_numpy()
    groups = df["scaffold"].fillna("acyclic").to_numpy()   # empty scaffold is read as NaN
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"{len(y)} molecules, {len(set(groups))} scaffolds, {y.mean():.0%} active")

    # 1. Random vs scaffold split for every model
    rows = []
    for split_name, cv in splitters().items():
        for name, model in models(pos_weight).items():
            s = cross_validate(pipeline(model), X, y, groups=groups, cv=cv,
                               scoring=list(METRICS), n_jobs=1)
            row = {"split": split_name, "model": name}
            for m, label in METRICS.items():
                row[label] = s[f"test_{m}"].mean()
                row[f"{label} sd"] = s[f"test_{m}"].std()
            rows.append(row)
            print(f"{split_name:<15}{name:<16} AUC {row['ROC AUC']:.3f} ± {row['ROC AUC sd']:.3f}")
    res = pd.DataFrame(rows)
    res.round(3).to_csv(RESULTS / "cv_results.csv", index=False)

    # 2. How many fingerprint bits does Random Forest need? (scaffold split)
    sweep = []
    rf = models(pos_weight)["Random Forest"]
    for k in K_SWEEP:
        s = cross_validate(pipeline(rf, k), X, y, groups=groups,
                           cv=splitters()["Scaffold split"], scoring="roc_auc")
        sweep.append({"k": str(k), "ROC AUC": s["test_score"].mean(), "sd": s["test_score"].std()})
        print(f"k={k:<5} AUC {sweep[-1]['ROC AUC']:.3f}")
    sweep = pd.DataFrame(sweep)
    sweep.round(3).to_csv(RESULTS / "feature_sweep.csv", index=False)

    # Figures
    pivot = res.pivot(index="model", columns="split", values="ROC AUC")
    ax = pivot.plot.barh(figsize=(7, 4), color=["#9aa5b1", "#2163ca"], width=0.75)
    ax.set_xlim(0.5, 1)
    ax.set_xlabel("ROC AUC (5-fold CV)")
    ax.set_ylabel("")
    ax.set_title("Random split overestimates performance on new chemotypes")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "random_vs_scaffold.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.errorbar(sweep["k"], sweep["ROC AUC"], yerr=sweep["sd"], marker="o", capsize=3, color="#2163ca")
    ax.set_xlabel("Selected fingerprint bits (SelectKBest)")
    ax.set_ylabel("ROC AUC (scaffold CV)")
    ax.set_title("Random Forest: feature selection")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_sweep.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    cv = splitters()["Scaffold split"]
    for name, model in models(pos_weight).items():
        method = "decision_function" if name.startswith("SVM") else "predict_proba"
        scores = cross_val_predict(pipeline(model), X, y, groups=groups, cv=cv, method=method)
        scores = scores[:, 1] if scores.ndim == 2 else scores
        RocCurveDisplay.from_predictions(y, scores, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_title("ROC curves, scaffold split (out-of-fold)")
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_scaffold.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
