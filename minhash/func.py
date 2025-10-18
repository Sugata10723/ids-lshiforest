import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_auc_score


def auc_by_attack(
    y_cat: pd.Series, y_label: pd.Series, scores: npt.NDArray, name: str
) -> None:
    results_df = pd.DataFrame(
        {
            "attack_cat": y_cat,
            "label": y_label,
            "score": scores,
        }
    )

    normal_df = results_df[results_df["label"] == 0]
    attack_df = results_df[results_df["label"] == 1]

    auc_scores_by_cat = {}
    unique_attack_cats = sorted(attack_df["attack_cat"].unique())

    for cat in unique_attack_cats:
        current_attack_df = attack_df[attack_df["attack_cat"] == cat]
        eval_df = pd.concat([normal_df, current_attack_df], axis=0)
        auc = roc_auc_score(eval_df["label"], eval_df["score"])
        auc_scores_by_cat[cat] = auc

    plot_df = pd.DataFrame(
        list(auc_scores_by_cat.items()), columns=["Attack Category", "AUC Score"]
    )
    plot_df = plot_df.sort_values("AUC Score", ascending=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.barplot(
        data=plot_df, x="Attack Category", y="AUC Score", color="steelblue", ax=ax
    )
    ax.set_title(f"LSHiForest Performance by Attack Category on {name}", fontsize=20)
    ax.set_xlabel("Attack Category", fontsize=14)
    ax.set_ylabel("AUC Score", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("AUC Scores (Each Attack Category vs. Normal Traffic):")
    print(plot_df.reset_index(drop=True))
    plt.savefig(
        f"results/{datetime.today().strftime('%y%m%d')}{name}_auc_by_category.png"
    )

def plot_anomaly_score(y_cat, y_test, scores) ->None:
    
