"""
Beating the Bracket: baseline checkpoint.

Question: on tournaments a model never saw, does it pick NCAA tournament
winners better than "the better seed wins"?

Baseline (this file): logistic regression on seed difference only, with an L2
(ridge) penalty tuned by leave-one-season-out cross-validation. Every score is
from nested CV, so it comes from a season the tuned model never saw.
Reference: the seed rule (better seed wins; equal seeds count as half right).
The last five tournaments are held out as the final test set and NOT used here.

Usage:  put the three Kaggle CSVs in data/, then run  python baseline.py
Output: results/results.json and results/baseline.png
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUT_DIR = "results"
SEED = 42                               # team-order randomization + bootstrap
N_TEST = 5                              # last five tournaments held out
CS = [1e-3, 1e-2, 1e-1, 1, 10, 100]     # C = 1 / ridge penalty strength
N_BOOT = 2000
X = ["SeedDiff"]                        # baseline uses seed difference only

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- load
seeds = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
tour = pd.read_csv(f"{DATA_DIR}/MNCAATourneyCompactResults.csv")
reg = pd.read_csv(f"{DATA_DIR}/MRegularSeasonCompactResults.csv")

# Seed strings look like "W01" or "X16a": region letter, number, play-in suffix.
seeds["SeedNum"] = seeds.Seed.str[1:3].astype(int)
seeds["PlayIn"] = seeds.Seed.str.len() > 3
seeds["SeedBase"] = seeds.Seed.str[:3]

# ------------------------------------- regular-season team features
# Only that season's regular-season games, so nothing from the tournament leaks in.
w = reg[["Season", "WTeamID", "WScore", "LScore"]].set_axis(["Season", "TeamID", "PF", "PA"], axis=1)
l = reg[["Season", "LTeamID", "LScore", "WScore"]].set_axis(["Season", "TeamID", "PF", "PA"], axis=1)
w["Win"], l["Win"] = 1, 0
g = pd.concat([w, l])
g["Margin"] = g.PF - g.PA
team = (g.groupby(["Season", "TeamID"])
         .agg(WinPct=("Win", "mean"), Margin=("Margin", "mean"))
         .reset_index())

# ------------------------------------- one row per tournament game
df = tour[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
n_raw = len(df)
for side in ["W", "L"]:
    s = seeds.rename(columns={c: side + c for c in ["TeamID", "SeedNum", "PlayIn", "SeedBase"]})
    df = df.merge(s[["Season", side + "TeamID", side + "SeedNum", side + "PlayIn", side + "SeedBase"]],
                  on=["Season", side + "TeamID"], how="left")
    t = team.rename(columns={c: side + c for c in ["TeamID", "WinPct", "Margin"]})
    df = df.merge(t, on=["Season", side + "TeamID"], how="left")

missing = df[["WSeedNum", "LSeedNum", "WWinPct", "LWinPct"]].isna().any(axis=1)
first_four = df.WPlayIn & df.LPlayIn & (df.WSeedBase == df.LSeedBase)   # play-in games
n_missing, n_ff = int(missing.sum()), int((first_four & ~missing).sum())
df = df[~missing & ~first_four].reset_index(drop=True)

# Kaggle lists the winner first. Randomize which team is "A" once, so the
# label is not always 1. flip=True means team A is the loser.
rng = np.random.default_rng(SEED)
flip = rng.random(len(df)) < 0.5
sign = np.where(flip, -1, 1)
df["SeedDiff"] = sign * (df.WSeedNum - df.LSeedNum)    # A minus B; negative = A is the better seed
df["WinPctDiff"] = sign * (df.WWinPct - df.LWinPct)
df["MarginDiff"] = sign * (df.WMargin - df.LMargin)
df["y"] = (~flip).astype(int)                         # 1 if team A won

# ------------------------------------- time-ordered split
seasons = np.sort(df.Season.unique())
test_seasons = seasons[-N_TEST:]
train = df[~df.Season.isin(test_seasons)].reset_index(drop=True)
train_seasons = np.sort(train.Season.unique())

# ------------------------------------- seed rule
def seed_rule_prob(d):
    return np.where(d.SeedDiff < 0, 1.0, np.where(d.SeedDiff > 0, 0.0, 0.5))

def acc(y, p):  # equal seeds (p = 0.5) count as half correct
    return np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float))

# ------------------------------------- logistic regression + CV
def fit(d, C):
    # The scaler is part of the pipeline, so it is fit on training folds only.
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C))
    m.fit(d[X], d.y)
    return m

def choose_C(d):
    """Pick C by leave-one-season-out log loss on the seasons in d."""
    totals = {}
    for C in CS:
        tot = 0.0
        for s in d.Season.unique():
            tr, va = d[d.Season != s], d[d.Season == s]
            p = fit(tr, C).predict_proba(va[X])[:, 1]
            tot += log_loss(va.y, p, labels=[0, 1]) * len(va)
        totals[C] = tot / len(d)
    return min(totals, key=totals.get), totals

# Nested leave-one-season-out: hold out one season, tune C on the rest, score the held-out season.
oof = np.zeros(len(train))
chosen = {}
for s in train_seasons:
    tr_idx, va_idx = train.Season != s, train.Season == s
    C, _ = choose_C(train[tr_idx])
    chosen[int(s)] = C
    oof[va_idx.values] = fit(train[tr_idx], C).predict_proba(train.loc[va_idx, X])[:, 1]

C_final, cv_curve = choose_C(train)      # the setting CV picks using all training seasons

y = train.y.values
p_seed = seed_rule_prob(train)
a_model, a_seed = acc(y, oof), acc(y, p_seed)
ll_game = -(y * np.log(oof) + (1 - y) * np.log(1 - oof))

# Spread across seasons
per = (pd.DataFrame({"Season": train.Season, "model": a_model, "seed": a_seed, "ll": ll_game})
         .groupby("Season").mean())
K = len(per)
se_season = lambda col: float(per[col].std(ddof=1) / np.sqrt(K))

# Paired bootstrap over games (same resampled games for both methods)
brng = np.random.default_rng(SEED)
n = len(train)
boot = {"model": [], "seed": [], "gap": [], "ll": []}
for _ in range(N_BOOT):
    i = brng.integers(0, n, n)
    boot["model"].append(a_model[i].mean())
    boot["seed"].append(a_seed[i].mean())
    boot["gap"].append(a_model[i].mean() - a_seed[i].mean())
    boot["ll"].append(ll_game[i].mean())
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

res = {
    "raw_tourney_games": n_raw,
    "dropped_first_four": n_ff,
    "dropped_missing": n_missing,
    "games_kept": len(df),
    "seasons_all": [int(seasons[0]), int(seasons[-1])],
    "n_seasons": len(seasons),
    "test_seasons_untouched": [int(s) for s in test_seasons],
    "test_games_untouched": int(df.Season.isin(test_seasons).sum()),
    "train_games": n,
    "train_seasons": K,
    "equal_seed_games_in_train": int((train.SeedDiff == 0).sum()),
    "share_team_A_wins": float(df.y.mean()),
    "C_grid": CS,
    "C_final_on_all_train": C_final,
    "C_chosen_per_outer_fold": {str(k): int(v) for k, v in pd.Series(chosen).value_counts().items()},
    "loso_logloss_by_C": {str(k): float(v) for k, v in cv_curve.items()},
    "model_acc": float(a_model.mean()),
    "model_acc_se_seasons": se_season("model"),
    "model_acc_boot_ci": ci(boot["model"]),
    "model_logloss": float(ll_game.mean()),
    "model_logloss_se_seasons": se_season("ll"),
    "model_logloss_boot_ci": ci(boot["ll"]),
    "seed_acc": float(a_seed.mean()),
    "seed_acc_se_seasons": se_season("seed"),
    "seed_acc_boot_ci": ci(boot["seed"]),
    "gap_acc_model_minus_seed": float(a_model.mean() - a_seed.mean()),
    "gap_boot_ci": ci(boot["gap"]),
    "coin_flip_logloss": float(np.log(2)),
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

# ---------------------------------------------------------------- figure
BLUE, GRAY = "#2a78d6", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})

# Panel A: overall accuracy with 95% bootstrap intervals
rows = [("Seed rule\n(better seed wins)", a_seed.mean(), ci(boot["seed"]), GRAY),
        ("Logistic regression\non seed difference", a_model.mean(), ci(boot["model"]), BLUE)]
for k, (name, m, (lo, hi), col) in enumerate(rows):
    a1.plot([lo, hi], [k, k], color=col, lw=2, solid_capstyle="round")
    a1.plot(m, k, "o", ms=9, color=col, mec="white", mew=1.5)
    a1.text(hi + 0.004, k, f"{m:.1%}", va="center", color=INK, fontsize=10)
a1.axvline(0.5, color=INK2, lw=1, ls=":")
a1.text(0.503, 1.45, "coin flip 50%", color=INK2, fontsize=8.5, va="center")
a1.set_yticks([0, 1], [r[0] for r in rows])
a1.set_ylim(-0.6, 1.7)
a1.set_xlim(0.48, 0.78)
a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
a1.set_xlabel("Share of games picked correctly (95% interval)")
a1.set_title("A. Both methods pick about 71% of winners", loc="left", color=INK, fontsize=11)

# Panel B: what logistic regression adds is a win probability for each seed gap
tr = train.assign(gap=train.SeedDiff.abs(),
                  better_won=np.where(train.SeedDiff < 0, train.y, 1 - train.y))
obs = tr[tr.gap > 0].groupby("gap").better_won.agg(["mean", "size"])
obs = obs[obs["size"] >= 20]
m_final = fit(train, C_final)
gaps = np.arange(1, 16)
curve = m_final.predict_proba(pd.DataFrame({"SeedDiff": -gaps}))[:, 1]   # better seed as team A
a2.plot(gaps, curve, color=BLUE, lw=2, label="Logistic regression")
a2.scatter(obs.index, obs["mean"], s=np.clip(obs["size"] / 2.5, 18, 120), color=GRAY,
           edgecolor="white", linewidth=1.5, zorder=3, label="Actual win rate (dot size = games)")
a2.axhline(0.5, color=INK2, lw=1, ls=":")
a2.set_xlim(0.3, 15.7)
a2.set_ylim(0.35, 1.05)
a2.set_xticks(gaps)
a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
a2.set_xlabel("Seed gap (e.g. 1 vs 16 = 15)")
a2.set_ylabel("Better seed's win rate")
a2.set_title("B. The bigger the seed gap, the safer the favorite", loc="left", color=INK, fontsize=11)
a2.legend(frameon=False, fontsize=8.5, loc="lower right")

for ax in (a1, a2):
    ax.grid(axis="x" if ax is a1 else "y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.text(0.01, 0.01,
         f"Men's NCAA tournaments {int(train_seasons[0])}-{int(train_seasons[-1])} "
         f"({n:,} games, First Four excluded). Each season scored by a model trained on the other seasons. "
         f"Tournaments {int(test_seasons[0])}-{int(test_seasons[-1])} are held out for the final test.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT_DIR}/baseline.png", dpi=200, facecolor="white")
