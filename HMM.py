import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import qth_survival_times
from hmmlearn.hmm import CategoricalHMM
from scipy.stats import binomtest

logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence non-fatal EM restart chatter

plt.rc('font', family='Times New Roman', weight='normal', size=12)

# =============================================================
# CONFIG
# =============================================================
root = Path("/Users/antoniagergen/Desktop/TRIA/TRIA_results")
out_dir = Path("/Users/antoniagergen/Desktop/TRIA/switch_model_output")
out_dir.mkdir(exist_ok=True)

N_RESTARTS = 10       # random restarts per subject, guards against poor local optima
MIN_TRIALS = 10       # subjects with fewer usable trials than this are skipped
MIN_STATE_SEPARATION = 0.10  # min gap in P(correct) between fitted states to trust the fit

# Dirichlet pseudo-counts on the transition matrix, biased toward self-transition
# (staying in the current strategy). Without this, an unconstrained Baum-Welch
# fit on binary, single-trial data finds it "cheaper" to flip states on every
# trial that happens to be correct/incorrect rather than recovering a slow,
# persistent latent strategy, i.e. it overfits trial-level noise as if it were
# genuine state switching. 50:1 favors runs on the order of tens of trials,
# matching the 20-50 trial switch timescale already observed behaviorally.
STICKY_TRANSMAT_PRIOR = np.array([[50.0, 1.0],
                                   [1.0, 50.0]])

subjects = sorted([p for p in root.iterdir() if p.is_dir()])

trial_rows = []
subject_summaries = []
flagged_subjects = []  # subjects whose fit didn't cleanly separate two states

# =============================================================
# PER-SUBJECT TRIAL-LEVEL HMM
# =============================================================
for subj_dir in subjects:
    subj = subj_dir.name
    behav_file = subj_dir / f"{subj}.csv"
    if not behav_file.exists():
        continue

    df = pd.read_csv(behav_file).sort_values("trial_number").reset_index(drop=True)
    if "correct" not in df.columns:
        continue
    df = df.dropna(subset=["correct"]).reset_index(drop=True)

    obs = df["correct"].astype(int).to_numpy().reshape(-1, 1)
    n_trials = len(obs)
    if n_trials < MIN_TRIALS:
        continue

    # Fits a 2-state categorical (Bernoulli) HMM on this subject's trial-by-trial
    # accuracy alone. correct == 1 means the response matched the theoretically
    # predicted target for that interaction type, regardless of trial_type, so
    # it is a valid theory-adherence signal on every trial, not just congruent
    # ones. Fitting per subject (rather than pooling with a fixed length list)
    # avoids assuming every subject has the same number of usable trials.
    best_model, best_ll = None, -np.inf
    for seed in range(N_RESTARTS):
        try:
            model = CategoricalHMM(n_components=2, n_features=2,
                                    n_iter=200, tol=1e-4, random_state=seed,
                                    transmat_prior=STICKY_TRANSMAT_PRIOR)
            model.fit(obs)
            ll = model.score(obs)
            if ll > best_ll:
                best_ll, best_model = ll, model
        except Exception:
            continue

    if best_model is None:
        continue

    p_correct_given_state = best_model.emissionprob_[:, 1]
    theory_state = int(np.argmax(p_correct_given_state))
    separation = abs(p_correct_given_state[0] - p_correct_given_state[1])

    # Permutation test: is there genuine temporal (switching) structure in
    # this subject's trial order, or would a 2-state model fit this well on
    # any random reordering of the same correct/incorrect trials? A generic
    # BIC penalty (parametric, based on asymptotic assumptions that don't
    # hold well here) turned out to be far too conservative for detecting a
    # single true switch in ~360 binary trials, and rejected genuine
    # single-switch subjects recovered correctly by the sticky-prior model
    # alone. Shuffling destroys any real switching structure while
    # preserving the subject's overall accuracy, giving a fair null.
    rng = np.random.default_rng(0)
    n_perm = 200
    shuffle_lls = np.empty(n_perm)
    shuffled = obs.copy()
    for p_i in range(n_perm):
        rng.shuffle(shuffled)
        perm_best_ll = -np.inf
        for seed in range(3):  # fewer restarts than the real fit, for speed
            try:
                m = CategoricalHMM(n_components=2, n_features=2, n_iter=100,
                                    tol=1e-3, random_state=seed,
                                    transmat_prior=STICKY_TRANSMAT_PRIOR)
                m.fit(shuffled)
                ll = m.score(shuffled)
                if ll > perm_best_ll:
                    perm_best_ll = ll
            except Exception:
                continue
        shuffle_lls[p_i] = perm_best_ll
    perm_p_value = float((shuffle_lls >= best_ll).mean())
    two_state_wins = perm_p_value < 0.05

    if two_state_wins:
        states = best_model.predict(obs)
        is_theory = (states == theory_state).astype(int)
        if separation < MIN_STATE_SEPARATION:
            flagged_subjects.append((subj, f"switching structure significant (perm p={perm_p_value:.3f}) but states poorly separated"))
    else:
        # No evidence of genuine switching structure: classify the whole
        # session by a binomial test of overall accuracy against chance
        # (0.5), rather than trusting an arbitrary 2-state split that isn't
        # earning its keep.
        test = binomtest(int(obs.sum()), n_trials, 0.5, alternative="greater")
        stable_state_is_theory = test.pvalue < 0.05
        is_theory = np.full(n_trials, int(stable_state_is_theory))
        flagged_subjects.append((subj, f"no significant switching structure (perm p={perm_p_value:.3f}); classified as stable via binomial test"))

    for i in range(n_trials):
        trial_rows.append({
            "subject": subj,
            "trial_number": int(df.loc[i, "trial_number"]),
            "correct": int(obs[i, 0]),
            "state": "theory" if is_theory[i] == 1 else "non_theory",
        })

    # ---- switch statistics, allowing back-and-forth oscillation ----
    away_idx = np.where(np.diff(is_theory) == -1)[0] + 1   # theory -> non_theory
    back_idx = np.where(np.diff(is_theory) == 1)[0] + 1     # non_theory -> theory

    started_theory = bool(is_theory[0] == 1)
    n_switches_away = len(away_idx)
    n_switches_back = len(back_idx)
    prop_theory = float(is_theory.mean())

    if n_switches_away > 0:
        first_switch_trial = int(df.loc[away_idx[0], "trial_number"])
        event_observed = 1
    else:
        first_switch_trial = int(df.loc[n_trials - 1, "trial_number"])
        event_observed = 0  # censored: never left theory-congruent responding

    if n_switches_back > 0 and n_switches_away > 0:
        first_return_trial = int(df.loc[back_idx[0], "trial_number"])
    else:
        first_return_trial = np.nan

    #decision rule
    if n_switches_away == 0 and n_switches_back == 0:
        behavior_group = "theory_throughout" if started_theory else "random_throughout"
    elif n_switches_away >= 1 and n_switches_back >= 1:
        behavior_group = "oscillator"
    else:
        behavior_group = "single_switch"

    subject_summaries.append({
        "subject": subj,
        "n_trials": n_trials,
        "started_theory": started_theory,
        "prop_theory": prop_theory,
        "n_switches_away_from_theory": n_switches_away,
        "n_switches_back_to_theory": n_switches_back,
        "first_switch_trial": first_switch_trial,
        "event_observed": event_observed,
        "first_return_trial": first_return_trial,
        "behavior_group": behavior_group,
        "p_correct_theory_state": float(p_correct_given_state[theory_state]),
        "p_correct_non_theory_state": float(p_correct_given_state[1 - theory_state]),
        "state_separation": separation,
        "switch_structure_perm_p": perm_p_value,
    })

trial_df = pd.DataFrame(trial_rows)
summary_df = pd.DataFrame(subject_summaries)

trial_df.to_csv(out_dir / "trial_level_states.csv", index=False)
summary_df.to_csv(out_dir / "subject_switch_summary.csv", index=False)

print("=== Subject-level summary ===")
print(summary_df.to_string(index=False))

if flagged_subjects:
    print("\nWARNING: the following subjects' classification is worth a manual look:")
    for subj, reason in flagged_subjects:
        print(f"  - {subj}: {reason}")

print("\n=== Behavior group counts (model output, before manual review) ===")
print(summary_df["behavior_group"].value_counts())

# =============================================================
# GENERAL STATISTICS
# =============================================================
switchers = summary_df[summary_df["event_observed"] == 1]
print(f"\nn subjects total: {len(summary_df)}")
print(f"n switchers (any switch away from theory): {len(switchers)}")
if len(switchers) > 0:
    print(f"median first-switch trial among switchers: "
          f"{switchers['first_switch_trial'].median():.1f}")
    print(f"IQR first-switch trial: "
          f"{switchers['first_switch_trial'].quantile(0.25):.1f} - "
          f"{switchers['first_switch_trial'].quantile(0.75):.1f}")
    print(f"mean n switches away among switchers: "
          f"{switchers['n_switches_away_from_theory'].mean():.2f}")

# =============================================================
# KAPLAN-MEIER: time to first switch away from theory
# =============================================================
kmf = KaplanMeierFitter()
kmf.fit(durations=summary_df["first_switch_trial"],
        event_observed=summary_df["event_observed"],
        label="Theory-consistent responding")

fig, ax = plt.subplots(figsize=(6, 5))
kmf.plot_survival_function(ax=ax)
ax.set_xlabel("Trial number")
ax.set_ylabel("Proportion still responding theory-consistently")
ax.set_title("Survival curve: retention of theory-consistent strategy")
plt.tight_layout()
plt.savefig(out_dir / "kaplan_meier_switch.png", dpi=300)
plt.show()

# qth_survival_times returns the time at which the survival function drops to
# or below q, i.e. the proportion of subjects still theory-consistent. To find
# the trial by which X% of subjects have switched away, query q = 1 - X.
target_switch_props = [0.10, 0.25, 0.50, 0.75, 0.90]
survival_targets = [1 - p for p in target_switch_props]
trial_at_percentile = qth_survival_times(survival_targets, kmf.survival_function_)
trial_at_percentile.index = [f"{int(p * 100)}% of subjects switched by" for p in target_switch_props]
print("\n=== Percentile switch times ===")
print(trial_at_percentile)
print(f"\nMedian switch trial (lifelines median_survival_time_): "
      f"{kmf.median_survival_time_}")

# =============================================================
# KAPLAN-MEIER: time spent away from theory before returning
# (directly addresses oscillation, i.e. switching back)
# =============================================================
osc = summary_df[summary_df["event_observed"] == 1].copy()
if len(osc) > 0:
    osc["returned"] = osc["first_return_trial"].notna().astype(int)
    osc["time_away_from_theory"] = np.where(
        osc["returned"] == 1,
        osc["first_return_trial"] - osc["first_switch_trial"],
        osc["n_trials"] - osc["first_switch_trial"],
    )

    kmf_return = KaplanMeierFitter()
    kmf_return.fit(durations=osc["time_away_from_theory"],
                   event_observed=osc["returned"],
                   label="Time away from theory-consistent responding")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    kmf_return.plot_survival_function(ax=ax2)
    ax2.set_xlabel("Trials since switching away from theory")
    ax2.set_ylabel("Proportion still in non-theory state")
    ax2.set_title("Survival curve: return to theory-consistent responding")
    plt.tight_layout()
    plt.savefig(out_dir / "kaplan_meier_return.png", dpi=300)
    plt.show()
    print(f"\nn subjects who switched back at least once: {osc['returned'].sum()} / {len(osc)}")

print(f"\nSaved: {out_dir / 'trial_level_states.csv'} (per-trial state labels, "
      f"for precise ERP time-locking)")
print(f"Saved: {out_dir / 'subject_switch_summary.csv'}")