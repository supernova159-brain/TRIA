"""
TRIA ERP Analysis 
Figures produced:
  Fig 1 — Grand average ERPs by congruency (FRN + P300 panels)
  Fig 2 — Grand average ERPs by interaction type (FRN + P300 panels)
  Fig 3 — Difference waves: incongruent − congruent, no_reveal − congruent
  Fig 4 — Topographic maps at key time points (difference wave)
  Fig 5 — Single-trial ERP dynamics heatmap + attenuation regression
  Fig 6 — Alpha suppression by interaction type (START epochs)

Statistical tests:
  • RM-ANOVA: congruency main effect (FRN, P300)
  • RM-ANOVA: interaction type main effect (FRN, P300)
  • Paired t-test: resolved vs unresolved (FRN, P300)
  • Post-hoc Bonferroni-corrected pairwise tests where ANOVA is significant
"""

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
#matplotlib.figure.Figure.savefig = lambda *a, **k: None 
import numpy as np
import mne
from pathlib import Path
import pandas as pd
import pingouin as pg
from scipy.stats import sem, ttest_rel

# =============================================================
# CONFIG
# =============================================================
root = Path("/Users/antoniagergen/Desktop/TRIA/TRIA_results")
subjects = sorted([p for p in root.iterdir() if p.is_dir()])

# =============================================================
# SUBJECT / TRIAL SELECTION
# =============================================================
mode = input("Run on 'all' subjects/trials or 'select' specific ones? [all/select]: ").strip().lower()

trial_limits = {}  # subject_folder_name -> max number of trials to keep

if mode == "select":
    raw = input(
        "Enter subject, trial_count pairs separated by commas "
        "(e.g. subj01, 25, subj02, 360): "
    ).strip()
    tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]

    if len(tokens) % 2 != 0:
        raise ValueError(
            "Input must contain pairs of subject, trial_count "
            "(e.g. subj01, 25, subj02, 360)."
        )

    for i in range(0, len(tokens), 2):
        subj_name = tokens[i]
        try:
            n_trials = int(tokens[i + 1])
        except ValueError:
            raise ValueError(f"'{tokens[i + 1]}' is not a valid trial count for {subj_name}.")
        trial_limits[subj_name] = n_trials

    selected_names = list(trial_limits.keys())
    available_names = {s.name for s in subjects}
    missing = [n for n in selected_names if n not in available_names]
    if missing:
        print(f"Warning: these subjects were not found in {root} and will be skipped: {missing}")

    subjects = [s for s in subjects if s.name in selected_names]

    print(f"\nRunning on {len(subjects)} selected subject(s):")
    for s in subjects:
        print(f"  {s.name}: {trial_limits[s.name]} trials")

elif mode != "all":
    raise ValueError("Please enter either 'all' or 'select'.")
else:
    print(f"\nRunning on all {len(subjects)} subjects, all available trials.")


def limit_epochs(epochs, sub):
    "permits trial caps for specific subjects, if requested"
    if sub in trial_limits:
        n_max = trial_limits[sub]
        if n_max < len(epochs):
            return epochs[:n_max]
        else:
            print(f"  Note: requested {n_max} trials for {sub}, but only {len(epochs)} available — using all.")
    return epochs

# ROI channels
chs_FRN  = ["FC1", "FC2", "Cz"] 
chs_P300 = ["Pz", "CP1", "CP2", "Cz"]
chs_SPN = ["Cz", "CP1", "CP2", "Pz"]

# Analysis time windows (seconds, relative to reveal onset)
win_FRN = (0.200, 0.270)
win_P300 = (0.300, 0.600) 
win_SPN = (0.100, 0.400)

# Conditions and plot colors
conditions = ["congruent", "incongruent", "no_reveal"]
interaction_types = ["generalization", "alliance", "displacement", "defense"]

condition_colors = {
    "congruent":   "#2166ac",
    "incongruent": "#d65f4dcb",
    "no_reveal":   "#692c78",
}
interaction_colors = {
    "generalization": "#E69F00",
    "alliance":       "#56B4E9",
    "displacement":   "#09A179",
    "defense":        "#CC79A7",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ['Times New Roman'],
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# =============================================================
# HELPER — mean amplitude in ROI window
# =============================================================
def roi_mean(evoked, chs, win):
    """Return mean amplitude (µV) over channels and time window."""
    available = [c for c in chs if c in evoked.ch_names]
    idx = [evoked.ch_names.index(c) for c in available]
    t = evoked.times
    t_mask = (t >= win[0]) & (t <= win[1])
    return evoked.data[np.ix_(idx, t_mask)].mean() * 1e6

def roi_peak(evoked_or_data, chs, win, positive=True):
    """
    Returns peak amplitude (µV) and latency (ms).

    positive=True  -> maximum (P300)
    positive=False -> minimum (FRN)
    """
    available = [c for c in chs if c in evoked_or_data.ch_names]
    idx = [evoked_or_data.ch_names.index(c) for c in available]

    t = evoked_or_data.times
    mask = (t >= win[0]) & (t <= win[1])

    roi = evoked_or_data.data[idx].mean(axis=0)

    if positive:
        peak_idx = np.argmax(roi[mask])
    else:
        peak_idx = np.argmin(roi[mask])

    amp = roi[mask][peak_idx] * 1e6
    latency = t[mask][peak_idx] * 1000

    return amp, latency
# =============================================================
# 1. LOADING EPOCHS AND EXTRACTING SINGLE-TRIAL AMPLITUDES
# =============================================================
all_epochs     = []
all_trial_data = []

for subj_dir in subjects:
    sub = subj_dir.name
    epo_file = subj_dir / f"{sub}_reveal-epo.fif"
    if not epo_file.exists():
        print(f"Missing {epo_file}, skipping")
        continue

    print(f"Loading {sub}")
    epochs = mne.read_epochs(epo_file, preload=True)
    epochs = limit_epochs(epochs, sub)
    epochs.metadata["subject"] = sub
    all_epochs.append(epochs)

    # Single-trial amplitudes for all statistical tests
    for comp_name, chs, win in [("FRN", chs_FRN, win_FRN), ("P300", chs_P300, win_P300)]:
        available = [c for c in chs if c in epochs.ch_names]
        tmin, tmax = win

        for cond in conditions:
            try:
                sub_ep = epochs[f"trial_type == '{cond}'"]
                crop = sub_ep.copy().pick(available).crop(tmin=tmin, tmax=tmax)

                data = crop.get_data()
                times = crop.times * 1000
            except Exception as e:
                print(f"  {sub}/{cond}: {e}")
                continue
            if len(sub_ep) == 0:
                continue

            data = sub_ep.copy().pick(available).crop(tmin=tmin, tmax=tmax).get_data()
            for i, trial in enumerate(data):

                roi = trial.mean(axis=0)

                if comp_name == "FRN":
                    peak_index = np.argmin(roi)
                else:
                    peak_index = np.argmax(roi)

                peak_amp = roi[peak_index] * 1e6
                latency = times[peak_index]

                mean_amp = roi.mean() * 1e6

                all_trial_data.append(dict(
                    subject=sub,
                    condition=cond,
                    component=comp_name,

                    amplitude=mean_amp,           

                    peak_amplitude=peak_amp,      
                    peak_latency=latency,         

                    trial_number=(
                        sub_ep.metadata["trial_number"].iloc[i]
                        if "trial_number" in sub_ep.metadata.columns else i
                    ),
                    interaction_type=(
                        sub_ep.metadata["interaction_type"].iloc[i]
                        if "interaction_type" in sub_ep.metadata.columns else None
                    ),
                ))

grand_epochs = mne.concatenate_epochs(all_epochs)
print(f"\nLoaded {len(all_epochs)} subjects, {len(grand_epochs)} total epochs")
trial_df = pd.DataFrame(all_trial_data)

# =============================================================
# 2. ERP TIMESERIES + TOPOGRAPHY AT 190–230 ms
# =============================================================
print("\nPlotting ERP timeseries and topomaps...")

topo_times = [0.150, 0.220, 0.290, 0.350, 0.420]  # seconds

for cond in conditions:
    try:
        ev = grand_epochs[f"trial_type == '{cond}'"].average()
    except Exception:
        print(f"Skipping {cond}, no data.")
        continue

    # A) ERP TIME SERIES (condition-specific ROI)
    times = ev.times * 1000

    # Select ROI based on condition
    if cond == "no_reveal":
        roi_chs = chs_SPN
        roi_name = "SPN"
    else:  # congruent and incongruent
        roi_chs = sorted(list(set(chs_FRN + chs_P300)))  # union of both ROIs
        roi_name = "FRN+P300"

    picks = mne.pick_channels(ev.ch_names, include=roi_chs)
    amp = ev.data[picks].mean(axis=0) * 1e6

    fig_ts, ax_ts = plt.subplots(figsize=(8, 5))

    ax_ts.plot(times, amp, color=condition_colors[cond], linewidth=2)

    ax_ts.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_ts.axvline(0, color="black", linestyle=":", linewidth=0.8)

    ax_ts.set_xlim(-200, 800)
    ax_ts.set_ylim(13, -7)  # positivity plotted downward

    ax_ts.set_title(f"{cond} ({roi_name} ROI)")
    ax_ts.set_xlabel("Time (ms)")
    ax_ts.set_ylabel("Amplitude (µV)")

    fig_ts.tight_layout()
    fig_ts.savefig(root / f"ERP_timeseries_{cond}.png", dpi=300)
    plt.close(fig_ts)

    # B) TOPOGRAPHY (190–230 ms)
    fig_topo = ev.plot_topomap(
        times=topo_times,
        ch_type="eeg",
        cmap="RdBu_r",
        contours=6,
        colorbar=True,
        show=False,
        time_unit="s"
    )

    fig_topo.suptitle(
        f"Topomaps {cond}",
        fontsize=12,
        fontweight="bold"
    )

    fig_topo.savefig(
        root / f"topomap_{cond}.png",
        dpi=300
    )
    plt.close(fig_topo)

print("Saved ERP timeseries + topomaps for all conditions.")

# =============================================================
# FIG 1 — Grand average ERPs by congruency
# =============================================================
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, roi_chs, win, comp in zip(
    axes,
    [chs_FRN, chs_P300],
    [win_FRN,  win_P300],
    ["FRN",    "P300"],
):
    for cond in conditions:
        try:
            evoked = grand_epochs[f"trial_type == '{cond}'"].average()
        except Exception:
            continue
        available = [c for c in roi_chs if c in evoked.ch_names]
        idx  = [evoked.ch_names.index(c) for c in available]
        amp  = evoked.data[idx].mean(axis=0) * 1e6
        times = evoked.times * 1000
        ax.plot(times, amp, label=cond, color=condition_colors[cond], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{comp} — {', '.join(roi_chs)}")
    ax.invert_yaxis()
    ax.legend()

fig1.suptitle("Grand Average ERPs by Congruency", fontsize=13, fontweight="bold")
fig1.tight_layout()
fig1.savefig(root / "fig1_grand_average_congruency.png", dpi=300)

# =============================================================
# FIG 2 — Grand average ERPs by interaction type
# =============================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, roi_chs, win, comp in zip(
    axes,
    [chs_FRN, chs_P300],
    [win_FRN,  win_P300],
    ["FRN",    "P300"],
):
    for itype in interaction_types:
        try:
            evoked = grand_epochs[f"interaction_type == '{itype}'"].average()
        except Exception:
            continue
        available = [c for c in roi_chs if c in evoked.ch_names]
        idx  = [evoked.ch_names.index(c) for c in available]
        amp  = evoked.data[idx].mean(axis=0) * 1e6
        times = evoked.times * 1000
        ax.plot(times, amp, label=itype, color=interaction_colors[itype], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{comp} — {', '.join(roi_chs)}")
    ax.invert_yaxis()
    ax.legend()

fig2.suptitle("Grand Average ERPs by Interaction Type", fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig(root / "fig2_grand_average_interaction_type.png", dpi=300)

# =============================================================
# FIG 2B — Grand average P300 by interaction type and trial type
# =============================================================
fig2B, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for ax, itype in zip(axes.ravel(), ["generalization", "alliance", "displacement", "defense"]):
    for cond in ["congruent", "incongruent", "no_reveal"]:
        try:
            evoked = grand_epochs[
                f"interaction_type == '{itype}' and trial_type == '{cond}'"
            ].average()
        except Exception:
            continue

        available = [c for c in chs_P300 if c in evoked.ch_names]
        if len(available) == 0:
            continue

        idx = [evoked.ch_names.index(c) for c in available]
        amp = evoked.data[idx].mean(axis=0) * 1e6
        times = evoked.times * 1000

        ax.plot(times, amp, label=cond, color=condition_colors[cond], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylim(15, -7.5)
    ax.set_title(itype)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()

for ax in axes.ravel():
    ax.invert_yaxis()

fig2B.suptitle(
    "Grand Average P300 by Trial Type Within Each Interaction Type",
    fontsize=13,
    fontweight="bold",
)
fig2B.tight_layout()
fig2B.savefig(root / "fig2B_P300_by_trialtype_within_interaction.png", dpi=300)

# =============================================================
# Per-subject P300 latency check (incongruent) — diagnostic only
# =============================================================
comp, chs, win, positive = "P300", chs_P300, win_P300, True
cond = "incongruent"

print(f"{'subject':<10}{'latency (ms)':<15}{'amplitude (µV)'}")

for i, epochs in enumerate(all_epochs):
    try:
        evoked = epochs[f"trial_type == '{cond}'"].average()
    except Exception:
        continue

    available = [c for c in chs if c in evoked.ch_names]
    idx = [evoked.ch_names.index(c) for c in available]

    roi = evoked.data[idx].mean(axis=0)
    mask = (evoked.times >= win[0]) & (evoked.times <= win[1])
    roi_win = roi[mask]
    times_win = evoked.times[mask]

    peak = np.argmax(roi_win) if positive else np.argmin(roi_win)
    lat = times_win[peak] * 1000
    amp = roi_win[peak] * 1e6

    subj_id = getattr(epochs, "subject_id", i)  # fallback to index if not set
    flag = " <-- near window edge" if (lat - win[0]*1000 < 20 or win[1]*1000 - lat < 20) else ""
    print(f"{subj_id:<10}{lat:<15.0f}{amp:<15.2f}{flag}")

## =============================================================
# SENSOR COMPARISON
# =============================================================

chs_frontal_check = ["FC1", "FC2", "F3", "F4"]
win_frontal_check = (0.250, 0.450)

frontal_rows = []

for epochs in all_epochs:
    subj_id = epochs.metadata["subject"].iloc[0]

    for cond in ["congruent", "incongruent"]:
        try:
            evoked = epochs[f"trial_type == '{cond}'"].average()
        except Exception:
            continue
        if evoked.nave == 0:
            continue

        available = [c for c in chs_frontal_check if c in evoked.ch_names]
        if len(available) == 0:
            continue
        idx = [evoked.ch_names.index(c) for c in available]

        roi = evoked.data[idx].mean(axis=0)
        mask = (evoked.times >= win_frontal_check[0]) & (evoked.times <= win_frontal_check[1])

        mean_amp = roi[mask].mean() * 1e6
        frontal_rows.append(dict(subject=subj_id, condition=cond, amplitude=mean_amp))

frontal_df = pd.DataFrame(frontal_rows)

print("\nFrontal (FC1, FC2, F3, F4), 250-450 ms — mean amplitude by condition:")
summary = frontal_df.groupby("condition")["amplitude"].agg(["mean", "std", "count"])
print(summary)

# Quick paired comparison, same structure as your congruency t-tests elsewhere
frontal_wide = frontal_df.pivot(index="subject", columns="condition", values="amplitude").dropna()
t_frontal, p_frontal = ttest_rel(frontal_wide["congruent"], frontal_wide["incongruent"])
print(f"\nCongruent vs incongruent, frontal ROI: t({len(frontal_wide)-1}) = {t_frontal:.2f}, p = {p_frontal:.3f}")

print(f"\nFor comparison, centro-parietal P300 ROI ({', '.join(chs_P300)}), {win_P300[0]*1000:.0f}-{win_P300[1]*1000:.0f} ms, "
      f"mean amplitude: congruent = 5.49 µV, incongruent = 5.42 µV (see main analysis)")

## =============================================================
# FIG 3 — FRN
# =============================================================

win_FRN_narrow = (0.190, 0.270)  # 190–270 ms
frn_conditions = ["congruent", "incongruent", "no_reveal"]  # no_reveal excluded

fig3 = plt.figure(figsize=(14, 5))
gs = fig3.add_gridspec(1, 4, width_ratios=[3, 3, 1, 1])
ax_wave = fig3.add_subplot(gs[0, 0:2])
ax_topo_cong   = fig3.add_subplot(gs[0, 2])
ax_topo_incong = fig3.add_subplot(gs[0, 3])

topo_axes = {"congruent": ax_topo_cong, "incongruent": ax_topo_incong}
peak_info = {}  # store peak latency + evoked per condition for topomaps

for cond in frn_conditions:
    try:
        evoked = grand_epochs[f"trial_type == '{cond}'"].average()
    except Exception:
        continue

    available = [c for c in chs_FRN if c in evoked.ch_names]
    if len(available) == 0:
        continue
    idx = [evoked.ch_names.index(c) for c in available]

    amp   = evoked.data[idx].mean(axis=0) * 1e6
    times = evoked.times * 1000

    ax_wave.plot(times, amp, label=cond,
                 color=condition_colors[cond], linewidth=2)
    win_mask = (evoked.times >= win_FRN_narrow[0]) & (evoked.times <= win_FRN_narrow[1])
    roi_trace = evoked.data[idx].mean(axis=0)  # volts, ROI-averaged, full time course
    win_trace = roi_trace[win_mask]
    win_times = evoked.times[win_mask]

    peak_idx = np.argmin(win_trace)          # most negative point
    peak_latency = win_times[peak_idx]        # seconds
    peak_amp_uv = win_trace[peak_idx] * 1e6
    print(cond, peak_latency * 1000, peak_amp_uv)

    peak_info[cond] = (evoked, peak_latency, peak_amp_uv)

# shade the FRN window
ax_wave.axvspan(win_FRN_narrow[0] * 1000, win_FRN_narrow[1] * 1000,
                 color="gray", alpha=0.12, zorder=0)

ax_wave.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_wave.axvline(0, color="black", linewidth=0.6, linestyle=":")
ax_wave.set_xlabel("Time (ms)")
ax_wave.set_ylabel("Amplitude (µV)")
ax_wave.set_title(f"FRN — {', '.join(chs_FRN)}")
ax_wave.invert_yaxis()
ax_wave.legend()

# --------------------------------------------------------
# Topomaps at each condition's peak negative latency
# --------------------------------------------------------
for cond, ax_topo in topo_axes.items():
    if cond not in peak_info:
        continue
    evoked, peak_latency, peak_amp_uv = peak_info[cond]

    evoked.plot_topomap(
        times=peak_latency,
        axes=ax_topo,
        show=False,
        colorbar=False,
        time_format="",
        sensors=True,
    )
    ax_topo.set_title(f"{cond}\n{peak_latency*1000:.0f} ms, {peak_amp_uv:.2f} µV",
                       fontsize=9)

fig3.suptitle("FRN — Congruent and Incongruent",
              fontsize=13, fontweight="bold")
fig3.tight_layout()
fig3.savefig(root / "fig3_FRN_congruency_topomaps.png", dpi=300)

# =============================================================
# P300 — Congruent vs Incongruent reveals, restricted to trials
# where the participant's response matches theory prediction
# =============================================================

CORRECT_COL   = "correct"      
TRIALTYPE_COL = "trial_type"

def plot_p300_theory_tracking(grand_epochs, correct_value, title_label, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in ["congruent", "incongruent"]:
        query = f"{TRIALTYPE_COL} == '{cond}' and {CORRECT_COL} == {correct_value}"
        try:
            evoked = grand_epochs[query].average()
        except Exception:
            continue

        available = [c for c in chs_P300 if c in evoked.ch_names]
        if len(available) == 0:
            continue
        idx = [evoked.ch_names.index(c) for c in available]

        amp = evoked.data[idx].mean(axis=0) * 1e6
        times = evoked.times * 1000
        n_trials = evoked.nave

        ax.plot(
            times, amp,
            label=f"{cond} (n={n_trials})",
            color=condition_colors[cond],
            linewidth=2,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"P300 — {title_label} — {', '.join(chs_P300)}")
    ax.invert_yaxis()
    ax.legend()

    plt.tight_layout()
    plt.savefig(root / filename, dpi=300)
    plt.close(fig)

# --------------------------------------------------------
# P300 for theory-consistent and theory-deviant responses
# --------------------------------------------------------
plot_p300_theory_tracking(
    grand_epochs,
    correct_value=1,
    title_label="Theory-Congruent Responses (correct=1)",
    filename="fig_P300_theory_congruent_congruency.png",
)

plot_p300_theory_tracking(
    grand_epochs,
    correct_value=0,
    title_label="Theory-Deviant Responses (correct=0)",
    filename="fig_P300_theory_deviant_congruency.png",
)

# =============================================================
# Trial-level LMM: P300 amplitude ~ trial_type * correct + (1|subject)
# =============================================================
import statsmodels.formula.api as smf
rows = []
for subj_idx, epochs in enumerate(all_epochs):
    subj_id = getattr(epochs, "subject_id", subj_idx)

    for cond in ["congruent", "incongruent"]:
        for correct_val in [0, 1]:
            query = f"trial_type == '{cond}' and correct == {correct_val}"
            try:
                sub_epochs = epochs[query]
            except Exception:
                continue

            if len(sub_epochs) == 0:
                continue

            available = [c for c in chs_P300 if c in sub_epochs.ch_names]
            idx = [sub_epochs.ch_names.index(c) for c in available]

            data = sub_epochs.get_data()[:, idx, :].mean(axis=1)  # trials x time
            times = sub_epochs.times
            mask = (times >= win_P300[0]) & (times <= win_P300[1])

            # one row per trial: mean amplitude in win_P300
            trial_amps = data[:, mask].mean(axis=1) * 1e6

            for amp in trial_amps:
                rows.append(dict(subject=subj_id, trial_type=cond,
                                  correct=correct_val, amplitude=amp))

df_trials = pd.DataFrame(rows)
print(f"Total trials in model: {len(df_trials)}")
print(df_trials.groupby(["trial_type", "correct"])["amplitude"].count())

# --------------------------------------------------------
# Mixed model: random intercept per subject
# --------------------------------------------------------
df_trials["trial_type"] = df_trials["trial_type"].astype("category")
df_trials["correct"] = df_trials["correct"].astype("category")

model = smf.mixedlm(
    "amplitude ~ C(trial_type, Treatment('congruent')) * C(correct, Treatment(1))",
    data=df_trials,
    groups=df_trials["subject"],
)
result = model.fit(method=["lbfgs"], maxiter=2000)
print(result.summary())
print("Converged:", result.converged)

df_trials["amplitude_z"] = (df_trials["amplitude"] - df_trials["amplitude"].mean()) / df_trials["amplitude"].std()

model2 = smf.mixedlm(
    "amplitude_z ~ C(trial_type, Treatment('congruent')) * C(correct, Treatment(1))",
    data=df_trials,
    groups=df_trials["subject"],
)
result2 = model2.fit(method=["lbfgs"], maxiter=2000)
print(result2.summary())
print("Converged:", result2.converged)


## Repeat for FRN 

# =============================================================
# FRN — Congruent vs Incongruent reveals, restricted to trials
# where the participant's response matches theory prediction
# =============================================================

def plot_frn_theory_tracking(grand_epochs, correct_value, title_label, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in ["congruent", "incongruent"]:
        query = f"{TRIALTYPE_COL} == '{cond}' and {CORRECT_COL} == {correct_value}"
        try:
            evoked = grand_epochs[query].average()
        except Exception:
            continue

        available = [c for c in chs_FRN if c in evoked.ch_names]
        if len(available) == 0:
            continue
        idx = [evoked.ch_names.index(c) for c in available]

        amp = evoked.data[idx].mean(axis=0) * 1e6
        times = evoked.times * 1000
        n_trials = evoked.nave

        ax.plot(
            times, amp,
            label=f"{cond} (n={n_trials})",
            color=condition_colors[cond],
            linewidth=2,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"FRN — {title_label} — {', '.join(chs_FRN)}")
    ax.invert_yaxis()
    ax.legend()

    plt.tight_layout()
    plt.savefig(root / filename, dpi=300)
    plt.close(fig)

plot_frn_theory_tracking(
    grand_epochs,
    correct_value=1,
    title_label="Theory-Congruent Responses (correct=1)",
    filename="fig_FRN_theory_congruent_congruency.png",
)

plot_frn_theory_tracking(
    grand_epochs,
    correct_value=0,
    title_label="Theory-Deviant Responses (correct=0)",
    filename="fig_FRN_theory_deviant_congruency.png",
)

# =============================================================
# Trial-level LMM: FRN amplitude ~ trial_type * correct + (1|subject)
# =============================================================
rows_frn = []
for subj_idx, epochs in enumerate(all_epochs):
    subj_id = getattr(epochs, "subject_id", subj_idx)

    for cond in ["congruent", "incongruent"]:
        for correct_val in [0, 1]:
            query = f"trial_type == '{cond}' and correct == {correct_val}"
            try:
                sub_epochs = epochs[query]
            except Exception:
                continue

            if len(sub_epochs) == 0:
                continue

            available = [c for c in chs_FRN if c in sub_epochs.ch_names]
            idx = [sub_epochs.ch_names.index(c) for c in available]

            data = sub_epochs.get_data()[:, idx, :].mean(axis=1)  # trials x time
            times = sub_epochs.times
            mask = (times >= win_FRN[0]) & (times <= win_FRN[1])

            trial_amps = data[:, mask].mean(axis=1) * 1e6

            for amp in trial_amps:
                rows_frn.append(dict(subject=subj_id, trial_type=cond,
                                      correct=correct_val, amplitude=amp))

df_trials_frn = pd.DataFrame(rows_frn)
print(f"Total trials in FRN model: {len(df_trials_frn)}")
print(df_trials_frn.groupby(["trial_type", "correct"])["amplitude"].count())

df_trials_frn["trial_type"] = df_trials_frn["trial_type"].astype("category")
df_trials_frn["correct"] = df_trials_frn["correct"].astype("category")

model_frn = smf.mixedlm(
    "amplitude ~ C(trial_type, Treatment('congruent')) * C(correct, Treatment(1))",
    data=df_trials_frn,
    groups=df_trials_frn["subject"],
)
result_frn = model_frn.fit(method=["lbfgs"], maxiter=2000)
print(result_frn.summary())
print("Converged:", result_frn.converged)

# =============================================================
# FIG 3 — Difference waves 
# =============================================================
try:
    ev_con = grand_epochs["trial_type == 'congruent'"].average()
    ev_inc = grand_epochs["trial_type == 'incongruent'"].average()
    ev_nor = grand_epochs["trial_type == 'no_reveal'"].average()
    _diff_ok = True
except Exception as e:
    print(f"Difference waves: {e}")
    _diff_ok = False

if _diff_ok:
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, roi_chs, win, comp in zip(
        axes,
        [chs_FRN, chs_P300],
        [win_FRN,  win_P300],
        ["FRN",    "P300"],
    ):
        available = [c for c in roi_chs if c in ev_con.ch_names]
        idx  = [ev_con.ch_names.index(c) for c in available]
        times = ev_con.times * 1000

        diff_inc = (ev_inc.data[idx].mean(axis=0) - ev_con.data[idx].mean(axis=0)) * 1e6 #effect of incongruent? Keep or not?
        diff_nor = (ev_nor.data[idx].mean(axis=0) - ev_con.data[idx].mean(axis=0)) * 1e6 #effect of congruent 
        diff_nor2 = (ev_nor.data[idx].mean(axis=0) - ev_inc.data[idx].mean(axis=0)) * 1e6 #effect of incongruent relative to no-reveal

        ax.plot(times, diff_inc, color=condition_colors["incongruent"], linewidth=2,
                label="Incongruent − Congruent")
        ax.plot(times, diff_nor, color=condition_colors["no_reveal"], linewidth=2,
                label="No-reveal − Congruent")
        ax.plot(times, diff_nor2, color=condition_colors["congruent"], linewidth=2,
                label="No-reveal − Incongruent")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude difference (µV)")
        ax.set_title(f"{comp} difference waves — {', '.join(roi_chs)}")
        ax.legend()

    fig3.suptitle("ERP Difference Waves for All Trial Types",
                  fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(root / "fig3_difference_waves.png", dpi=300)

# =============================================================
# FIG 4 — Topographic maps for incongruent and congruent reveals, P300 and FRN
# =============================================================
topo_times = [0.390,0.410, 0.430]  # seconds

ev_con  = grand_epochs["trial_type == 'congruent'"].average()
ev_inc  = grand_epochs["trial_type == 'incongruent'"].average()

fig4a = ev_con.plot_topomap(
    times=topo_times, ch_type="eeg",
    cmap="RdBu_r", contours=6, colorbar=True, show=False,
)
fig4a.suptitle("Scalp topography of P300 on congruent trials", fontsize=12, fontweight="bold")
fig4a.savefig(root / "fig4a_topomap_congruent.png", dpi=300)

### PLOT HERE INCONGRUENTS 
fig4b = ev_inc.plot_topomap(
    times=topo_times, ch_type="eeg",
    cmap="RdBu_r", contours=6, colorbar=True, show=False,
)
fig4b.suptitle("Scalp topography P300 on incongruent trials", fontsize=12, fontweight="bold")
fig4b.savefig(root / "fig4b_topomap_incongruent.png", dpi=300)

# =============================================================
# FIG 4 — Subject-averaged P300 waveform (congruent, incongruent, no_reveal) by
# HMM-classified response strategy group
# =============================================================
# hmm_groups = {
#     "Single switch: theory→deviant": ["subj01", "subj05", "subj06", "subj09",
#                                        "subj16", "subj17", "subj19"],
#     "Single switch: deviant→theory": ["subj02", "subj15"],
#     "Oscillating": ["subj03", "subj07", "subj14"],
#     "Theory-consistent throughout": ["subj08", "subj10"],
#     "Random": ["subj13", "subj20"],
# }

# # --------------------------------------------------------
# # Per-subject mean ROI waveform, per condition
# # --------------------------------------------------------
# subj_condition_waves = {}  # (subject, condition) -> (times, amp array in µV)

# for epochs in all_epochs:
#     subj_id = epochs.metadata["subject"].iloc[0]

#     for cond in conditions:  # congruent, incongruent, no_reveal
#         try:
#             evoked = epochs[f"trial_type == '{cond}'"].average()
#         except Exception:
#             continue
#         if evoked.nave == 0:
#             continue

#         available = [c for c in chs_P300 if c in evoked.ch_names]
#         if len(available) == 0:
#             continue
#         idx = [evoked.ch_names.index(c) for c in available]

#         amp = evoked.data[idx].mean(axis=0) * 1e6
#         subj_condition_waves[(subj_id, cond)] = (evoked.times * 1000, amp)

# # --------------------------------------------------------
# # Plot: one panel per HMM group, subject-averaged per condition
# # --------------------------------------------------------
# fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
# axes_flat = axes.ravel()

# for ax, (group_name, subj_list) in zip(axes_flat, hmm_groups.items()):
#     n_found = 0
#     for cond in conditions:
#         curves = []
#         for subj_id in subj_list:
#             key = (subj_id, cond)
#             if key in subj_condition_waves:
#                 curves.append(subj_condition_waves[key][1])
#         if len(curves) == 0:
#             continue

#         times = subj_condition_waves[(subj_list[0], cond)][0] if (subj_list[0], cond) in subj_condition_waves else None
#         # fall back: grab times from whichever subject actually has this condition
#         for subj_id in subj_list:
#             if (subj_id, cond) in subj_condition_waves:
#                 times = subj_condition_waves[(subj_id, cond)][0]
#                 break

#         mean_curve = np.mean(curves, axis=0)
#         n_found = max(n_found, len(curves))
#         ax.plot(times, mean_curve, label=cond, color=condition_colors[cond], linewidth=2)

#     ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
#     ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
#     ax.set_title(f"{group_name} (n={len(subj_list)})", fontsize=10)
#     ax.set_xlabel("Time (ms)")
#     ax.set_ylabel("Amplitude (µV)")
#     ax.invert_yaxis()
#     ax.legend(fontsize=8)

# # hide the unused 6th panel (5 groups in a 2x3 grid)
# axes_flat[-1].axis("off")

# fig.suptitle("P300 Waveform by HMM-Classified Response Strategy (Subject-Averaged)",
#              fontsize=13, fontweight="bold")
# fig.tight_layout()
# fig.savefig(root / "fig_P300_by_HMM_group.png", dpi=300)

# =============================================================
# STATISTICS — 1. RM-ANOVA: congruency main effect
# =============================================================
print("\n" + "="*60)
print("STATISTICS")
print("="*60)
print("\n[1] RM-ANOVA: main effect of congruency (FRN, P300)")

for comp in ["FRN", "P300"]:
    comp_df = trial_df[trial_df["component"] == comp]
    # One mean amplitude per subject × condition cell
    cell_means = (
        comp_df.groupby(["subject", "condition"])["amplitude"]
        .mean().reset_index()
    )
    if cell_means["subject"].nunique() < 2:
        print(f"  {comp}: <2 subjects, skipping")
        continue
    try:
        aov = pg.rm_anova(data=cell_means, dv="amplitude",
                           within="condition", subject="subject", detailed=False, effsize="np2")
        row = aov[aov["Source"] == "condition"].iloc[0]
        print(f"\n  {comp}: F({int(row['ddof1'])},{int(row['ddof2'])}) = "
              f"{row['F']:.2f}, p = {row['p_unc']:.4f}, η²p = {row['np2']:.3f}")

        if row["p_unc"] < .05:
            ph = pg.pairwise_tests(data=cell_means, dv="amplitude",
                                    within="condition", subject="subject",
                                    padjust="bonf")
            print(f"  Post-hoc (Bonferroni):")
            print("  Available columns:", ph.columns.tolist())  

            # Robustly pick whatever p-value column actually exists
            p_candidates = ["p-corr", "p-unc", "p-val", "p"]
            p_col = next((c for c in p_candidates if c in ph.columns), None)

            if p_col is None:
                print(ph.to_string(index=False))  
            else:
                cols_to_show = [c for c in ["A", "B", "T", "dof", p_col] if c in ph.columns]
                print(ph[cols_to_show].to_string(index=False))
    except Exception as e:
        print(f"  {comp} ANOVA failed: {e}")

print(cell_means.pivot(index="subject", columns="condition", values="amplitude")) #!

# =============================================================
# STATISTICS — 2. RM-ANOVA: interaction type main effect
# =============================================================
print("\n[2] RM-ANOVA: main effect of interaction type (FRN, P300)")

for comp in ["FRN", "P300"]:
    comp_df = trial_df[trial_df["component"] == comp].dropna(subset=["interaction_type"])
    cell_means = (
        comp_df.groupby(["subject", "interaction_type"])["amplitude"]
        .mean().reset_index()
    )
    if cell_means["subject"].nunique() < 2:
        print(f"  {comp}: <2 subjects, skipping")
        continue
    try:
        aov = pg.rm_anova(data=cell_means, dv="amplitude",
                           within="interaction_type", subject="subject", detailed=False, effsize="np2")
        row = aov[aov["Source"] == "interaction_type"].iloc[0]
        print(f"\n  {comp}: F({int(row['ddof1'])},{int(row['ddof2'])}) = "
              f"{row['F']:.2f}, p = {row['p_unc']:.4f}, η²p = {row['np2']:.3f}")

        if row["p_unc"] < 0.05:
            ph = pg.pairwise_tests(data=cell_means, dv="amplitude",
                                within="interaction_type", subject="subject",
                                padjust="bonf")
            print("PAIRWISE COLUMNS:", ph.columns.tolist())
            print(ph.head())
            print("  Post-hoc (Bonferroni):")
            # Correct column names from pingouin
            p_cols = [c for c in ph.columns if c.startswith("p")]
            if len(p_cols) == 0:
                raise ValueError("No p-value column found in pairwise_tests output.")
            p_col = p_cols[0]

            print(f"  Post-hoc, Bonferroni Corrected:")
            print(ph[["A", "B", "T", "dof", p_col]].to_string(index=False))
    except Exception as e:
        print(f"  {comp} ANOVA failed: {e}")

# =============================================================
# PEAK ERP SUMMARY
# =============================================================
print("\n" + "=" * 30)
print("PEAK ERP SUMMARY")
print("=" * 30)

for comp, chs, win, positive in [
    ("FRN", chs_FRN, win_FRN, False),
    ("P300", chs_P300, win_P300, True),
]:

    print(f"\n{comp}")

    for cond in ["congruent", "incongruent"]:

        amps = []
        lats = []

        for epochs in all_epochs:

            try:
                evoked = epochs[f"trial_type == '{cond}'"].average()
            except Exception:
                continue

            available = [c for c in chs if c in evoked.ch_names]
            idx = [evoked.ch_names.index(c) for c in available]

            roi = evoked.data[idx].mean(axis=0)

            mask = (evoked.times >= win[0]) & (evoked.times <= win[1])

            roi = roi[mask]
            times = evoked.times[mask]

            peak = np.argmax(roi) if positive else np.argmin(roi)
            lats.append(times[peak] * 1000)
            amps.append(roi.mean() * 1e6)   # instead of roi[peak] * 1e6 

        print(f"\n{cond.capitalize()}")
        print(f"Amplitude: {np.mean(amps):.2f} ± {np.std(amps, ddof=1):.2f} µV")
        print(f"Latency : {np.mean(lats):.0f} ± {np.std(lats, ddof=1):.0f} ms")

print("\n" + "="*60)
print("Figures saved to:", root)
print("="*60)
plt.show()
