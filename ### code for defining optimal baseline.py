
"""
Baseline window selection for TRIA reveal- and start-locked epochs.

For each subject, evaluates a set of candidate baseline windows and
selects the one that minimizes across-trial amplitude variance in a
post-stimulus window of interest (100-400 ms)

Expects, per subject, under TRIA_results/<subject>/:
    <subject>.vhdr / .eeg / .vmrk  (BrainVision raw data)
    <subject>.csv                  (trial-level behavioral metadata)
"""

import numpy as np
import mne
import pandas as pd
from pathlib import Path

root = Path("/Users/antoniagergen/Desktop/TRIA/TRIA_results")
subjects = ["subj01", "subj02", "subj03", "subj05", "subj06", "subj07",
            "subj08", "subj09", "subj10", "subj13", "subj14", "subj15",
            "subj16", "subj17", "subj19", "subj20"]

CANDIDATE_BASELINES_START = [
    (-0.200, -0.050),
    (-0.150, -0.050),
    (-0.100, -0.020),
    (-0.080, -0.010),
    (-0.050, -0.005),
]
CANDIDATE_BASELINES_REVEAL = [
    (-0.250, -0.200),  
    (-0.220, -0.175),  
    (-0.170, -0.120),  
    (-0.150, -0.100),
    (-0.120, -0.060),
    (-0.100, -0.040),
    (-0.080, -0.020),
    (-0.060, -0.010),  
]
def score_baseline(epochs, baseline):
    ep = epochs.copy().apply_baseline(baseline)
    # Mean signal in a post-stimulus window of interest per trial
    # Lower variance across trials = better baseline
    times = ep.times
    mask = (times >= 0.1) & (times <= 0.4)
    data = ep.get_data()[:, :, mask]  # trials x channels x times
    trial_means = data.mean(axis=2)   # trials x channels
    return np.mean(np.var(trial_means, axis=0))

def find_best_baseline(epochs, candidates, label=""):
    scores = []
    for bl in candidates:
        s = score_baseline(epochs, bl)
        scores.append((bl, s))
        print(f"  {label} baseline {bl}: score = {s:.6f}")
    best_bl, best_score = min(scores, key=lambda x: x[1])
    print(f"  --> {label} BEST BASELINE = {best_bl} (score={best_score:.6f})\n")
    return best_bl

start_codes    = [1, 2, 4, 8]          # trial start
response_codes = [16, 32, 64, 128]     # response

event_id_start = {
    "start/generalization": 1,
    "start/alliance":       2,
    "start/displacement":   4,
    "start/defense":        8,
}
event_id_reveal = {
    "reveal/generalization": 16,
    "reveal/alliance":       32,
    "reveal/displacement":   64,
    "reveal/defense":        128,
}

for sub in subjects:
    print(f"\n==================== {sub} ====================")
    subj_dir = root / sub

    # Load raw
    raw = mne.io.read_raw_brainvision(
        subj_dir / f"{sub}.vhdr", preload=True
    )
    raw.set_channel_types({"VEOG": "eog", "Photosensor": "misc"})

    sfreq = raw.info["sfreq"]
    shift_samples = int(0.170 * sfreq)  # 170ms in samples
    beh = pd.read_csv(subj_dir / f"{sub}.csv")
    events_all, _ = mne.events_from_annotations(raw)
    print(f"  Total VMRK events: {len(events_all)}")
    events_start    = events_all[np.isin(events_all[:, 2], start_codes)]
    events_response = events_all[np.isin(events_all[:, 2], response_codes)]
    print(f"  Start markers:    {len(events_start)}")
    print(f"  Response markers: {len(events_response)}")
    assert len(events_start) == len(beh), (
        f"Start marker count ({len(events_start)}) != CSV rows ({len(beh)})"
    )
    assert len(events_response) == len(beh), (
        f"Response marker count ({len(events_response)}) != CSV rows ({len(beh)})"
    )

   #shift by 170ms to get reveal events
    events_reveal = events_response.copy()
    events_reveal[:, 0] = events_reveal[:, 0] + shift_samples

    epochs_start = mne.Epochs(
        raw, events_start,
        event_id=event_id_start,
        tmin=-0.2, tmax=1.5,
        baseline=None,
        preload=True,
        metadata=beh.reset_index(drop=True),
    )
    epochs_reveal = mne.Epochs(
        raw, events_reveal,
        event_id=event_id_reveal,
        tmin=-0.3, tmax=0.8,
        baseline=None,
        preload=True,
        metadata=beh.reset_index(drop=True),
    )

    print(f"  Start epochs:  {len(epochs_start)}")
    print(f"  Reveal epochs: {len(epochs_reveal)}")

    #evaluation step 
    best_start_bl  = find_best_baseline(
        epochs_start,  CANDIDATE_BASELINES_START,  label="START"
    )
    best_reveal_bl = find_best_baseline(
        epochs_reveal, CANDIDATE_BASELINES_REVEAL, label="REVEAL"
    )

    print(f"  Best baselines for {sub}:")
    print(f"    START  : {best_start_bl}")
    print(f"    REVEAL : {best_reveal_bl}")
    print(f"    Note   : REVEAL baseline < -0.170 = pre-response window")
    print(f"             REVEAL baseline > -0.170 = post-response / pre-reveal window")