from curses import raw
import sys
import matplotlib.pyplot as plt
import numpy as np
import mne
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from autoreject import AutoReject

# ----------------- CONFIGURATION -----------------
root = Path("/Users/antoniagergen/Desktop/TRIA/TRIA_results")

# List available subjects
subjects = sorted([p for p in root.iterdir() if p.is_dir()])
available = [p.name for p in subjects]
print("Available subjects:", available)

# Prompt for subject ID
sub = input("\nEnter the subject ID you want to analyse (e.g., 'subj01'): ").strip()
# Validate
if sub not in available:
    print(f"Subject '{sub}' not found in TRIA_results. Exiting.")
    raise SystemExit

# Select only that subject
subj_dir = root / sub

raw_file = subj_dir / f"{sub}.vhdr"
if not raw_file.exists():
    raise FileNotFoundError(f"Missing {raw_file}")

def process_subject(sub, subj_dir, raw_file):
    print(f"\nProcessing {sub}")
    raw = mne.io.read_raw_brainvision(raw_file, preload=True)
    print(raw.ch_names)
    print(raw.get_montage())
    raw.plot_sensors()
    raw.set_eeg_reference(['TP9', 'TP10'])

    def flag_bad_channels(raw, picks='eeg', z_thresh=3.5):
        data = raw.get_data(picks=picks)
        ch_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, misc=False)]
        variances = data.var(axis=1)
        z = (variances - np.median(variances)) / (
            np.median(np.abs(variances - np.median(variances))) * 1.4826
        )
        bads = [ch_names[i] for i in range(len(ch_names)) if np.abs(z[i]) > z_thresh]
        return bads

    auto_bads = flag_bad_channels(raw)
    print(f"{sub}: auto-flagged bad channels: {auto_bads}")

    raw.info['bads'] = list(set(raw.info['bads'] + auto_bads))

    # Remove TP9 and TP10 from bads to keep them for referencing
    raw.info['bads'] = [ch for ch in raw.info['bads'] if ch not in ['TP9', 'TP10']]

    interpolated_channels = list(raw.info['bads'])

    #Interpolation of bad channels
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)

    #Mastoids as MISC
    raw.set_channel_types({'TP9': 'misc', 'TP10': 'misc'})
    raw.plot_sensors(kind='3d')
    plt.show()
    print(f"{sub}: interpolated channels: {interpolated_channels}")

    # MNE get annotations
    events_raw, event_id_raw = mne.events_from_annotations(raw)
    print("event_id_raw:", event_id_raw)

    #Event map from found annotations
    event_id_map = {}
    for key in event_id_raw.keys():
        if key.startswith("Stimulus/S"):
            num = key.split("S")[-1].lstrip("0")
            if num == "":
                num = "0"
            event_id_map[key] = int(num)
    event_id_map = dict(sorted(event_id_map.items(), key=lambda x: x[1]))
    print("event_id_map:", event_id_map)

    # ROIs
    chs_FRN   = ["FC1", "FC2", "Cz"]
    chs_P300  = ["Pz", "CP1", "CP2", "Cz"]
    chs_alpha = ["P3", "P4", "O1", "O2", "Oz"]

    # TRIAL TYPES
    interaction_map = {
        1: "generalization",  16: "generalization",
        2: "alliance",        32: "alliance",
        4: "displacement",    64: "displacement",
        8: "defense",         128: "defense",
    }
    start_codes = [1, 2, 4, 8]
    resp_codes  = [16, 32, 64, 128]
    beh = pd.read_csv(subj_dir / f"{sub}.csv")
    beh["trial_type"] = beh["trial_type"].astype(str).str.strip().str.lower()

    # ----------------- FILTERING for ICA (1st filtering) -----------------
    raw_ica = raw.copy()
    raw_ica.notch_filter(freqs=[60, 120], fir_design="firwin")
    raw_ica.filter(l_freq=1.0, h_freq=40., fir_design="firwin")

    # ----------------- BLINK INSPECTION -----------------
    print("\n=== BLINK INSPECTION (Fp1 proxy) ===")
    from mne.preprocessing import find_eog_events
    eog_events = find_eog_events(raw_ica, ch_name='Fp1')
    blink_count = len(eog_events)
    print(f"Detected blinks via Fp1: {blink_count}")
    sfreq = raw_ica.info['sfreq']
    blink_times = eog_events[:, 0] / sfreq
    if len(blink_times) > 1:
        isi = np.diff(blink_times)
        print(f"Median ISI: {np.median(isi):.3f} s  |  Min: {np.min(isi):.3f} s  |  Max: {np.max(isi):.3f} s")
    # Plot Fp1
    raw_ica.copy().pick(['Fp1']).plot(duration=20, n_channels=1,
                                        title='Fp1 — blink proxy (check for spikes)')
    plt.show()
    # Plotting blink-locked epochs on Fp1
    if blink_count > 0:
        blink_epochs = mne.Epochs(
            raw_ica, eog_events, event_id=998,
            tmin=-0.5, tmax=0.5, picks=['Fp1', 'Fp2'],
            baseline=None, preload=True
        )
        blink_epochs.plot(n_epochs=min(50, len(blink_epochs)), picks=['Fp1', 'Fp2'])
        plt.show()

    print("\nPress ENTER to continue to ICA, or type 'q' + ENTER to quit.")
    if input().strip().lower() == 'q':
        raise SystemExit

    # ----------------- ICA -----------------
    ica = mne.preprocessing.ICA(
        n_components=20,
        random_state=97,
        max_iter=800,
        method="fastica",
    )
    # Fit ICA on EEG only (excludes misc: VEOG, Photosensor, TP9, TP10)
    picks_ica = mne.pick_types(raw_ica.info, eeg=True, eog=False, misc=False)
    ica.fit(raw_ica, picks=picks_ica)
    # --- Blink detection Fp1 ---
    eog_epochs_fp1 = mne.preprocessing.create_eog_epochs(
        raw_ica, ch_name='Fp1', tmin=-0.5, tmax=0.5
    )
    eog_epochs_fp1.apply_baseline((None, 0))
    eog_inds_fp1, eog_scores_fp1 = ica.find_bads_eog(eog_epochs_fp1, ch_name='Fp1')
    # --- Blink detection Fp2 ---
    eog_epochs_fp2 = mne.preprocessing.create_eog_epochs(
        raw_ica, ch_name='Fp2', tmin=-0.5, tmax=0.5
    )
    eog_epochs_fp2.apply_baseline((None, 0))
    eog_inds_fp2, eog_scores_fp2 = ica.find_bads_eog(eog_epochs_fp2, ch_name='Fp2')

    # Merge blink ICs
    ica.exclude = list(set(eog_inds_fp1 + eog_inds_fp2))
    # Save blink exclusions, run muscle detection on a clean slate
    blink_exclude = list(set(eog_inds_fp1 + eog_inds_fp2))
    ica.exclude = []          # <-- clear before muscle detection

    # Muscle detection (using default threshold of 0.5, very conservative, IC rejection adjusted manually)
    try:
        muscle_inds, muscle_scores = ica.find_bads_muscle(raw_ica, threshold=0.5)
        print(f"Detected muscle components: {muscle_inds}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        muscle_inds = []
    ica.exclude = list(set(blink_exclude + muscle_inds))
    print("Blink ICs:", blink_exclude)
    print("Muscle ICs:", muscle_inds)
    print("Final excluded:", ica.exclude)
    print("Blink ICs:", ica.exclude)
    
    # Diagnostics
    ica.plot_scores(eog_scores_fp1)  
    ica.plot_components()
    ica.plot_sources(raw_ica, show_scrollbars=False)

    print("\nReview ICA plots. Add any muscle component indices to ica.exclude if needed.")
    print("Press ENTER to apply ICA and continue, or type 'q' to quit.")
    if input().strip().lower() == 'q':
        raise SystemExit
    
    print("Channels used for ICA:")
    print([raw_ica.ch_names[p] for p in picks_ica])
    print("Final excluded ICs:", ica.exclude)

# ======================================================
# ERP DATASET
# ======================================================

    clean = raw.copy()
    clean.set_annotations(raw.annotations)
    clean.notch_filter(freqs=[60, 120], fir_design="firwin")
    clean.filter(l_freq=0.1, h_freq=30., fir_design="firwin")

    # Apply ICA learned from 1-Hz data
    ica.apply(clean)

    # Remove channels not needed for analysis
    clean.drop_channels(["Fp1", "Fp2"])
    clean.drop_channels(["VEOG"])

    # ======================================================
    # 1. BUILD EEG TRIAL TABLE
    # ======================================================
    events, event_id = mne.events_from_annotations(clean, event_id=event_id_map)
    print("unique codes in events:", np.unique(events[:, 2]))
    print("event_id returned:", event_id)
    print("start_codes we're looking for:", start_codes)
    print("total events:", len(events))

    sfreq = clean.info["sfreq"]
    eeg_trials = []

    for i, ev in enumerate(events):
        code = ev[2]

        if code not in start_codes:
            continue

        start_sample = ev[0]
        start_time   = start_sample / sfreq
        interaction  = interaction_map[code]
        expected_resp = code << 4  # 1→16, 2→32, 4→64, 8→128

        resp_sample = None
        resp_time   = None

        # Search forward, stop at next trial start to avoid cross-trial grab
        for ev2 in events[i+1:]:
            if ev2[2] in start_codes:
                break
            if ev2[2] == expected_resp:
                resp_sample = ev2[0]
                resp_time   = ev2[0] / sfreq
                break

        if resp_sample is None:
            print(f"  WARNING: no response for trial at sample {start_sample}, skipping")
            continue

        reveal_sample = resp_sample + int(0.170 * sfreq)  # 170 ms after response

        eeg_trials.append(dict(
            start_sample  = start_sample,
            resp_sample   = resp_sample,
            reveal_sample = reveal_sample,
            start_time    = start_time,
            resp_time     = resp_time,
            interaction   = interaction,
        ))

    print(f"Found {len(eeg_trials)} EEG trials.")
    print("sfreq:", sfreq)
    print("first reveal sample:", eeg_trials[0]["reveal_sample"])
    print("first resp sample:", eeg_trials[0]["resp_sample"])
    print("delta ms:", (eeg_trials[0]["reveal_sample"] - eeg_trials[0]["resp_sample"]) / sfreq * 1000)


    # ======================================================
    # 2. ALIGN WITH BEHAVIORAL CSV (content, not position)
    # ======================================================
    csv_seq = beh['interaction_type'].tolist()
    eeg_seq = [t['interaction'] for t in eeg_trials]

    i = 0
    matched_csv_idx = []
    skipped_csv_idx = []

    for interaction in eeg_seq:
        while i < len(csv_seq) and csv_seq[i] != interaction:
            skipped_csv_idx.append(i)
            i += 1
        if i == len(csv_seq):
            raise ValueError(
                f"{sub}: ran out of CSV rows while aligning — EEG trial sequence "
                f"doesn't fit within CSV order. Check for reordering or extra EEG trials."
            )
        matched_csv_idx.append(i)
        i += 1

    print(f"{sub}: matched {len(matched_csv_idx)} EEG trials to CSV rows.")
    print(f"{sub}: skipped {len(skipped_csv_idx)} CSV rows with no EEG match -> {skipped_csv_idx}")

    beh_aligned = beh.iloc[matched_csv_idx].reset_index(drop=True)

    if len(beh_aligned) != len(eeg_trials):
        raise ValueError(
            f"Mismatch after alignment: EEG has {len(eeg_trials)} trials but "
            f"matched {len(beh_aligned)} CSV rows"
        )

    metadata = beh_aligned.copy()
    metadata["interaction_eeg"] = [t["interaction"]  for t in eeg_trials]
    metadata["resp_time"]       = [t["resp_time"]    for t in eeg_trials]
    metadata["start_time"]      = [t["start_time"]   for t in eeg_trials]

    # sanity check: content match should now be trivially perfect
    assert (metadata["interaction_type"] == metadata["interaction_eeg"]).all(), \
        f"{sub}: interaction_type / interaction_eeg mismatch after alignment!"

    # ======================================================
    # 3. EVENT ARRAYS
    # ======================================================
    code_map_start = {"generalization": 1, "alliance": 2, "displacement": 4, "defense": 8}
    code_map_reveal = {"generalization": 16, "alliance": 32, "displacement": 64, "defense": 128}

    events_start = np.array([
        [t["start_sample"],  0, code_map_start[t["interaction"]]]
        for t in eeg_trials
    ])
    events_reveal = np.array([
        [t["reveal_sample"], 0, code_map_reveal[t["interaction"]]]
        for t in eeg_trials
    ])

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

    # ======================================================
    # 4a. START-LOCKED EPOCHS
    # ======================================================

    epochs_start = mne.Epochs(
        clean,
        events_start,
        event_id=event_id_start,
        tmin=-0.5,        
        tmax=3.2,         
        baseline=None,    
        preload=True,
        metadata=metadata
    )

    fixation_baseline = (-0.5, 0.0)

    epochs_start.metadata["baseline_start"] = fixation_baseline[0]
    epochs_start.metadata["baseline_end"] = fixation_baseline[1]

    # ======================================================
    # 4b. REVEAL-LOCKED EPOCHS
    # ======================================================
    epochs_reveal = mne.Epochs(
        clean, events_reveal, event_id=event_id_reveal,
        tmin=-0.3, tmax=0.8,
        baseline=(-0.06, -0.01),  
        preload=True,
        reject=None,
        metadata=metadata,
    )

    ar_reveal = AutoReject(random_state=97)
    ar_reveal.fit(epochs_reveal)
    epochs_reveal_clean, reject_log_reveal = ar_reveal.transform(epochs_reveal, return_log=True)

    ar_start = AutoReject(random_state=97)
    ar_start.fit(epochs_start)
    epochs_start_clean, reject_log_start = ar_start.transform(epochs_start, return_log=True)

    print(f"Reveal-locked: {len(epochs_reveal_clean)} / {len(epochs_reveal)} kept")
    print(f"Start-locked:  {len(epochs_start_clean)} / {len(epochs_start)} kept")

    # --- Inspect a sample of excluded epochs before committing ---
    n_bad_reveal = reject_log_reveal.bad_epochs.sum()
    print(f"Rejected {n_bad_reveal} reveal-locked epochs.")
    if n_bad_reveal > 0:
        reject_log_reveal.plot('horizontal')
        plt.show()
        bad_idx = np.where(reject_log_reveal.bad_epochs)[0]
        n_show = min(5, len(bad_idx))
        epochs_reveal[bad_idx[:n_show]].plot(
            n_epochs=n_show, picks=chs_P300 + chs_FRN,
            title=f"{sub}: sample of REJECTED reveal-locked epochs"
        )
        plt.show()

    print("\nReview rejected epochs. Press ENTER to accept and continue, or 'q' to quit.")
    if input().strip().lower() == 'q':
        raise SystemExit

    epochs_reveal = epochs_reveal_clean
    epochs_start  = epochs_start_clean
  
    # ======================================================
    # 5. PLOT ERPs PER INTERACTION TYPE
    # ======================================================
    import matplotlib as mpl
    
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Times New Roman',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    palette = {
        'generalization': '#4C72B0',
        'alliance':       '#55A868',
        'displacement':   '#C44E52',
        'defense':        '#8172B2',
    }
    colors = {"congruent": palette['generalization'], "incongruent": palette['alliance'], "no_reveal": palette['displacement']}

    for interaction in ["generalization", "alliance", "displacement", "defense"]:

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"{interaction.capitalize()} — Reveal-locked", fontsize=14)
        ax_frn = axes[0]
        ax_p3  = axes[1]
        ax_frn.set_title("FRN – Frontocentral (Fz, FC1, FC2, Cz)")
        ax_p3.set_title( "P300 – Centroparietal (Pz, CP1, CP2, Cz)")

        for trial_type in ["congruent", "incongruent", "no_reveal"]:
            cond = (
                f"interaction_type == '{interaction}' and "
                f"trial_type == '{trial_type}'"
            )
            sub_ep = epochs_reveal[cond]
            if len(sub_ep) == 0:
                print(f"  No epochs for {interaction}/{trial_type}, skipping")
                continue

            color = colors[trial_type]
            times = sub_ep.times

            # FRN (mean across ROI channels)
            data_FRN = sub_ep.average(picks=chs_FRN).data.mean(axis=0) * 1e6
            ax_frn.plot(times, data_FRN, color=color, label=f"{trial_type} (n={len(sub_ep)})", linewidth=1.5)

            # P300
            data_P3 = sub_ep.average(picks=chs_P300).data.mean(axis=0) * 1e6
            ax_p3.plot(times, data_P3, color=color, label=f"{trial_type} (n={len(sub_ep)})", linewidth=1.5)

        for ax in [ax_frn, ax_p3]:
            ax.axvline(0,    color='black', linestyle='--', linewidth=1,   label='reveal onset')
            ax.axhline(0,    color='black', linestyle='-',  linewidth=0.5)
            ax.set_xlabel("Time relative to reveal onset (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.legend(loc='upper right', fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim([epochs_reveal.tmin, epochs_reveal.tmax])

        plt.tight_layout()
        plt.savefig(subj_dir / f"{sub}_{interaction}_ERP.png", dpi=150)
        plt.show()

    # ======================================================
    # ALPHA ANALYSIS: TRIAL-START TO RESPONSE
    # ======================================================

    alpha_raws = []
    sfreq = clean.info['sfreq']

    for i, ev in enumerate(events):
        code = ev[2]
        if code not in start_codes:
            continue

        start_time = ev[0] / sfreq
        expected_resp = code << 4

        resp_time = None
        for ev2 in events[i+1:]:
            if ev2[2] in start_codes:
                break
            if ev2[2] == expected_resp:
                resp_time = ev2[0] / sfreq
                break

        if resp_time is None:
            continue

        seg = clean.copy().crop(tmin=start_time, tmax=resp_time)
        alpha_raws.append(seg)

    # ======================================================
    # TIME-FREQUENCY ANALYSIS (8–12 Hz ALPHA)
    # ======================================================

    freqs = np.linspace(8, 12, 5)
    n_cycles = freqs / 2

    alpha_values = []

    for seg in alpha_raws:
        data = seg.get_data(picks=chs_alpha)  # (n_chs, n_times)

        power = mne.time_frequency.tfr_array_morlet(
            data[np.newaxis, :, :],
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power"
        )[0]  # (n_chs, n_freqs, n_times)

        alpha_values.append(power.mean())  # one scalar per trial

    metadata["alpha_power"] = alpha_values

    # ----------------- SAVE SINGLE-SUBJECT OUTPUT -----------------
    # cleaned epochs + metadata

    print("epochs_start metadata:", epochs_start.metadata.columns.tolist())
    print("epochs_reveal metadata:", epochs_reveal.metadata.columns.tolist())
    print("n epochs_start:", len(epochs_start))
    print("n epochs_reveal:", len(epochs_reveal))
    print("Do you want to save the epochs? (y/n)")
    response = input().lower()
    if response == 'y':
        epochs_start.save(subj_dir / f"{sub}_start-epo.fif", overwrite=True)
        epochs_reveal.save(subj_dir / f"{sub}_reveal-epo.fif", overwrite=True)  # baseline already applied before shift, clear stored interval
    else:
        print("Skipping saving epochs. Preprocessing complete for this subject.")

process_subject(sub, subj_dir, raw_file)