import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pingouin as pg
from scipy import stats
from statsmodels.formula.api import mixedlm
from itertools import combinations

# Configuration
folder_path = '/Users/antoniagergen/Desktop/TRIA/TRIA_results'
all_files = glob.glob(os.path.join(folder_path, 'subj*', 'subj*.csv'))
print(f"Loaded {len(all_files)} subject(s): {[os.path.basename(f) for f in all_files]}")

dfs = []
for f in all_files:
    d = pd.read_csv(f)
    d['subject_id'] = os.path.basename(f).split('.')[0]
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

# Outlier exclusion
total_before = len(df)
outliers = df[df['reaction_time'] > 10]
print(f"\nExcluded {len(outliers)} trial(s) with RT > 10s:")
if len(outliers) > 0:
    print(outliers[['trial_number', 'interaction_type', 'trial_type', 'reaction_time']])
df = df[df['reaction_time'] <= 10].reset_index(drop=True)
print(f"Remaining trials: {len(df)} / {total_before}")

# Basic means
mean_rt_by_type = df.groupby('interaction_type')['reaction_time'].mean().sort_values()
total_mean_accuracy = df['correct'].mean()
mean_accuracy_by_type = df.groupby('interaction_type')['correct'].mean().sort_values()

print("Mean reaction time by interaction type:")
print(mean_rt_by_type)
print("\nOverall mean accuracy:")
print(total_mean_accuracy)
print("\nMean accuracy by interaction type:")
print(mean_accuracy_by_type)

if 'trial_number' not in df.columns:
    if 'subject_id' in df.columns:
        df['trial_number'] = df.groupby('subject_id').cumcount() + 1
    else:
        df['trial_number'] = df.groupby(0).cumcount() + 1

# Response bias
resp_counts = df['response'].value_counts(normalize=True)
print("\nResponse proportions (overall):")
print(resp_counts)

resp_by_trial_type = df.pivot_table(
    index='trial_type', columns='response', values='correct', aggfunc='count'
)
resp_by_trial_type = resp_by_trial_type.div(resp_by_trial_type.sum(axis=1), axis=0)
print("\nResponse proportions by trial_type:")
print(resp_by_trial_type)

# Sequence effects (descriptive) 
if 'subject_id' in df.columns:
    df = df.sort_values(['subject_id', 'trial_number'])
    group_keys = ['subject_id']
else:
    df = df.sort_values('trial_number')
    group_keys = None

if group_keys:
    df['prev_trial_type'] = df.groupby(group_keys)['trial_type'].shift(1)
    df['prev_interaction_type'] = df.groupby(group_keys)['interaction_type'].shift(1)
else:
    df['prev_trial_type'] = df['trial_type'].shift(1)
    df['prev_interaction_type'] = df['interaction_type'].shift(1)

seq_rt_ti = df.dropna(subset=['prev_trial_type']).groupby(
    ['prev_trial_type', 'interaction_type'])['reaction_time'].mean()
seq_acc_ti = df.dropna(subset=['prev_trial_type']).groupby(
    ['prev_trial_type', 'interaction_type'])['correct'].mean()
print("\nMean RT by previous trial_type and current interaction_type:")
print(seq_rt_ti)
print("\nMean accuracy by previous trial_type and current interaction_type:")
print(seq_acc_ti)

seq_rt_ii = df.dropna(subset=['prev_interaction_type']).groupby(
    ['prev_interaction_type', 'interaction_type'])['reaction_time'].mean()
seq_acc_ii = df.dropna(subset=['prev_interaction_type']).groupby(
    ['prev_interaction_type', 'interaction_type'])['correct'].mean()
print("\nMean RT by previous and current interaction_type:")
print(seq_rt_ii)
print("\nMean accuracy by previous and current interaction_type:")
print(seq_acc_ii)

prev_trial_accuracy = (
    df.dropna(subset=['prev_trial_type'])
      .groupby('prev_trial_type')['correct']
      .mean()
      .sort_values()
)

# Accuracy and trial number correlation
for subj, grp in df.groupby('subject_id'):
    r, p = stats.pearsonr(grp['trial_number'], grp['correct'])
    print(f"{subj}: r={r:.3f}, p={p:.3f}")

# Check for missing cells in the design due to ANOVA fail
def report_missing_cells(data, within_cols, subject_col='subject_id'):
    all_subjects = data[subject_col].unique()
    cell_cols = within_cols
    counts = data.groupby([subject_col] + cell_cols).size()
    full_index = pd.MultiIndex.from_product(
        [all_subjects] + [data[c].dropna().unique() for c in cell_cols],
        names=[subject_col] + cell_cols
    )
    counts = counts.reindex(full_index, fill_value=0)
    missing = counts[counts == 0]
    n_expected_cells = len(full_index) / len(all_subjects)
    if len(missing) == 0:
        print(f"[{cell_cols}] All {len(all_subjects)} subjects have all "
              f"{int(n_expected_cells)} cells populated. Design is balanced.")
    else:
        incomplete_subjects = missing.index.get_level_values(subject_col).unique()
        print(f"[{cell_cols}] {len(incomplete_subjects)} / {len(all_subjects)} "
              f"subject(s) missing at least one cell:")
        for s in incomplete_subjects:
            missing_cells = missing.loc[s].index.tolist()
            print(f"  {s}: missing {missing_cells}")
        print("  -> pg.rm_anova will silently drop these subjects for this "
              "design, which is very likely why F/p came back as NaN.")
    return missing

print("\n-- CELL COMPLETENESS CHECK --")
seq_df = df.dropna(subset=['prev_interaction_type']).copy()
report_missing_cells(seq_df, ['prev_interaction_type', 'interaction_type'])
report_missing_cells(df, ['interaction_type'])

# ---------- Style ----------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 16,
    'axes.titleweight': 'bold',
})

palette = {
    'generalization': "#686767",
    'alliance':       '#686767',
    'displacement':   '#686767',
    'defense':        '#686767',
}
colors = [palette[k] for k in mean_rt_by_type.index]
colors_acc = [palette[k] for k in mean_accuracy_by_type.index]

def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return None


#Function for adding significance brackets to bar plots
def add_significance_brackets(ax, order, pairwise_df, heights, a_col='A', b_col='B',
                               p_col='p_corr', bar_width=0.55, step_frac=0.12,
                               line_offset_frac=0.14):
    pos = {label: i for i, label in enumerate(order)}
    y_max = max(heights.values())
    step = y_max * step_frac
    line_offset = y_max * line_offset_frac
    level = 0
    for _, row in pairwise_df.iterrows():
        a, b, p = row[a_col], row[b_col], row[p_col]
        if a not in pos or b not in pos or pd.isna(p):
            continue
        stars = p_to_stars(p)
        if stars is None:
            continue
        x1, x2 = pos[a], pos[b]
        y = y_max + line_offset + step * level
        tick = step * 0.2
        ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
                lw=1.2, color='black')
        ax.text((x1 + x2) / 2, y + tick, stars, ha='center', va='bottom', fontsize=13)
        level += 1
    if level > 0:
        ax.set_ylim(top=y_max + line_offset + step * (level + 1))


# STATISTICS
print("\n========== TABLE 1: Main effects of interaction type (subject-level) ==========")
print(f"{'Interaction':<16} {'M acc':>6} {'SD acc':>7} {'t':>6} {'df':>4} {'p':>6} {'M RT':>7} {'SD RT':>7}")
print("-" * 66)

subj_means = df.groupby(['subject_id', 'interaction_type']).agg(
    acc=('correct', 'mean'),
    rt=('reaction_time', 'mean')
).reset_index()

for itype in ['generalization', 'alliance', 'displacement', 'defense']:
    sub = subj_means[subj_means['interaction_type'] == itype]
    acc = sub['acc']
    t, p = stats.ttest_1samp(acc, 0.5)
    df_ttest = len(acc) - 1
    m_rt = sub['rt'].mean()
    sd_rt = sub['rt'].std()
    p_str = f"{p:.3f}" if p >= 0.001 else "<.001"
    print(f"{itype:<16} {acc.mean():>6.2f} {acc.std():>7.2f} {t:>6.2f} {df_ttest:>4} {p_str:>6} {m_rt:>7.2f} {sd_rt:>7.2f}")

# --- Repeated-measures ANOVA: interaction_type main effect (4 cells, need to be balanced) ---
try:
    aov_rt = pg.rm_anova(dv='reaction_time', within='interaction_type',
                          subject='subject_id', data=df, detailed=True,
                          correction=True)
    print("\n=== Repeated-measures ANOVA: RT by interaction_type ===")
    print(aov_rt)
except Exception:
    import traceback
    print("\n!!! rm_anova (RT, interaction_type) failed !!!")
    traceback.print_exc()

try:
    aov_acc = pg.rm_anova(dv='correct', within='interaction_type',
                           subject='subject_id', data=df, detailed=True,
                           correction=True)
    print("\n=== Repeated-measures ANOVA: Accuracy by interaction_type ===")
    print(aov_acc)
except Exception:
    import traceback
    print("\n!!! rm_anova (accuracy, interaction_type) failed !!!")
    traceback.print_exc()

# --- Pairwise post-hoc comparisons (renamed API, Bonferroni-corrected) ---
pairwise_rt = pg.pairwise_tests(dv='reaction_time', within='interaction_type',
                                 subject='subject_id', data=df, padjust='bonf')
print("\n=== Pairwise RT comparisons ===")
print(pairwise_rt)

pairwise_acc = pg.pairwise_tests(dv='correct', within='interaction_type',
                                  subject='subject_id', data=df, padjust='bonf')
print("\n=== Pairwise accuracy comparisons ===")
print(pairwise_acc)

subj_acc = df.groupby(['subject_id', 'interaction_type'])['correct'].mean().unstack()
print(subj_acc.round(2))

print("\nSD of accuracy across subjects, per interaction type:")
print(subj_acc.std().round(3))

print("\nSubjects near ceiling (>85%) or at/below chance (<=50%) in any condition:")
print(subj_acc[(subj_acc.max(axis=1) > 0.85) | (subj_acc.min(axis=1) <= 0.5)])

subject_accuracy_again = (
    df.groupby("subject_id")["correct"]
      .mean()
      .sort_values()
)

subject_standard_deviation = (
    df.groupby("subject_id")["correct"]
      .std()
      .sort_values()
)

print(subject_accuracy_again)
print(subject_standard_deviation)

#Sequencec effects: linear mixed models (LMM) for RT and accuracy
print("\n========== Sequence effects: linear mixed models ==========")

lmm_data = df.dropna(subset=['prev_interaction_type']).copy()

rt_model = mixedlm(
    "reaction_time ~ C(prev_interaction_type) * C(interaction_type)",
    data=lmm_data, groups=lmm_data["subject_id"]
)
rt_fit = rt_model.fit()
print("\n=== Mixed model: RT ~ prev_interaction_type * interaction_type ===")
print(rt_fit.summary())

acc_model = mixedlm(
    "correct ~ C(prev_interaction_type) * C(interaction_type)",
    data=lmm_data, groups=lmm_data["subject_id"]
)
acc_fit = acc_model.fit()
print("\n=== Mixed model: accuracy ~ prev_interaction_type * interaction_type ===")
print(acc_fit.summary())

# Plotting
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)
ax_rt = fig.add_subplot(gs[0, 0])
ax_acc_type = fig.add_subplot(gs[0, 1])
ax_prev = fig.add_subplot(gs[1, :])

# Accuracy by previous trial type
ax = ax_prev
bar_colors = ['#686767', '#686767', '#686767']
ax.bar(prev_trial_accuracy.index, prev_trial_accuracy.values,
       color=bar_colors, width=0.55, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Mean accuracy', fontsize=11)
ax.set_title('Accuracy by previous trial type', fontsize=12)
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.6, label='chance')
ax.legend(fontsize=9, frameon=False)
for bar, val in zip(ax.patches, prev_trial_accuracy.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=10, color='#333333')

# RT by interaction type, with significance brackets from pairwise_rt
ax = ax_rt
order = list(mean_rt_by_type.index)
bars = ax.bar(order, mean_rt_by_type.values,
              color=colors, width=0.55, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Mean reaction time (s)', fontsize=11)
ax.set_title('RT by interaction type', fontsize=12)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=25, ha='right')
for bar, val in zip(bars, mean_rt_by_type.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=10, color='#333333')
add_significance_brackets(ax, order, pairwise_rt,
                           heights=mean_rt_by_type.to_dict())

# Accuracy by interaction type, with significance brackets from pairwise_acc
ax = ax_acc_type
order_acc = list(mean_accuracy_by_type.index)
bars = ax.bar(order_acc, mean_accuracy_by_type.values,
              color=colors_acc, width=0.55, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Mean accuracy', fontsize=11)
ax.set_title('Accuracy by interaction type', fontsize=12)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.6, label='chance')
ax.legend(fontsize=9, frameon=False)
ax.set_xticks(range(len(order_acc)))
ax.set_xticklabels(order_acc, rotation=25, ha='right')
for bar, val in zip(bars, mean_accuracy_by_type.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=10, color='#333333')
add_significance_brackets(ax, order_acc, pairwise_acc,
                           heights=mean_accuracy_by_type.to_dict())

plt.tight_layout()
plt.savefig(os.path.join(folder_path, 'behavioural_summary.png'), dpi=200, bbox_inches='tight')
plt.show()

#Strategy deviance over trials 
subjects = df['subject_id'].unique()
n_subj = len(subjects)
fig, axes = plt.subplots(n_subj, 1, figsize=(12, 2.2 * n_subj), sharex=True)
if n_subj == 1:
    axes = [axes]
window = 15
for ax, subj in zip(axes, subjects):
    sub = df[df['subject_id'] == subj].sort_values('trial_number')
    ax.scatter(sub['trial_number'], sub['correct'], s=10, alpha=0.35, color='#4C72B0')
    rolling = sub['correct'].rolling(window, center=True, min_periods=5).mean()
    ax.plot(sub['trial_number'], rolling, color='#C44E52', linewidth=1.8)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0\n(deviant)', '0.5', '1\n(theory)'])
    ax.set_ylabel('Correct', fontsize=9)
    ax.set_title(subj, fontsize=10, loc='left')
axes[-1].set_xlabel('Trial number')
fig.suptitle('Response consistency with theory prediction across the session', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, 'strategy_deviance_by_subject.png'), dpi=200, bbox_inches='tight')
plt.show()