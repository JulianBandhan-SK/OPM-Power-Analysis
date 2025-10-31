# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:03:39 2025
@author: Julian Bandhan
"""

import importlib
from functions.common_imports import np, plt, random, stats, scipy, pg, sns, pd, loadmat, Axes3D, image, Line2D, MixedLM, sm
from functions.psd_plots import plot_stuff_vs_stuff, plot_stuff_vs_stuffv2, extract_band_psd
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2, make_4d_atlas_nifti
from functions.lme_stuff import lme_stuff, run_lme
import warnings
from statsmodels.stats.multitest import multipletests
import os

# --------------------------------------------------------------------------
# --- Configuration ---
use_std = True        # <-- set to True for mean ± SD, False for mean ± SE
# --------------------------------------------------------------------------

# --- Load data ---
subject_data = pd.read_excel(
    r'Y:/projects/OPM-Analysis/OPM_rest_babyeinsteins/demographics/opm_rest_babyeinstein_v7stat.xlsx'
)
df = subject_data.copy()

# Ensure key numeric columns are numeric (avoid 'int' + 'str' issues)
for col in ['age', 'trials', 'num_bad_channels', 'HM']:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# A de-duplicated table for per-subject attributes (e.g., sex)
subj_info = df.drop_duplicates('subject')

# --- 1. Total counts ---
total_scans = len(df)                           # every row is one scan
total_unique_subjects = subj_info['subject'].nunique()
num_males = subj_info.loc[subj_info['sex'] == 'M', 'subject'].nunique()
num_females = subj_info.loc[subj_info['sex'] == 'F', 'subject'].nunique()

# --- Helper to compute spread: SD or SE ---
def spread(series: pd.Series) -> float:
    """
    Return SD or SE depending on use_std flag.
    """
    s = series.std(ddof=1)
    if use_std:
        return s
    else:
        n = series.count()
        return s / np.sqrt(n)

# --- 2. Averages with ± chosen spread ---
avg_age = df['age'].mean()
spread_age = spread(df['age'])

avg_trials = df['trials'].mean()
spread_trials = spread(df['trials'])

avg_bad_channels = df['num_bad_channels'].mean()
spread_bad_channels = spread(df['num_bad_channels'])

avg_head_motion = df['HM'].mean()
spread_head_motion = spread(df['HM'])

# --- 3. Average trials per age group ---
if 'age_bin' in df and df['age_bin'].notna().any():
    # apply the same SD/SE logic per age_bin
    def agg_func(x):
        return pd.Series({
            'avg_trials': x.mean(),
            'spread_trials': spread(x),
            'count': x.count()
        })
    avg_trials_by_age_bin = (
        df.groupby('age_bin', dropna=True)['trials']
          .apply(agg_func)
          .sort_index()
    )
else:
    avg_trials_by_age_bin = pd.DataFrame(columns=['avg_trials', 'spread_trials', 'count'])

# --- 4. Average trials per participant (across their scans) ---
trials_by_subject = df.groupby('subject')['trials'].mean()
avg_trials_per_ptp = trials_by_subject.mean()
spread_trials_per_ptp = spread(trials_by_subject)

# --- 5. Pack results for display ---
label = "std" if use_std else "SE"
summary = {
    'total_scans': total_scans,
    'unique_subjects': total_unique_subjects,
    'unique_males': num_males,
    'unique_females': num_females,
    f'avg_age_yrs (± {label})': f"{avg_age:.2f} ± {spread_age:.2f}",
    f'avg_trials_per_scan (± {label})': f"{avg_trials:.2f} ± {spread_trials:.2f}",
    f'avg_trials_per_subject (± {label})': f"{avg_trials_per_ptp:.2f} ± {spread_trials_per_ptp:.2f}",
    f'avg_bad_channels (± {label})': f"{avg_bad_channels:.2f} ± {spread_bad_channels:.2f}",
    f'avg_head_motion (± {label})': f"{avg_head_motion:.2f} ± {spread_head_motion:.2f}",
}

print(pd.Series(summary))
print("\nAverage trials per age group:\n", avg_trials_by_age_bin)

# --------------------------------------------------------------------------
# --- Sessions per subject summary ---
sessions_per_subject = df.groupby('subject')['session'].nunique()
session_summary = sessions_per_subject.value_counts().sort_index()
print("\nSessions per subject:\n", session_summary)
# --------------------------------------------------------------------------
