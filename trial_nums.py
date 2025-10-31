# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:41:35 2024

@author: Julian Bandhan
"""
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr

#Read in sheet with trial counts etc
#ptp = pd.read_excel(r'Y:\projects\OPM-Analysis\OPM_gamma\demographics\opm_gamma_statvScrap.xlsx.xlsx', sheet_name= 'Sheet1')
ptp = pd.read_excel(r'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\demographics\opm_rest_babyeinstein_statv6_1.xlsx', sheet_name= 'Sheet1')
"""
My trials are columned under 'trials', and original trials are columned under 'original_trials'
Change names, e.g. ptp['trials'], to match whatever you names yours
"""
#Functions for pulling number per age_bin/age
trial_nums = lambda x: ptp.loc[(ptp['age_bin']==x) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'trials']]
trial_nums_orig = lambda x: ptp.loc[(ptp['age_bin']==x) & (ptp['original_trials'] > 30)& (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'original_trials']]

HM_nums = lambda x: ptp.loc[(ptp['age_bin']==x) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'HM']]



# %% Plots Retained Trial vs age 

fig = plt.Figure()
plt.xlabel('Age')
plt.ylabel('Number of Trials - QC')

for i in range(1,6):
    avg_trials = np.sum(trial_nums(i).trials)/len(trial_nums(i))
    plt.scatter(trial_nums(i).age, trial_nums(i).trials, label = f'{i}YR: {round(avg_trials)} /sub')
    

x = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'age']
y = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 40)& (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'trials']
# x = ptp.age
# y = ptp.trials
corr_coeff, p_value = pearsonr(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label = f'p-value = {p_value:.2e}')
#plt.scatter(x,y)
plt.legend()

# %% Plots  HM vs age 

fig = plt.Figure()
plt.xlabel('Age')
plt.ylabel('HM - QC')

for i in range(1,6):
    avg_HM = np.sum(HM_nums(i).HM)/len(HM_nums(i))
    plt.scatter(HM_nums(i).age, HM_nums(i).HM, label = f'{i}YR: {round(avg_HM)} /sub')
    

x = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'age']
y = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'HM']

valid = x.notna() & y.notna()
corr_coeff, p_value = pearsonr(x[valid], y[valid])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label = f'p-value = {p_value:.2e}')
#plt.scatter(x,y)
plt.legend()

# %% Plots Original trials vs Age

fig = plt.Figure()
plt.xlabel('Age')
plt.ylabel('Number of Trials')

for i in range(1,6):
    avg_trials = np.sum(trial_nums_orig(i).original_trials)/len(trial_nums_orig(i))
    plt.scatter(trial_nums_orig(i).age, trial_nums_orig(i).original_trials, label = f'{i}YR: {round(avg_trials)} /sub')
    

x = ptp.loc[(ptp['age_bin']!=0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels']<20), 'age']
y = ptp.loc[(ptp['age_bin']!=0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 )& (ptp['num_bad_channels']<20), 'original_trials']

corr_coeff, p_value = pearsonr(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label = f'p-value = {p_value:.2e}')

plt.legend()

# %% Plots Both original and retained trials stacked 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr

#ptp = pd.read_excel(r'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\demographics\opm_rest_babyeinstein_statvScrap.xlsx', sheet_name='Sheet1')

trial_nums = lambda x: ptp.loc[(ptp['age_bin'] == x) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'trials']]
trial_nums_orig = lambda x: ptp.loc[(ptp['age_bin'] == x) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'original_trials']]

# Create a figure with two vertically stacked subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Plot 1: Trials
axs[0].set_title('Trials vs. Age')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Number of Trials')

for i in range(1, 6):
    avg_trials = np.sum(trial_nums(i).trials) / len(trial_nums(i))
    axs[0].scatter(trial_nums(i).age, trial_nums(i).trials, label=f'{i}YR: {round(avg_trials)} /sub')

x = ptp.loc[(ptp['age_bin'] != 0) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 21), 'age']
y = ptp.loc[(ptp['age_bin'] != 0) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 21), 'trials']

corr_coeff, p_value = pearsonr(x, y)
axs[0].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'p-value = {p_value:.2e}')
axs[0].legend()

# Plot 2: Original Trials
axs[1].set_title('Original Trials vs. Age')
axs[1].set_xlabel('Age')
axs[1].set_ylabel('Number of Original Trials')

for i in range(1, 6):
    avg_trials = np.sum(trial_nums_orig(i).original_trials) / len(trial_nums_orig(i))
    axs[1].scatter(trial_nums_orig(i).age, trial_nums_orig(i).original_trials, label=f'{i}YR: {round(avg_trials)} /sub')

x_orig = ptp.loc[(ptp['age_bin'] != 0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 20), 'age']
y_orig = ptp.loc[(ptp['age_bin'] != 0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 20), 'original_trials']

corr_coeff_orig, p_value_orig = pearsonr(x_orig, y_orig)
axs[1].plot(np.unique(x_orig), np.poly1d(np.polyfit(x_orig, y_orig, 1))(np.unique(x_orig)), label=f'p-value = {p_value_orig:.2e}')
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

#plt.savefig(fr"Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\analysis\stats\trial_numsv2.png")

# %% NEWER code
# -*- coding: utf-8 -*-
"""
All plots + stats use ONLY MixedLM:
  Trials ~ Age + (1|Subject)
  HM ~ Age + (1|Subject)
  OriginalTrials ~ Age + (1|Subject)
"""

%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# --- Load data ---
ptp = pd.read_excel(
    r'Y:/projects/OPM-Analysis/OPM_rest_babyeinsteins/demographics/opm_rest_babyeinstein_v7stat.xlsx',
    sheet_name='Sheet1'
).copy()

# Ensure numeric
for col in ['age', 'trials', 'original_trials', 'num_bad_channels', 'HM', 'age_bin']:
    if col in ptp.columns:
        ptp[col] = pd.to_numeric(ptp[col], errors='coerce')

# Helper: per-bin subsets (no QC thresholds)
trial_nums      = lambda b: ptp.loc[(ptp['age_bin'] == b), ['age', 'trials']].dropna()
trial_nums_orig = lambda b: ptp.loc[(ptp['age_bin'] == b), ['age', 'original_trials']].dropna()
HM_nums         = lambda b: ptp.loc[(ptp['age_bin'] == b), ['age', 'HM']].dropna()

def fit_lme(ycol, df, xcol='age', group='subject', reml=True):
    """
    Fit MixedLM: ycol ~ xcol + (1|group), returning (result, X, y, mask).
    Drops rows with NaNs in ycol, xcol, or group.
    """
    use = df[[ycol, xcol, group]].dropna().copy()
    if use.empty or use[xcol].nunique() < 2:
        return None, None
    X = sm.add_constant(use[xcol].to_numpy())
    y = use[ycol].to_numpy()
    g = use[group].astype(str).to_numpy()
    mdl = MixedLM(endog=y, exog=X, groups=g)
    res = mdl.fit(reml=reml)
    return res, use

def add_lme_line(ax, res, xvals, label_prefix=''):
    """
    Plot population-level fit line using fixed effects only.
    Handles both pandas Series and plain numpy arrays from res.params.
    """
    # If it's a pandas Series, get by name, else by position
    if hasattr(res.params, 'index'):
        const = res.params['const']
        slope_name = [k for k in res.params.index if k != 'const'][0]
        beta = res.params[slope_name]
        p = res.pvalues[slope_name]
        se = res.bse[slope_name]
    else:
        const = res.params[0]
        beta = res.params[1]
        p = res.pvalues[1]
        se = res.bse[1]

    yfit = const + beta * xvals
    ax.plot(
        xvals, yfit, linewidth=2,
        label=f"{label_prefix}coef={beta:.3f}, p={p:.2e}"
    )


def print_lme_summary(title, res):
    print(f"\n===== {title} =====")
    print(res.summary())
    print("Fixed effects:\n", res.params)
    print("P-values:\n", res.pvalues)

# =========================
# 1) Trials ~ Age + (1|Subject)
# =========================
plt.figure(figsize=(8,5))
plt.title('Trials vs Age (MixedLM)')
plt.xlabel('Age (years)')
plt.ylabel('Trials')

for i in range(1, 6):
    dat = trial_nums(i)
    if len(dat) == 0: 
        continue
    avg_trials = dat['trials'].mean()
    plt.scatter(dat['age'], dat['trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials)} /sub')

res_trials, df_trials = fit_lme('trials', ptp, xcol='age', group='subject', reml=True)
if res_trials is not None:
    xgrid = np.linspace(df_trials['age'].min(), df_trials['age'].max(), 200)
    add_lme_line(plt.gca(), res_trials, xgrid, label_prefix='')
    print_lme_summary("LME: Trials ~ Age + (1|Subject)", res_trials)

plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 2) HM ~ Age + (1|Subject)
# =========================
plt.figure(figsize=(8,5))
plt.title('Head Motion vs Age (MixedLM)')
plt.xlabel('Age (years)')
plt.ylabel('HM')

for i in range(1, 6):
    dat = HM_nums(i)
    if len(dat) == 0:
        continue
    plt.scatter(dat['age'], dat['HM'], s=18, alpha=0.8, label=f'{i}YR')

res_hm, df_hm = fit_lme('HM', ptp, xcol='age', group='subject', reml=True)
if res_hm is not None:
    xgrid = np.linspace(df_hm['age'].min(), df_hm['age'].max(), 200)
    add_lme_line(plt.gca(), res_hm, xgrid, label_prefix='')
    print_lme_summary("LME: HM ~ Age + (1|Subject)", res_hm)

plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 3) OriginalTrials ~ Age + (1|Subject)
# =========================
plt.figure(figsize=(8,5))
plt.title('Original Trials vs Age (MixedLM)')
plt.xlabel('Age (years)')
plt.ylabel('Original Trials')

for i in range(1, 6):
    dat = trial_nums_orig(i)
    if len(dat) == 0:
        continue
    avg_trials_o = dat['original_trials'].mean()
    plt.scatter(dat['age'], dat['original_trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials_o)} /sub')

res_orig, df_orig = fit_lme('original_trials', ptp, xcol='age', group='subject', reml=True)
if res_orig is not None:
    xgrid = np.linspace(df_orig['age'].min(), df_orig['age'].max(), 200)
    add_lme_line(plt.gca(), res_orig, xgrid, label_prefix='')
    print_lme_summary("LME: OriginalTrials ~ Age + (1|Subject)", res_orig)

plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 4) Stacked figure using MixedLM lines
# =========================
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# (a) Trials
axs[0].set_title('Trials vs Age (MixedLM)')
axs[0].set_ylabel('Trials')
for i in range(1, 6):
    dat = trial_nums(i)
    if len(dat) == 0:
        continue
    avg_trials = dat['trials'].mean()
    axs[0].scatter(dat['age'], dat['trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials)} /sub')
if res_trials is not None:
    xgrid = np.linspace(df_trials['age'].min(), df_trials['age'].max(), 200)
    add_lme_line(axs[0], res_trials, xgrid, label_prefix='')
axs[0].legend()

# (b) Original Trials
axs[1].set_title('Original Trials vs Age (MixedLM)')
axs[1].set_xlabel('Age (years)')
axs[1].set_ylabel('Original Trials')
for i in range(1, 6):
    dat = trial_nums_orig(i)
    if len(dat) == 0:
        continue
    avg_trials_o = dat['original_trials'].mean()
    axs[1].scatter(dat['age'], dat['original_trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials_o)} /sub')
if res_orig is not None:
    xgrid = np.linspace(df_orig['age'].min(), df_orig['age'].max(), 200)
    add_lme_line(axs[1], res_orig, xgrid, label_prefix='')
axs[1].legend()

plt.tight_layout()
plt.show()


