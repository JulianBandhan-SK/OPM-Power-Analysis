# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:09:55 2025

@author: Julian Bandhan
"""
import importlib
from functions.common_imports import np, plt, random, stats, scipy, pg, sns, pd, loadmat, Axes3D, image, Line2D, MixedLM, sm
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2, make_4d_atlas_nifti
import warnings
from statsmodels.stats.multitest import multipletests
import os


#Set atlas file (4D or 3D)
atlasimg = fr"Y:\projects\OPM\opm_pipeline_templates\Adult\MEGatlas_38reg\MEG_atlas_38_regions_4D.nii.gz"
atlasimg3d = fr"Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\analysis\atlas\MEG_atlas_38_regions_3D.nii.gz"

#%matplotlib qt

plt.rcParams['font.family'] = 'Arial'  # Set global font to Arial

#Set a directory to save figs and stuff
base_dir = fr'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\analysis\powerspec'
os.makedirs(base_dir, exist_ok=True)


# %% 1 - Load in the Data

#Load in PSD data
prime_data = loadmat(r'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\code\ICA_163ptp_powers_meg38_zscored.mat')['data']

subject_data = pd.read_excel(r'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\demographics\opm_rest_babyeinstein_v7stat.xlsx')



# If it's a structured array (1x1 struct), extract it properly
if isinstance(prime_data, np.ndarray):
    prime_data = prime_data[0, 0]  # Extract first struct if needed
    
    
#drop scans if needed
drop_idx =  [28,30,68,124]
subject_data = subject_data.drop(drop_idx)


# Create subject/session/run list - This is dependent on how Matlab outputs the PSD's
subjects = prime_data['subject'] 
subjects = np.delete(subjects,drop_idx,axis = 0)

if isinstance(subjects, np.ndarray):
    if subjects.dtype.type is np.object_:  # If it's a nested cell array
        subjects = [str(s[0]) for s in subjects]  # Extract elements
        
# Clean and combine directly from subject_data
subjects = [s.strip("[]'") for s in subjects]

# Clean and combine directly from subject_data
subject_visitId = [
    fr"{subj.strip('[]\'')}-sess-{int(sess):02}-run-{int(run):02}"
    for subj, sess, run in zip(subject_data[fr'subject'], subject_data[fr'session'], subject_data[fr'run'])
]

#Sex array
prime_sex = subject_data['sex']


#Frequency array
prime_freq = prime_data['freq'][:,0]


# All psd data
prime_psd = prime_data['psd']
prime_psd = np.delete(prime_psd, drop_idx, axis = 0)

ages = prime_data['age'][:,0]
ages  = np.delete(ages, drop_idx, axis = 0)

#Age bins
age_bin = np.array(ages)
age_bin = np.where(age_bin <= 1, 1, np.floor(age_bin)) #Rounds up ptp below 1yr to 1 for binning

hm = subject_data['HM']



# Load in the region names
reg_name = pd.read_csv(r'Y:\projects\OPM\opm_pipeline_templates\Adult\MEGatlas_38reg\meg38_labels.txt', header=None, delimiter='\t')[0]
reg_name_full = pd.read_csv(r'Y:\projects\OPM\opm_pipeline_templates\Adult\MEGatlas_38reg\meg38_labels_full.txt', header=None, delimiter='\t')[0]

#some basic nums
num_sub = len(subjects)
num_reg = len(reg_name)

# %% 2 - Fooofing stuff first
################################################
# We're gonna fooof 3-30Hz, mostly to keep the gamma bands unfooofed
###############################################

#Importing here just causse
import importlib
import functions.fooof_func as fooof_func

importlib.reload(fooof_func)

from functions.fooof_func import run_specparam_batch, SpectralModel,run_fooof_batch
fooof_settings = {
    'freq_range': [3, 40],  # or change to [4, 40] etc.
    'peak_width_limits': [2, 4],
    'max_n_peaks': np.inf,
    'peak_threshold': 0,
    'min_peak_height': 0,
    'aperiodic_mode': 'fixed',
    'space': 'linear'  
}

fooof_results = run_specparam_batch(prime_psd, prime_freq, num_sub, num_reg, fooof_settings)

fooof_periodic_psd = fooof_results['periodic']
fooof_aperiodic_psd = fooof_results['aperiodic']
fooof_spectrum_psd = fooof_results['spectrum']
fooof_params = fooof_results['params']  #NOTE!!!: Using 'knee', the index here are: 0 = Offset, 1 = knee, 2 = Slope
fooof_params[:,:,0] = 10**fooof_params[:,:,0]
fooof_rmse = fooof_results['r_squared']
fooof_models = fooof_results['models'] 
n_bins = fooof_results['n_bins']
freq_bins_low = fooof_results['freqs_used']  # freq_bins_low = [3, 4, 5, ..., 30] if freq_range=[3,30]

offset_mean = np.mean(fooof_params[:,:,0],axis = 1)
slope_mean = np.mean(fooof_params[:,:,-1],axis = 1)

    
# %% 3 - Properly Framing the Data 


periodic_adj = fooof_periodic_psd[:, 0:46, :38]  #Slices the periodic psd data from 3 to 30Hz

# set Frequency 
freq_bins = prime_freq[6:138]
num_freqs= len(freq_bins)

#Re-Cutting the fooof freq bins for 3 to 30Hz 
freq_bins_low2 = freq_bins_low[0:46]
num_freqs_low = len(freq_bins_low2)


#Datafame containing all raw PSD
df_psd_all = pd.DataFrame({
    'Subject': np.repeat(subjects, num_freqs),
    'Age': np.repeat(ages, num_freqs),
    'Age_bin':np.repeat(age_bin, num_freqs),
    'Frequency': np.tile(freq_bins,num_sub),
    'PSD': np.mean(prime_psd[:,6:138,:], axis = 2).flatten()
    })

# DataFrame contianing low frequency (3-30Hz) + fooof'd data
df_psd_low = pd.DataFrame({
    'Subject':np.repeat(subjects,num_freqs_low),
    'Age':np.repeat(ages,num_freqs_low),
    'Age_bin':np.repeat(age_bin,num_freqs_low),
    'HM':np.repeat(hm,num_freqs_low),
    'Frequency':np.tile(freq_bins_low2,num_sub),
    'PSD_raw':np.mean(prime_psd[:,6:52,:], axis = 2).flatten(),
    'PSD':np.mean(periodic_adj,axis = 2).flatten(),
    'ap_PSD': np.mean(fooof_aperiodic_psd[:,0:46,:],axis = 2).flatten(),
    'Spectrum':np.mean(fooof_spectrum_psd[:,0:46,:],axis = 2).flatten()
    })


#Dataframe containing Slope + Offset per region
df_fooof_params_reg = pd.DataFrame({
    'Subject': np.tile(subjects,num_reg),
    'Subject_visit_ID':np.tile(subject_visitId,num_reg),
    'Sex': np.tile(prime_sex,num_reg),
    'Age': np.tile(ages,num_reg),
    'HM': np.tile(hm,num_reg),
    'Region': np.repeat(reg_name, num_sub),
    'Slope': fooof_params[:,:,-1].T.flatten(),
    'Offset': fooof_params[:,:,0].T.flatten()
    })


#Dataframe for regional slope-offset
# Flatten subject-region grid correctly
subjects_flat = np.tile(subjects, num_reg)              # Repeat each subject block for each region
subject_ids_flat = np.tile(subject_visitId, num_reg)
sex_flat = np.tile(prime_sex, num_reg) 
ages_flat = np.tile(ages, num_reg)
hm_flat = np.tile(hm, num_reg)
regions_flat = np.repeat(reg_name, num_sub)             # Repeat each region once per subject

# Construct dataframe properly for regional slope-offset
df_fooof_params_avg = pd.DataFrame({
    'Subject': subjects,
    'Age': ages,
    'Sex': prime_sex,
    'Age_bin':age_bin,
    'Slope': slope_mean,
    'Offset': offset_mean
})


#Dataframes for psd stuff
###############################################################
# For NORMALIZED timeseries data, make sure you SUM the powers
# For ZSCORE timeseries data, use MEAN
###############################################################




#Dataframe for each ptp's region's avg psd for each band
theta_psd = np.mean(fooof_periodic_psd[:,0:5,:], axis = 1)
theta_psd_flat = theta_psd.T.flatten()
alpha_psd = np.mean(fooof_periodic_psd[:,5:17,:], axis = 1)
alpha_psd_flat = alpha_psd.T.flatten()
beta_psd = np.mean(fooof_periodic_psd[:,17:46,:], axis = 1)
beta_psd_flat = beta_psd.T.flatten()
lowgamma_psd = np.mean(prime_psd[:,52:94,:], axis = 1)
lowgamma_psd_flat = lowgamma_psd.T.flatten()
highgamma_psd = np.mean(prime_psd[:,111:137,:], axis = 1)
highgamma_psd_flat = highgamma_psd.T.flatten()

df_psd_reg_band = pd.DataFrame({
    'Subject': subjects_flat,
    'Subject_visit_ID': subject_ids_flat,
    'Region': regions_flat,
    'Age': ages_flat,
    'Sex':sex_flat,
    'HM': hm_flat,
    'Theta': theta_psd_flat,
    'Alpha': alpha_psd_flat,
    'Beta': beta_psd_flat, #<------------------
    'LowGamma':lowgamma_psd_flat,
    'HighGamma':highgamma_psd_flat
    })


#Dataframe for each ptp's  raw region's avg psd for each band
theta_raw_psd = np.mean(prime_psd[:, 6:11, :], axis = 1)
theta_raw_psd_flat = theta_raw_psd.T.flatten()
alpha_raw_psd = np.mean(prime_psd[:, 11:23, :], axis = 1)
alpha_raw_psd_flat = alpha_raw_psd.T.flatten()
beta_raw_psd = np.mean(prime_psd[:, 21:52, :], axis = 1)
beta_raw_psd_flat = beta_raw_psd.T.flatten()
lowgamma_psd = np.mean(prime_psd[:,52:94,:], axis = 1)
lowgamma_psd_flat = lowgamma_psd.T.flatten()
highgamma_psd = np.mean(prime_psd[:,111:137,:], axis = 1)
highgamma_psd_flat = highgamma_psd.T.flatten()

df_raw_psd_reg_band = pd.DataFrame({
    'Subject': subjects_flat,
    'Subject_visit_ID': subject_ids_flat,
    'Region': regions_flat,
    'Age': ages_flat,
    'Sex': sex_flat,
    'HM': hm_flat,
    'Theta': theta_raw_psd_flat,
    'Alpha': alpha_raw_psd_flat,
    'Beta': beta_raw_psd_flat,#<-----------------
    'LowGamma':lowgamma_psd_flat,
    'HighGamma':highgamma_psd_flat
    })


# Dataframes for each band's individual freq's per region
# Helper function to construct df
def build_band_df(psd_band, freqs, band_label):
    n_freqs = len(freqs)
    
    return pd.DataFrame({
        'Subject': np.tile(np.repeat(subjects, n_freqs), num_reg),
        'Subject_visit_ID': np.tile(np.repeat(subject_visitId, n_freqs), num_reg),
        'Age': np.tile(np.repeat(ages, n_freqs), num_reg),
        'HM' : np.tile(np.repeat(hm, n_freqs), num_reg),
        'Region': np.repeat(reg_name, num_sub * n_freqs),
        'Frequency': np.tile(freqs, num_sub * num_reg),
        'PSD': psd_band.T.flatten(),
        'Band': band_label
    })

# Build individual band dataframes
theta_psd_ind = fooof_periodic_psd[:, 0:5, :]  # [165, 3, 38]
alpha_psd_ind = fooof_periodic_psd[:, 5:17, :]  # [165, 6, 38]
beta_psd_ind  = fooof_periodic_psd[:, 17:46, :] # [165, 14, 38]

theta_df = build_band_df(theta_psd_ind, freq_bins_low[0:5], 'Theta')
alpha_df = build_band_df(alpha_psd_ind, freq_bins_low[5:17], 'Alpha')
beta_df  = build_band_df(beta_psd_ind,  freq_bins_low[17:46], 'Beta')

# Build individual band dataframes
theta_psd_raw_ind = prime_psd[:, 6:11, :]  # [165, 3, 38]
alpha_psd_raw_ind = prime_psd[:, 11:23, :]  # [165, 6, 38]
beta_psd_raw_ind  = prime_psd[:, 23:52, :] # [165, 14, 38]

theta_raw_df = build_band_df(theta_psd_raw_ind, prime_freq[6:11], 'Theta')
alpha_raw_df = build_band_df(alpha_psd_raw_ind, prime_freq[11:23], 'Alpha')
beta_raw_df  = build_band_df(beta_psd_raw_ind,  prime_freq[23:52], 'Beta')
    
     



# %% 4 - Plotting Various PSDs

sns.set(style = 'white')

band_param = 'low' #'all', 'low', 'gamma'
y_param = 'PSD'
log = True

fig, ax = plt.subplots(figsize = (11.7,10)) # make square somehow

    
if band_param == 'all':
    df_plot = df_psd_all
    yval = 'PSD'
    ax.set_xlim(3,80)
    #line = ax.axvline(x = 30, linestyle='dotted', linewidth=4)
else:   
    yval = y_param
    if band_param == 'low':
        df_plot = df_psd_low
        #df_plot = df_psd_all[df_psd_all['Frequency']<= 30]  
        ax.set_xlim(3,30)
    else:
        df_plot = df_psd_all[(df_psd_all['Frequency']>= 30) & (df_psd_all['Frequency']<= 80)] 
        ax.set_xlim(30,80)
        
def _shade(ax, a, b, color, alpha=0.1, hatch=None, zorder=0, hatch_ec="#000000", hatch_lw=0):
    from matplotlib.colors import to_rgb
    xmin, xmax = ax.get_xlim()
    a, b = max(a, xmin), min(b, xmax)
    if b <= a:
        return

    if hatch is None:
        ax.axvspan(a, b,
                   facecolor=color,
                   alpha=alpha,          # ok for normal bands
                   edgecolor='none',
                   zorder=zorder)
    else:
        # transparent facecolor via RGBA, leave patch alpha unset
        rgba = (*to_rgb(color), 0.0)
        patch = ax.axvspan(a, b,
                           facecolor=rgba,     # transparent fill so hatch shows over background
                           edgecolor=hatch_ec, # hatch color (uses edgecolor)
                           alpha = 0.3,
                           hatch=hatch,
                           zorder=zorder)
        patch.set_linewidth(hatch_lw)          # 0 = no border outline
        
col_redpurple = "#99004d"        
        
        
# palette (tweak hexes if you want different hues)
col_redpurple = "#CC3399"  # red-purple
col_purple    = "#400080"  # purple
col_grey      = '#000000'
col_green     = "#00802b"  # green
col_orange    = "#ff6c00"  # orange
col_red       = "#ff0000"  # red

# ranges (Hz). 55–65 intentionally left unshaded per your spec.
# _shade(ax, -np.inf, 6,  col_redpurple)  # from min to 6
# _shade(ax, 6,  13, col_purple)         # 6–13
# _shade(ax, 13, 30, col_green)          # 13–30
# _shade(ax, 30, 55, col_orange)         # 30–55
# _shade(ax, 65, 80, col_red)            # 65–80
# #_shade(ax, 55, 65, col_grey)            # 65–80
# _shade(ax, 55, 65, color="#ffffff", hatch='////', zorder=0, hatch_ec="#000000", hatch_lw=0)

        
# Lineplot  , hue = 'Age_bin'
sns.lineplot(df_plot, x = "Frequency", y = yval, errorbar=('ci',95), estimator='mean',hue = 'Age_bin', palette = 'bright', linewidth = 4)

#sns.lineplot(df_plot, x = "Frequency", y = yval, errorbar=('ci', 95), estimator='mean', color = 'k', linewidth = 4)
if band_param in ['low']:
    line = ax.axvline(x = 6, linestyle='dotted', linewidth=2)
    line = ax.axvline(x = 13, linestyle='dotted', linewidth=2)

if band_param in ['all']:
    line = ax.axvline(x = 6, linestyle='dotted', linewidth=2)
    line = ax.axvline(x = 13, linestyle='dotted', linewidth=2)
    line = ax.axvline(x = 30, linestyle='dotted', linewidth=2)
    line = ax.axvline(x = 55, linestyle='dotted', linewidth=2)
    line = ax.axvline(x = 65, linestyle='dotted', linewidth=2)
    

if log:
    ax.set_yscale('log')
    logtxt = 'log'
else:
    logtxt = 'norm'

    
#ax.set_title("Average Power", fontsize=60, pad = 20)
ax.set_xlabel("Frequency (Hz) ", fontsize=70)
#ax.set_xticks([20,40,60,80])
#ax.set_xticks([6,13,30])
#ax.set_yticks([1e-3,1e-2])
#ax.set_ylabel("Power (log $fT^2$/Hz)", fontsize=50)
ax.set_ylabel(r'Power (Hz$^{{-1}}$)', fontsize=70) # (log $(\frac{fT^2}{Hz})$)
ax.tick_params(axis='both', which='major', labelsize=70)

#ax.set_ylim([np.min(df_plot[yval]),np.max(df_plot[yval])])

# from matplotlib.ticker import ScalarFormatter
# # Format y-axis in scientific notation (e.g., 1×10⁻⁴)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-2, 2))  # Force sci notation in reasonable range
# ax.yaxis.set_major_formatter(formatter)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))  # Optional
# ax.yaxis.get_offset_text().set_size(50) 

# Modify legend labels to "1-year-old", etc.
# handles, labels = ax.get_legend_handles_labels()
# custom_labels = [f"{int(float(label))}-year-old" for label in labels]
# ax.legend(handles, custom_labels, fontsize=40,loc = 'upper center')
ax.get_legend().remove()


# Grid and layout
plt.grid(False)
plt.tight_layout()


filepath = os.path.join(base_dir, r'psd',fr'{logtxt}scale_{band_param}_avg{y_param}.png')
plt.savefig(filepath)
plt.show()




# %% 6 - Time for LME on Regional Slope Offset - THIS ONE HAS USEFUL FUNCTINOS

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
import warnings

# ---------- helpers (over-ridden from run_lme in .functions) ----------
def can_fit_mixed_X(df_slice, y_col, xcols, group_col='Subject'):

    return True

def fit_mixed_only_X(df_slice, y_col, xcols, group_col='Subject'):

    
    df_slice = df_slice.dropna(subset=[y_col] + xcols)
    X = sm.add_constant(df_slice[xcols], has_constant='add')
    print(X)
    y = df_slice[y_col]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hush convergence warnings in loops
            mdl = MixedLM(y, X, groups=df_slice[group_col])
            res = mdl.fit()#method='lbfgs', reml=True, maxiter=1000, disp=False)
        return dict(res.params), dict(res.pvalues)
    except Exception:
        # couldn’t fit MixedLM; return NaNs for the requested params
        print('F- you')
        print((['const'] + xcols))
        nan_params = {k: np.nan for k in (['const'] + xcols)}
        nan_pvals  = {k: np.nan for k in (['const'] + xcols)}
        return nan_params, nan_pvals

def fdr_nan_safe(pvals, alpha=0.05):

    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    p_corr = np.full_like(p, np.nan, dtype=float)
    reject = np.zeros_like(p, dtype=bool)
    if finite.sum() > 0:
        rej, p_c, _, _ = multipletests(p[finite], alpha=alpha, method='fdr_bh')
        p_corr[finite] = p_c
        reject[finite] = rej
    return p_corr, reject

# ---------- main (LME-only, Age + Sex_code) ----------
slope_pvals = np.full(num_reg, np.nan)
slope_coeff = np.full(num_reg, np.nan)

offset_pvals = np.full(num_reg, np.nan)
offset_coeff = np.full(num_reg, np.nan)

# (optional bookkeeping if you want to see which failed)
slope_lme_method  = [0]*num_reg  # 1 = fit ok, 0 = skipped/failed
offset_lme_method = [0]*num_reg

for reg_idx, reg in enumerate(reg_name):
    df_temp = df_fooof_params_reg[df_fooof_params_reg['Region'] == reg].copy()

    # numeric Sex covariate (0 = F, 1 = M)
    df_temp['Sex_code'] = (df_temp['Sex'] == 'M').astype(int)

    # Columns needed per model
    base_cols = ['Subject', 'Age', 'Sex_code', 'HM']

    # ----- Offset ~ Age + Sex_code -----
    cols_needed = base_cols + ['Offset']
    df_off = df_temp[cols_needed].dropna()
    if can_fit_mixed_X(df_off, y_col='Offset', xcols=['Age','Sex_code', 'HM'], group_col='Subject'):
        params, pvals = fit_mixed_only_X(df_off, y_col='Offset', xcols=['Age','Sex_code'], group_col='Subject')
        offset_pvals[reg_idx] = pvals.get('Age', np.nan)
        offset_coeff[reg_idx] = params.get('Age', np.nan)
        offset_lme_method[reg_idx] = 1  # fitted

    # ----- Slope ~ Age + Sex_code -----
    cols_needed = base_cols + ['Slope']
    df_slp = df_temp[cols_needed].dropna()
    if can_fit_mixed_X(df_slp, y_col='Slope', xcols=['Age','Sex_code', 'HM'], group_col='Subject'):
        params, pvals = fit_mixed_only_X(df_slp, y_col='Slope', xcols=['Age','Sex_code'], group_col='Subject')
        slope_pvals[reg_idx] = pvals.get('Age', np.nan)
        slope_coeff[reg_idx] = params.get('Age', np.nan)
        slope_lme_method[reg_idx] = 1  # fitted

# ---------- FDR (NaN-safe) ----------
slope_pvals_correct, slope_reject   = fdr_nan_safe(slope_pvals, alpha=0.05)
offset_pvals_correct, offset_reject = fdr_nan_safe(offset_pvals, alpha=0.05)

# Mask coefficients by FDR decisions (keep NaNs as NaNs)
slope_coeff_FDR  = slope_coeff.copy()
slope_coeff_FDR[~slope_reject]   = 0.0

offset_coeff_FDR = offset_coeff.copy()
offset_coeff_FDR[~offset_reject] = 0.0

# Optional: zero-out corrected p-values where not rejected (preserve NaNs)
slope_pvals_correct[~slope_reject]   = 0.0
offset_pvals_correct[~offset_reject] = 0.0

df_reg_lme_results = pd.DataFrame({
    'Region': reg_name,
    'Slope_pval': slope_pvals_correct,
    'Slope_coeff': slope_coeff,
    'Slope_coeff_FDR': slope_coeff_FDR,
    'Slope_method': slope_lme_method,
    'Offset_pval': offset_pvals_correct,
    'Offset_coeff': offset_coeff,
    'Offset_coeff_FDR': offset_coeff_FDR,
    'Offset_method': offset_lme_method
})


# %% 6.0.2- LME on Offset/Slope with Sex

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
        

# ---------- main ----------
sexes = ['M', 'F']
metrics = ['Slope', 'Offset']

# Use NaNs by default; avoids fake significance from zeros
slope_pvals_M = np.full(num_reg, np.nan)
slope_coeff_M = np.full(num_reg, np.nan)
slope_pvals_F = np.full(num_reg, np.nan)
slope_coeff_F = np.full(num_reg, np.nan)

offset_pvals_M = np.full(num_reg, np.nan)
offset_coeff_M = np.full(num_reg, np.nan)
offset_pvals_F = np.full(num_reg, np.nan)
offset_coeff_F = np.full(num_reg, np.nan)

for reg_idx, reg in enumerate(reg_name):
    df_reg = df_fooof_params_reg[df_fooof_params_reg['Region'] == reg].copy()

    for sex in sexes:
        df_sex = df_reg[df_reg['Sex'] == sex][['Age', 'Subject', 'Slope', 'Offset']].dropna()
        if df_sex.empty:
            continue

        # ----- Offset ~ Age -----
        params, pvals = fit_mixed_only_X(df_sex, y_col='Offset', xcols = ['Age'], group_col='Subject')
        if sex == 'M':
            offset_pvals_M[reg_idx] = pvals.get('Age', np.nan)
            offset_coeff_M[reg_idx] = params.get('Age', np.nan)
        else:
            offset_pvals_F[reg_idx] = pvals.get('Age', np.nan)
            offset_coeff_F[reg_idx] = params.get('Age', np.nan)

        # ----- Slope ~ Age -----
        params, pvals = fit_mixed_only_X(df_sex, y_col='Slope', xcols= ['Age'], group_col='Subject')
        if sex == 'M':
            slope_pvals_M[reg_idx] = pvals.get('Age', np.nan)
            slope_coeff_M[reg_idx] = params.get('Age', np.nan)
        else:
            slope_pvals_F[reg_idx] = pvals.get('Age', np.nan)
            slope_coeff_F[reg_idx] = params.get('Age', np.nan)

# ---------- FDR (NaN-safe) ----------
slope_pvals_M_corr, slope_rej_M = fdr_nan_safe(slope_pvals_M, alpha=0.05)
slope_pvals_F_corr, slope_rej_F = fdr_nan_safe(slope_pvals_F, alpha=0.05)
offset_pvals_M_corr, offset_rej_M = fdr_nan_safe(offset_pvals_M, alpha=0.05)
offset_pvals_F_corr, offset_rej_F = fdr_nan_safe(offset_pvals_F, alpha=0.05)

# Mask coefficients by FDR decisions (keep NaNs as NaNs)
slope_coeff_M_FDR  = slope_coeff_M.copy()
slope_coeff_M_FDR[~slope_rej_M] = 0.0
slope_coeff_F_FDR  = slope_coeff_F.copy()
slope_coeff_F_FDR[~slope_rej_F] = 0.0

offset_coeff_M_FDR = offset_coeff_M.copy()
offset_coeff_M_FDR[~offset_rej_M] = 0.0
offset_coeff_F_FDR = offset_coeff_F.copy()
offset_coeff_F_FDR[~offset_rej_F] = 0.0

# Optional: zero out non-significant corrected pvals (preserve NaNs)
slope_pvals_M_corr[~slope_rej_M] = 0.0
slope_pvals_F_corr[~slope_rej_F] = 0.0
offset_pvals_M_corr[~offset_rej_M] = 0.0
offset_pvals_F_corr[~offset_rej_F] = 0.0


# %% 6.1 - Plot test: Plotting the slope/offset for all region - no use
from scipy.stats import linregress
    

param = 'Offset'

for reg_idx, reg in enumerate(reg_name):
    df_reg = df_fooof_params_reg[df_fooof_params_reg['Region'] == reg]

    # Plot scatter or regplot
    fig6_1 = plot_stuff_vs_stuffv2(df_reg['Age'].values, df_reg[f'{param}'].values,
                                   f'{reg_name_full[reg_idx]}', plot_type='regplot', ylabel=f'{param}')

    # Identify repeated subjects (i.e., multiple sessions)
    subject_counts = df_reg['Subject'].value_counts()
    repeated_subjects = subject_counts[subject_counts > 1].index

    # Plot connected lines for repeated sessions
    for idx, subj in enumerate(repeated_subjects):
        #color = color_list[idx % len(color_list)]
        subj_data = df_reg[df_reg['Subject'] == subj].sort_values('Age')  # optional: sort by age for line direction
        plt.plot(subj_data['Age'].values, subj_data[f'{param}'].values,
                 color='darkred', alpha=0.4, linewidth=2)

    # Linregress + LME values
    linslope, _, _, linpval, _ = linregress(df_reg['Age'], df_reg[f'{param}'])
    legend_handles = [
        Line2D([0], [0], color='darkred', lw=5, label=fr'LME p-value  = {slope_pvals[reg_idx]:.2e}'), #<--- Change for SLope/offset
        Line2D([0], [0], color='darkred', lw=5, label=fr'Lin p-value  = {linpval:.2e}')
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=40)

    # Save
    plt.savefig(fr'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\analysis\powerspec\slope_off\off_reg\{reg}_{param}.png')
    plt.close('all')



    

# %% 7 - Brainplots for Slope/Offset

from functions.brainplots import make_4d_atlas_nifti, surface_brain_plotv2
from functions.brainplots import make_atlas_nifti
from scipy.stats import zscore


offset_array = np.array(df_reg_lme_results['Offset_coeff_FDR'])
min_nonzero = np.min(offset_array[offset_array > 0])

# Subtract min_nonzero only from non-zero entries
offset_values = offset_coeff_FDR#np.where(offset_array != 0, offset_array - 0.8*min_nonzero, 0)

fig2 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d),offset_values),
                          r'C:\mne\fs',# surf = "inflated",
                          cmap = "coolwarm",
                          symmetric=True,
                          cbar_label=fr"Offset $\beta$ - FDR", datmin = -np.max(np.array(df_reg_lme_results['Offset_coeff_FDR'])), 
                          datmax = np.max(np.array(df_reg_lme_results['Offset_coeff_FDR']))
                          , cort = (0.6,0.6,0.6)
                          #, threshold = 0.027
                          )
filepath = os.path.join(base_dir,fr'slope_off',fr'offset_fdr_bp.png')
fig2.savefig(filepath)



# %% 7.1 - Plotting Average Slope/Offset Brain Plots
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2, make_4d_atlas_nifti
from scipy.stats import zscore

value = np.mean(fooof_params[:,:,1], axis = 0) #1 for slope, 0 for offset

fig2_2 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), value - min(value)),
                          r'C:\mne\fs',# surf = "inflated",
                          cmap = "BuGn",
                          symmetric=False,
                          cbar_label=fr"Slope (a.u.)", datmin = np.min(value), #Hz$^{{-1}}$
                          datmax = np.max(value)
                          , threshold = 0.05
                          )
filepath = os.path.join(base_dir,fr'slope_off',fr'mean_slope_bp.png')
fig2_2.savefig(filepath)




# %% 8.1 - Plot Offset/SLope vs Age per sex

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter

reg_oi_idx = 37  # Index of region of interest
reg_oi = reg_name_full[reg_oi_idx]
param = 'Slope'
print_pval = True

# Set up plot colors
colors = {'M': 'royalblue', 'F': 'hotpink'}
alpha = 0.85
lalpha = 1

# Get full subject-level data for the selected region
df_temp2 = df_fooof_params_reg[df_fooof_params_reg['Region'] == reg_name[reg_oi_idx]].copy()
#df_temp2['Offset'] = np.log(df_temp2['Offset'])

# Create figure
fig4, ax4 = plt.subplots(figsize=(10.15, 10.5))

# Plot separate scatter and trend lines for M and F
for sex in ['M', 'F']:
    df_sex = df_temp2[df_temp2['Sex'] == sex]
    sns.regplot(
        data=df_sex,
        x='Age',
        y=param,
        ax=ax4,
        ci=None,
        line_kws={"linewidth": 5, "color": colors[sex], "alpha": lalpha},
        scatter_kws={"s": 100, "alpha": alpha, "color": colors[sex]}#,
        #label=f"{sex}"
    )

# Connect repeated sessions with lines, colored by sex
subject_counts = df_temp2['Subject'].value_counts()
repeated_subjects = subject_counts[subject_counts > 1].index

for subj in repeated_subjects:
    subj_data = df_temp2[df_temp2['Subject'] == subj].sort_values('Age')
    sex = subj_data['Sex'].iloc[0]
    plt.plot(
        subj_data['Age'].values,
        subj_data[param].values,
        color=colors[sex],
        alpha=0.4,
        linewidth=3
    )

# Axes and formatting
ax4.set_xlabel('Age', fontsize=90)
ax4.set_ylabel(fr'{param} (Hz$^{{-1}}$)', fontsize=80)
ax4.set_xticks([1, 2, 3, 4, 5])
#ax4.set_yticks([-5,-3,0])
ax4.tick_params(axis='both', which='major', labelsize=85)
#ax4.set_yscale('log')
ax4.grid(True)



# Y-axis formatting (scientific)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-2, 2))
# ax4.yaxis.set_major_formatter(formatter)
# ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax4.yaxis.get_offset_text().set_size(50)
#ax4.set_yscale('logit')




# Legend and save
if print_pval:
    param_pval_M = slope_pvals_M_corr[reg_oi_idx] if param == 'Slope' else offset_pvals_M_corr[reg_oi_idx]
    param_pval_F = slope_pvals_F_corr[reg_oi_idx] if param == 'Slope' else offset_pvals_F_corr[reg_oi_idx]
    param_pval = slope_pvals_correct[reg_oi_idx] if param == 'Slope' else offset_pvals_correct[reg_oi_idx]
    #param_method = slope_lme_method[reg_oi_idx] if param == 'Slope' else offset_lme_method[reg_oi_idx]
    legend_handles = [
        Line2D([0], [0], color='royalblue', lw=5, label=fr'p-value = {param_pval_M:.2e}'),
        Line2D([0], [0], color='hotpink', lw=5, label=fr'p-value = {param_pval_F:.2e}')
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=40)
    filepath = os.path.join(base_dir, fr'slope_off', fr'MF_{param}_{reg_name[reg_oi_idx]}_pval.png')
else:
   # plt.legend(fontsize=40)
    filepath = os.path.join(base_dir, fr'slope_off', fr'MF_{param}_{reg_name[reg_oi_idx]}.png')
    #plt.get_legend().remove()
    

plt.tight_layout()
#fig4.savefig(filepath)
plt.show()



# %% 8.1.2 - Slope/Offset ~ Age + Sex plot

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter

reg_oi_idx = 18  # Index of region of interest
reg_oi = reg_name_full[reg_oi_idx]
param = 'Offset'
if param == 'Slope':
    units = '(a.u.)'
    mpval = slope_pvals_M_corr[reg_oi_idx]
    fpval = slope_pvals_F_corr[reg_oi_idx]
else:
    units = (fr'(Hz$^{{-1}}$)')
    mpval = offset_pvals_M_corr[reg_oi_idx]
    fpval = offset_pvals_F_corr[reg_oi_idx]
print_pval = False


# Set up plot colors
colors = {'M': 'royalblue', 'F': 'hotpink'}
alpha = 0.85
lalpha = 1

# Get full subject-level data for the selected region
df_temp2 = df_fooof_params_reg[df_fooof_params_reg['Region'] == reg_name[reg_oi_idx]].copy()
#df_temp2['Offset'] = np.log(df_temp2['Offset'])

# Create figure
fig4, ax4 = plt.subplots(figsize=(10.07, 10.5))

# Plot separate scatter and trend lines for M and F
for sex in ['M', 'F']:
    df_sex = df_temp2[df_temp2['Sex'] == sex]
    sns.regplot(
        data=df_sex,
        x='Age',
        y=param,
        ax=ax4,
        ci=None,
        line_kws={"linewidth": 5, "color": colors[sex], "alpha": lalpha},
        scatter_kws={"s": 100, "alpha": alpha, "color": colors[sex]}#,
        #label=f"{sex}"
    )

# Connect repeated sessions with lines, colored by sex
subject_counts = df_temp2['Subject'].value_counts()
repeated_subjects = subject_counts[subject_counts > 1].index

for subj in repeated_subjects:
    subj_data = df_temp2[df_temp2['Subject'] == subj].sort_values('Age')
    sex = subj_data['Sex'].iloc[0]
    plt.plot(
        subj_data['Age'].values,
        subj_data[param].values,
        color=colors[sex],
        alpha=0.4,
        linewidth=3
    )

# Axes and formatting
ax4.set_xlabel('Age', fontsize=90)
#ax4.set_ylabel(fr'{param} (Hz$^{{-1}}$)', fontsize=80)
ax4.set_ylabel(fr'{param} {units}', fontsize=80)
ax4.set_xticks([1, 2, 3, 4, 5])
ax4.set_yticks([0,1])
ax4.tick_params(axis='both', which='major', labelsize=85)
#ax4.set_yscale('log')
ax4.grid(True)



# Y-axis formatting (scientific)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-2, 2))
# ax4.yaxis.set_major_formatter(formatter)
# ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax4.yaxis.get_offset_text().set_size(50)
#ax4.set_yscale('logit')




# Legend and save
if print_pval:
    param_pval = slope_pvals_correct[reg_oi_idx] if param == 'Slope' else offset_pvals_correct[reg_oi_idx]
    coef = slope_coeff_FDR[reg_oi_idx] if param == 'Slope' else offset_coeff_FDR[reg_oi_idx]
    #param_method = slope_lme_method[reg_oi_idx] if param == 'Slope' else offset_lme_method[reg_oi_idx]
    legend_handles = [
        Line2D([0], [0], color='k', lw=5, label=fr'p= {param_pval:.2e}'),
        Line2D([0], [0], color='orange', lw=5, label=fr'coef = {coef:.2e}'),
        Line2D([0], [0], color='blue', lw=5, label=fr'M = {mpval:.2e}'),
        Line2D([0], [0], color='pink', lw=5, label=fr'F = {fpval:.2e}'),
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=40)
    filepath = os.path.join(base_dir, fr'slope_off', fr'MF_{param}_{reg_name[reg_oi_idx]}_pval.png')
else:
   # plt.legend(fontsize=40)
    filepath = os.path.join(base_dir, fr'slope_off', fr'MF_{param}_{reg_name[reg_oi_idx]}.png')
    #plt.get_legend().remove()
      

plt.tight_layout()
fig4.savefig(filepath)
plt.show()




# %% 10 - LME on the regional PSD per band (age+sex)
theta_pval = np.zeros(num_reg)
theta_coeff = np.zeros(num_reg)
theta_const = np.zeros(num_reg)

alpha_pval = np.zeros(num_reg)
alpha_coeff = np.zeros(num_reg)
alpha_const = np.zeros(num_reg)

beta_pval = np.zeros(num_reg)
beta_coeff = np.zeros(num_reg)
beta_const = np.zeros(num_reg)

lowgamma_pval = np.zeros(num_reg)
lowgamma_coeff = np.zeros(num_reg)
lowgamma_const = np.zeros(num_reg)

highgamma_pval = np.zeros(num_reg)
highgamma_coeff = np.zeros(num_reg)
highgamma_const = np.zeros(num_reg)

df_forthis = df_psd_reg_band.copy()

# Set whether you're using RAW or fooofed data
for reg_idx, reg in enumerate(reg_name):
    df_tmp = df_forthis[df_forthis['Region']==reg].copy()
    df_tmp['Sex_code'] = (df_tmp['Sex'] == 'M').astype(int)
    tparams, tpvals = fit_mixed_only_X(df_tmp, xcols=['Age','Sex_code'], y_col='Theta', group_col='Subject')
    theta_pval[reg_idx], theta_coeff[reg_idx], theta_const[reg_idx] = tpvals.get('Age', np.nan), tparams.get('Age', np.nan), tparams.get('const', np.nan)
    
    aparams, apvals = fit_mixed_only_X(df_tmp, xcols=['Age','Sex_code'], y_col='Alpha', group_col='Subject')
    alpha_pval[reg_idx], alpha_coeff[reg_idx], alpha_const[reg_idx] = apvals.get('Age', np.nan), aparams.get('Age', np.nan), aparams.get('const', np.nan)

    bparams, bpvals = fit_mixed_only_X(df_tmp, xcols=['Age','Sex_code'], y_col='Beta', group_col='Subject')
    beta_pval[reg_idx], beta_coeff[reg_idx], beta_const[reg_idx] = bpvals.get('Age', np.nan), bparams.get('Age', np.nan), bparams.get('const', np.nan)

    lparams, lpvals = fit_mixed_only_X(df_tmp, xcols=['Age','Sex_code'], y_col='LowGamma', group_col='Subject')
    lowgamma_pval[reg_idx], lowgamma_coeff[reg_idx], lowgamma_const[reg_idx] = lpvals.get('Age', np.nan), lparams.get('Age', np.nan), lparams.get('const', np.nan)

    hparams, hpvals = fit_mixed_only_X(df_tmp, xcols=['Age','Sex_code'], y_col='HighGamma', group_col='Subject')
    highgamma_pval[reg_idx], highgamma_coeff[reg_idx], highgamma_const[reg_idx] = hpvals.get('Age', np.nan), hparams.get('Age', np.nan), hparams.get('const', np.nan)
    
# ok gotta FDR the pvalues for each region
theta_pval_fdr, theta_pval_result = fdr_nan_safe(theta_pval, alpha = 0.05)
alpha_pval_fdr, alpha_pval_result = fdr_nan_safe(alpha_pval, alpha = 0.05)
beta_pval_fdr,beta_pval_result = fdr_nan_safe(beta_pval, alpha = 0.05)
lowgamma_pval_fdr,lowgamma_pval_result = fdr_nan_safe(lowgamma_pval, alpha = 0.05)
highgamma_pval_fdr,highgamma_pval_result= fdr_nan_safe(highgamma_pval, alpha = 0.05)

# Zero out non-significant beta values
theta_coeff_masked = theta_coeff.copy()
theta_coeff_masked[~theta_pval_result] = 0

alpha_coeff_masked = alpha_coeff.copy()
alpha_coeff_masked[~alpha_pval_result] = 0

beta_coeff_masked = beta_coeff.copy()
beta_coeff_masked[~beta_pval_result] = 0

lowgamma_coeff_masked = lowgamma_coeff.copy()
lowgamma_coeff_masked[~lowgamma_pval_result] = 0

highgamma_coeff_masked = highgamma_coeff.copy()
highgamma_coeff_masked[~highgamma_pval_result] = 0

# %% 10.0.1 - LME ono regional PSD per band per sex

import numpy as np
from statsmodels.stats.multitest import multipletests

# ---------- 1.  pre-allocate ----------
bands        = ['Theta','Alpha','Beta','LowGamma','HighGamma']
sexes        = ['M','F']
results_pval = {band: {s: np.zeros(num_reg) for s in sexes} for band in bands}
results_coef = {band: {s: np.zeros(num_reg) for s in sexes} for band in bands}

df_forthis2 = df_psd_reg_band.copy()
# ---------- 2.  run the LMEs ----------
for reg_idx, reg in enumerate(reg_name):
    df_reg = df_forthis2[df_forthis2['Region'] == reg]

    for sex in sexes:
        df_sex = df_reg[df_reg['Sex'] == sex]
        if df_sex.empty:
            continue

        for band in bands:
            params, pvals = fit_mixed_only_X(df_sex,
                             xcols=['Age','HM'],
                             y_col=band,
                             group_col='Subject')
            results_pval[band][sex][reg_idx] = pvals.get('Age', np.nan)
            results_coef[band][sex][reg_idx] = params.get('Age', np.nan)

# ---------- 3.  FDR-correct per sex ----------
results_pval_fdr   = {}
results_coef_mask  = {}

for band in bands:
    results_pval_fdr[band]  = {}
    results_coef_mask[band] = {}

    for sex in sexes:
        pvals  = results_pval[band][sex]
        pvals_fdr, rej = fdr_nan_safe(pvals, alpha=.05)

        # store corrected p-values
        results_pval_fdr[band][sex] = pvals_fdr

        # mask non-significant coefficients
        coef         = results_coef[band][sex].copy()
        coef[~rej]   = 0
        results_coef_mask[band][sex] = coef



# %% 10.1 - Brain Plotting PSD relation
from functions.brainplots import make_4d_atlas_nifti, surface_brain_plotv2
#from functions.brainplots import make_atlas_nifti2, surface_brain_plotv4
from functions.brainplots import make_atlas_nifti
from scipy.stats import zscore

FDR = ' - FDR'
FDR_file = '_FDR'
band = 'theta'
dat = alpha_coeff_masked

# Ok, plotting the brain 
fig6 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d),dat - np.min(abs(dat))),
                          r'C:\mne\fs', #surf = "inflated",
                          cmap = "coolwarm",
                          symmetric=True,
                          cbar_label=fr"$\{band}$ Coeff{FDR}", datmin = -np.max(abs(dat)),
                          datmax = np.max(abs(dat))#,
                          ,threshold=0.001
                          , cort = (0.6,0.6,0.6)
                          )
filepath = os.path.join(base_dir,fr'psd_bands',fr'{band}_coeff_bp{FDR_file}.png')
#fig6.savefig(filepath)



# %% 10.3 - Plotting PSD vs Age per region w/ sex

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter

reg_oi_idx = 24  # Index of region of interest
reg_oi = reg_name_full[reg_oi_idx]
band = 'Beta'
greek = 'beta'
print_pval = True
log_q = True


# Set up plot colors
colors = {'M': 'royalblue', 'F': 'hotpink'}
alpha = 0.85
lalpha = 1

# Get full subject-level data for the selected region
df_temp4 = df_raw_psd_reg_band[df_raw_psd_reg_band['Region'] == reg_name[reg_oi_idx]].copy()


if log_q:
    log = 'log'
    df_temp4[band] = np.log(df_temp4[band])
else:
    log = ''

# Create figure
fig6, ax6 = plt.subplots(figsize=(11.5, 10))

# Plot separate scatter and trend lines for M and F
# for isex in ['M', 'F']:
#     df_sex = df_temp4[df_temp4['Sex'] == isex].copy()

#     # Ensure numeric + drop non-finite (handles log-created ±inf)
#     df_sex['Age'] = pd.to_numeric(df_sex['Age'], errors='coerce')
#     df_sex[band]  = pd.to_numeric(df_sex[band],  errors='coerce')
#     df_sex = df_sex.replace([np.inf, -np.inf], np.nan).dropna(subset=['Age', band])

#     # Need at least 2 unique x values for a line
#     if df_sex['Age'].nunique() >= 2:
#         sns.regplot(
#             data=df_sex, x='Age', y=band, ax=ax6,
#             ci=None, n_boot=0, truncate=False,   # no CI, no truncation issues
#             line_kws={"linewidth": 5, "color": colors[isex], "alpha": lalpha},
#             scatter_kws={"s": 100, "alpha": alpha, "color": colors[isex]},
#         )
#     else:
#         # fallback: just scatter if not enough points to fit
#         ax6.scatter(df_sex['Age'], df_sex[band], s=100, alpha=alpha, color=colors[isex])
for isex in ['M', 'F']:
    df_sex = df_temp4[df_temp4['Sex'] == isex]
    sns.regplot(
        data=df_sex,
        x='Age',
        y=band,
        ax=ax6,
        ci = None,
        line_kws={"linewidth": 5, "color": colors[isex], "alpha": lalpha},
        scatter_kws={"s": 100, "alpha": alpha, "color": colors[isex]}#,
        #label=f"{isex}"
    )

# Connect repeated sessions with lines, colored by sex
subject_counts = df_temp4['Subject'].value_counts()
repeated_subjects = subject_counts[subject_counts > 1].index

for subj in repeated_subjects:
    subj_data = df_temp4[df_temp4['Subject'] == subj].sort_values('Age')
    isex = subj_data['Sex'].iloc[0]
    plt.plot(
        subj_data['Age'].values,
        subj_data[band].values,
        color=colors[isex],
        alpha=0.4,
        linewidth=3
    )

# Axes and formatting
if log_q:
    #log = log
    ax6.set_yscale('linear')
else:
    log = ''
    ax6.set_yscale('linear')
    
ax6.set_xlabel('Age', fontsize=90)
if log_q:
    ax6.set_ylabel(fr'$\{greek}$ log(Hz$^{{-1}}$)', fontsize=90)
else:
    ax6.set_ylabel(fr'$\{greek}$ (Hz$^{{-1}}$)', fontsize=90)
ax6.set_xticks([1, 2, 3, 4, 5])
#ax6.set_yticks([0.0,0.02,0.04])
ax6.tick_params(axis='both', which='major', labelsize=85)
ax6.grid(True)


# Y-axis formatting (scientific)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-2, 2))
# ax4.yaxis.set_major_formatter(formatter)
# ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax4.yaxis.get_offset_text().set_size(50)
#)





# Legend and save
if print_pval:
    band_pval_M = results_pval_fdr[band]['M'][reg_oi_idx]
    band_pval_F = results_pval_fdr[band]['F'][reg_oi_idx]
    #param_method = slope_lme_method[reg_oi_idx] if param == 'Slope' else offset_lme_method[reg_oi_idx]
    legend_handles = [
        Line2D([0], [0], color='royalblue', lw=5, label=fr'p-value = {band_pval_M:.2e}'),
        Line2D([0], [0], color='hotpink', lw=5, label=fr'p-value = {band_pval_F:.2e}')
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=40)
    filepath = os.path.join(base_dir, fr'psd_bands', fr'raw_wMF_{reg_name[reg_oi_idx]}_{band}_avgPSD_vs_Agepval_{log}.png')
else:
   # plt.legend(fontsize=40)
    filepath = os.path.join(base_dir, fr'psd_bands', fr'raw_wMF_{reg_name[reg_oi_idx]}_{band}_avgPSD_vs_Age_{log}.png')
    #plt.get_legend().remove()
    

plt.tight_layout()
#fig6.savefig(filepath)
plt.show()


# %% 10.4 - Plotting PSD vs Age REgional + sex 

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter

reg_oi_idx = 24  # Index of region of interest
reg_oi = reg_name_full[reg_oi_idx]
band = 'Beta'
greek = 'beta'
bandreg_pval = beta_pval_fdr[reg_oi_idx]
bandreg_coeff = beta_coeff_masked[reg_oi_idx]
cept = beta_const[reg_oi_idx]

Mbandreg_pval = results_pval_fdr[band]['M'][reg_oi_idx]
Fbandreg_pval = results_pval_fdr[band]['F'][reg_oi_idx]
print_pval = False
log_q = False
savegif = False


# Set up plot colors
colors = {'M': '0.8', 'F': '0.8'}
alpha = 0.85
lalpha = 1

# Get full subject-level data for the selected region
df_temp4 = df_psd_reg_band[df_psd_reg_band['Region'] == reg_name[reg_oi_idx]].copy()


if log_q:
    log = 'log'
    df_temp4[band] = np.log(df_temp4[band])
else:
    log = ''
    
x_grid = np.linspace(df_temp4['Age'].min(), df_temp4['Age'].max(), 200)
firstval = cept - (bandreg_coeff * x_grid[0])
y_line = firstval + bandreg_coeff * x_grid

# Create figure
fig6, ax6 = plt.subplots(figsize=(15, 10))



for sex in ['M', 'F']:
    sns.scatterplot(
        data=df_temp4[df_temp4['Sex'] == sex],
        x='Age', y=band,
        ax=ax6,
        s=180,
        facecolor='0.75',    # fill colour
        edgecolor=colors[sex],    # outline colour
        alpha=0.85,               # fill alpha
        linewidth=0.9,            # outline width
        zorder=3                  # keeps points above line if needed
    )
# Connect repeated sessions with lines, colored by sex
subject_counts = df_temp4['Subject'].value_counts()
repeated_subjects = subject_counts[subject_counts > 1].index

#repeated_subjects = ['TDC103', 'TDC050']
for subj in repeated_subjects:
    subj_data = df_temp4[df_temp4['Subject'] == subj].sort_values('Age')
    isex = subj_data['Sex'].iloc[0]
    # print(isex, subj_data['Subject'].iloc[0])
    # print(subj_data['Age'])
    # print(subj_data['Alpha'])
    plt.plot(
        subj_data['Age'].values,
        subj_data[band].values,
        color='0.85',
        alpha=0.9,
        linewidth=3
    )
    
ax6.plot(x_grid, y_line, color='0.1', lw=7, zorder=4)


    
ax6.set_xlabel('Age', fontsize=70)
if log_q:
    ax6.set_ylabel(fr'$\{greek}$ log(Hz$^{{-1}}$)', fontsize=70)
else:
    ax6.set_ylabel(fr'$\{greek}$ (Hz$^{{-1}}$)', fontsize=70)
ax6.set_xticks([1, 2, 3, 4, 5])
#ax6.set_yticks([0.0,0.02,0.04])
#ax6.set_ylim([0,0.015])
ax6.tick_params(axis='both', which='major', labelsize=65, direction='in')
#ax6.tick_params(axis='both', which='both')

#ax6.grid(True)
ax6.grid(False)  # no background grid
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)


# Y-axis formatting (scientific)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-2, 2))
# ax4.yaxis.set_major_formatter(formatter)
# ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#ax4.yaxis.get_offset_text().set_size(50)
#)





# Legend and save
if print_pval:
    
    #Custom legend handles for p-value
    legend_handles = [
        Line2D([0], [0], color='k', lw=5, label=fr'p(sx+hm)  = {bandreg_pval:3e}'),
        Line2D([0], [0], color='orange', lw=5, label=fr'coef  = {bandreg_coeff:3e}'),
        Line2D([0], [0], color='royalblue', lw=5, label=fr'M  = {Mbandreg_pval:3e}'),
        Line2D([0], [0], color='hotpink', lw=5, label=fr'F  = {Fbandreg_pval:3e}')
    ]

    plt.legend(handles=legend_handles, loc='upper right', fontsize=40)
    filepath = os.path.join(base_dir, fr'psd_bands', fr'raw_{reg_name[reg_oi_idx]}_{band}_avgPSD_vs_Agepval{log}_w_HM.png')
else:
   # plt.legend(fontsize=40)
    filepath = os.path.join(base_dir, fr'psd_bands', fr'raw_{reg_name[reg_oi_idx]}_{band}_avgPSD_vs_Age{log}_w_HM.png')
    #plt.get_legend().remove()
    

plt.tight_layout()
if savegif:
    fig6.savefig(filepath)
plt.show()


# %% 12 - Brain Plot for Average PSD

from functions.brainplots import make_4d_atlas_nifti, surface_brain_plotv2
from functions.brainplots import make_atlas_nifti2, surface_brain_plotv4, make_atlas_nifti
from scipy.stats import zscore

# theta_psd_ind = np.sum(fooof_periodic_psd[:, 0:5, :], axis = 1) / np.sum(fooof_periodic_psd[:, 0:46, :], axis = 1)  
# alpha_psd_ind = np.sum(fooof_periodic_psd[:, 5:17, :], axis = 1) / np.sum(fooof_periodic_psd[:, 0:46, :], axis = 1) 
# beta_psd_ind = np.sum(fooof_periodic_psd[:, 17:46, :], axis=1) / np.sum(fooof_periodic_psd[:, 0:46, :], axis=1)
# lo_beta_psd_ind = np.sum(fooof_periodic_psd[:, 17:30, :], axis=1) / np.sum(fooof_periodic_psd[:, 0:46, :], axis=1)
# hi_beta_psd_ind = np.sum(fooof_periodic_psd[:, 30:46, :], axis=1) / np.sum(fooof_periodic_psd[:, 0:46, :], axis=1)

theta_psd_ind = np.mean(fooof_periodic_psd[:, 0:5, :], axis = 1)
alpha_psd_ind = np.mean(fooof_periodic_psd[:, 5:17, :], axis = 1)
beta_psd_ind = np.mean(fooof_periodic_psd[:, 17:46, :], axis = 1)
lo_beta_psd_ind = np.mean(fooof_periodic_psd[:, 17:30, :], axis = 1)
hi_beta_psd_ind = np.mean(fooof_periodic_psd[:, 30:46, :], axis = 1)

#############################################################################################

# theta_psd_raw_ind = np.sum(prime_psd[:, 6:11, :], axis = 1) / np.sum(prime_psd[:, 6:52, :], axis=1)  
# alpha_psd_raw_ind = np.sum(prime_psd[:, 11:23, :], axis = 1) / np.sum(prime_psd[:, 6:52, :], axis=1)  
# beta_psd_raw_ind  = np.sum(prime_psd[:, 21:52, :], axis = 1) / np.sum(prime_psd[:, 6:52, :], axis=1)
# lo_beta_psd_raw_ind  = np.sum(prime_psd[:, 21:36, :], axis = 1) / np.sum(prime_psd[:, 6:52, :], axis=1)
# hi_beta_psd_raw_ind  = np.sum(prime_psd[:, 36:52, :], axis = 1) / np.sum(prime_psd[:, 6:52, :], axis=1)

theta_psd_raw_ind = np.mean(prime_psd[:, 6:11, :], axis = 1)
alpha_psd_raw_ind = np.mean(prime_psd[:, 11:23, :], axis = 1)
beta_psd_raw_ind = np.mean(prime_psd[:, 21:52, :], axis = 1)
lo_beta_psd_raw_ind = np.mean(prime_psd[:, 21:36, :], axis = 1)
hi_beta_psd_raw_ind = np.mean(prime_psd[:, 36:52, :], axis = 1)

lowgamma_psd_ind = np.mean(prime_psd[:,52:94,:], axis = 1)
highgamma_psd_ind = np.mean(prime_psd[:,111:137,:], axis = 1) 

band = 'highgamma'
psd_tmp = np.mean(highgamma_psd_ind, axis = 0)



# Ok, plotting the brain 
fig9 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), psd_tmp - np.min(psd_tmp)),
                          r'C:\mne\fs', surf = "pial", fade = True,
                          cmap = "Reds", 
                          symmetric=False,
                          cbar_label=fr"$\gamma_2$ (Hz$^{{-1}}$)", datmin = np.min(psd_tmp), datmax = np.max(psd_tmp) #$\{band}$$\gamma_2$
                          ,threshold = 0.00005
                          )

filename = os.path.join(base_dir,fr'psd_bands',fr'raw{band}_bp_mean.png')

fig9.savefig(filename) #C:\mne\fs


