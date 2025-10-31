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
base_dir = fr'Y:\projects\OPM-Analysis\OPM_rest_ASDvTDC\analysis\powerspec'
os.makedirs(base_dir, exist_ok=True)


# %% 1 - Load in the Data

#Load in PSD data
prime_data = loadmat(r'Y:/projects/OPM-Analysis/OPM_rest_ASDvTDC/code/ASDTDC_powers_meg38_zscored2.mat')['data']

subject_data = pd.read_excel(r'Y:\projects\OPM-Analysis\OPM_rest_ASDvTDC\demographics\opm_rest_asd_tdc_matched.xlsx')



# If it's a structured array (1x1 struct), extract it properly
if isinstance(prime_data, np.ndarray):
    prime_data = prime_data[0, 0]  # Extract first struct if needed
    
    
#drop scans if needed
drop_idx =  []
subject_data.drop(drop_idx)


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


#Dx array
dx = subject_data['dx']

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
    'peak_width_limits': [2, 5],
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


# %% 2.01 - Plot an individual ptp fooof

from fooof.plts.spectra import plot_spectrum
#from fooof.core.funcs import gaussian
import numpy as np

def gaussian(f, c, a, w):
    """A Gaussian function in log power space.

    Parameters:
    f : array
        Frequency vector.
    c : float
        Center frequency of the Gaussian.
    a : float
        Height (amplitude) of the Gaussian.
    w : float
        Width (standard deviation) of the Gaussian.

    Returns:
    Array of the Gaussian evaluated at each frequency.
    """
    return a * np.exp(-0.5 * ((f - c) / (w / 2)) ** 2)


ptpsc = 124       
for i in range(38):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    reg_idx = i
    fm = fooof_models[ptpsc][reg_idx]
    
    # Extract data
    freqs = fm.freqs
    spectrum = fm.power_spectrum
    aperiodic_fit = fm._ap_fit  # This one exists
    peak_params = fm.peak_params_
    
    # Reconstruct periodic fit manually
    periodic_fit = np.zeros_like(freqs)
    for peak in peak_params:
        center, height, width = peak
        periodic_fit += gaussian(freqs, *peak)
    
    # Reconstruct full fit
    full_fit = aperiodic_fit + periodic_fit
    
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.set(style='white')
    
    ax.plot(freqs, spectrum, label='Original Spectrum', color='crimson', linewidth=5)
    ax.plot(freqs, aperiodic_fit, label='Aperiodic Fit', color='slateblue', linestyle='--', linewidth=5, alpha = 1)
    #ax.plot(freqs, periodic_fit, label='Periodic Fit', color='#228B22', linewidth=5)
    ax.plot(freqs, full_fit, label='Full Model', color='black', linestyle='-', linewidth=4, alpha=1)
    
    # Customization (your previous settings)
    ax.axvline(x=6, linestyle='dotted', color='silver', linewidth=4)
    ax.axvline(x=13, linestyle='dotted', color='silver', linewidth=4)
    ax.set_xlabel("Frequency (Hz)", fontsize=75)
    ax.set_ylabel(fr'Power {reg_name[i]}', fontsize=45)
    ax.set_xticks([6, 13, 30])
    ax.tick_params(axis='both', labelsize=75)
    
    # Scientific Y-axis
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_size(50)
    
    
    param = 'Periodic2'
    # Final touches
    ax.legend(fontsize=30)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
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
    'Dx' : np.repeat(dx, num_freqs),
    'Age_bin':np.repeat(age_bin, num_freqs),
    'Frequency': np.tile(freq_bins,num_sub),
    'PSD': np.mean(prime_psd[:,6:138,:], axis = 2).flatten()
    })

# DataFrame contianing low frequency (3-30Hz) + fooof'd data
df_psd_low = pd.DataFrame({
    'Subject':np.repeat(subjects,num_freqs_low),
    'Age':np.repeat(ages,num_freqs_low),
    'Age_bin':np.repeat(age_bin,num_freqs_low),
    'Dx' : np.repeat(dx, num_freqs_low),
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
    'Dx' : np.tile(dx, num_reg),
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
dx_flat = np.tile(dx, num_reg)
hm_flat = np.tile(hm, num_reg)
regions_flat = np.repeat(reg_name, num_sub)             # Repeat each region once per subject

# Construct dataframe properly for regional slope-offset
df_fooof_params_avg = pd.DataFrame({
    'Subject': subjects,
    'Age': ages,
    'Sex': prime_sex,
    'Age_bin':age_bin,
    'Dx' : dx,
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
    'Dx': dx_flat,
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
    'Dx': dx_flat,
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
        'Dx': np.tile(np.repeat(dx, n_freqs), num_reg),
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

fig, ax = plt.subplots(figsize = (11.25,10)) # make square somehow

    
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
_shade(ax, -np.inf, 6,  col_redpurple)  # from min to 6
_shade(ax, 6,  13, col_purple)         # 6–13
_shade(ax, 13, 30, col_green)          # 13–30
_shade(ax, 30, 55, col_orange)         # 30–55
_shade(ax, 65, 80, col_red)            # 65–80
#_shade(ax, 55, 65, col_grey)            # 65–80
_shade(ax, 55, 65, color="#ffffff", hatch='////', zorder=0, hatch_ec="#000000", hatch_lw=0)

        
# Lineplot  , hue = 'Age_bin'
#sns.lineplot(df_plot, x = "Frequency", y = yval, errorbar=('ci',95), estimator='mean',color = 'black', linewidth = 4)
# Lineplot
sns.lineplot(df_plot, x = "Frequency", y = yval, errorbar=('ci',95), estimator='mean', hue = 'Dx', hue_order=["TDC", "ASD"], palette = 'bright', linewidth = 4)
#sns.lineplot(df_plot, x = "Frequency", y = 'Spectrum', errorbar=('ci', 95), estimator='mean', color = 'darkslategrey', linewidth = 4)


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
#ax.get_legend().remove()


# Grid and layout
plt.grid(False)
plt.tight_layout()


filepath = os.path.join(base_dir, r'psd',fr'{logtxt}scale_{band_param}_avg{y_param}.png')
#plt.savefig(filepath)
plt.show()


# %% 4 - sepa

sns.set(style = 'white')

band_param = 'low' #'all', 'low', 'gamma'
y_param = 'Spectrum'
log = True

fig, ax = plt.subplots(figsize = (11.25,10)) # make square somehow

    
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
_shade(ax, -np.inf, 6,  col_redpurple)  # from min to 6
_shade(ax, 6,  13, col_purple)         # 6–13
_shade(ax, 13, 30, col_green)          # 13–30
_shade(ax, 30, 55, col_orange)         # 30–55
_shade(ax, 65, 80, col_red)            # 65–80
#_shade(ax, 55, 65, col_grey)            # 65–80
_shade(ax, 55, 65, color="#ffffff", hatch='////', zorder=0, hatch_ec="#000000", hatch_lw=0)

        
# Lineplot  , hue = 'Age_bin'
#sns.lineplot(df_plot, x = "Frequency", y = yval, errorbar=('ci',95), estimator='mean',color = 'black', linewidth = 4)
# Lineplot
# Plot each ASD participant
for pid, df_sub in df_plot[df_plot['Dx'] == 'ASD'].groupby('Subject'):
    sns.lineplot(data=df_sub, x="Frequency", y=yval, color='orange', alpha=0.7, linewidth=1.5)

# Plot each TDC participant
for pid, df_sub in df_plot[df_plot['Dx'] == 'TDC'].groupby('Subject'):
    sns.lineplot(data=df_sub, x="Frequency", y=yval, color='steelblue', alpha=0.7, linewidth=1.5)

# Optionally add mean curves on top for clarity
sns.lineplot(data=df_plot[df_plot['Subject'] == 'ASD033'], x="Frequency", y=yval, color='orange', linewidth=5, label='ASD Mean')
sns.lineplot(data=df_plot[df_plot['Dx'] == 'TDC'], x="Frequency", y=yval, color='steelblue', linewidth=5, label='TDC Mean')

#sns.lineplot(df_plot, x = "Frequency", y = 'Spectrum', errorbar=('ci', 95), estimator='mean', color = 'darkslategrey', linewidth = 4)


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


# Grid and layout
plt.grid(False)
plt.tight_layout()


#filepath = os.path.join(base_dir, r'psd',fr'{logtxt}scale_{band_param}_avg{y_param}.png')
#plt.savefig(filepath)
plt.show()

# %% 4.1 - t-testing PSDs
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# ==== CONFIG ====
df_source   = df_psd_low          # use df_psd_all for 3–80 Hz; here 3–30 Hz
ycol        = 'PSD'               # or 'Spectrum', 'PSD', 'ap_PSD'
use_log10   = False                # log-transform PSD before t-tests (often recommended)
exclude_ids = []                  # e.g., ['ASD023','ASD019','TDC112','TDC044']
average_visits_per_subject = True # collapse multiple visits per subject to a single mean

# ---- prep ----
df = df_source.copy()
if exclude_ids:
    df = df[~df['Subject'].isin(exclude_ids)]

# Keep only the value column we test
df = df[['Subject','Dx','Age','Frequency', ycol]].rename(columns={ycol: 'Value'})

# Optional: log10-transform (adds tiny epsilon for safety)
if use_log10:
    df['Value'] = np.log10(df['Value'] + 1e-20)

# Optional: average repeated visits per Subject *per frequency*
if average_visits_per_subject:
    df = (df
          .groupby(['Subject','Dx','Frequency'], as_index=False)['Value']
          .mean())

# ---- per-frequency Welch t-test ----
rows = []
for f, subdf in df.groupby('Frequency', sort=True):
    asd_vals = subdf.loc[subdf['Dx'] == 'ASD', 'Value'].to_numpy()
    tdc_vals = subdf.loc[subdf['Dx'] == 'TDC', 'Value'].to_numpy()
    if len(asd_vals) >= 2 and len(tdc_vals) >= 2:
        tstat, pval = ttest_ind(asd_vals, tdc_vals, equal_var=False, nan_policy='omit')
        # Cohen's d (using pooled SD, unequal n OK)
        n1, n2 = len(asd_vals), len(tdc_vals)
        m1, m2 = np.nanmean(asd_vals), np.nanmean(tdc_vals)
        s1, s2 = np.nanstd(asd_vals, ddof=1), np.nanstd(tdc_vals, ddof=1)
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
        d = (m1 - m2) / sp if sp > 0 else np.nan

        rows.append({
            'Frequency': f,
            'n_ASD': n1, 'n_TDC': n2,
            'mean_ASD': m1, 'mean_TDC': m2,
            'diff_ASD_minus_TDC': m1 - m2,
            't': tstat, 'p': pval, 'cohen_d': d
        })

results = pd.DataFrame(rows).sort_values('Frequency')

# FDR across frequencies
rej, p_fdr, _, _ = multipletests(results['p'].values, alpha=0.05, method='fdr_bh')
results['p_FDR'] = p_fdr
results['sig_FDR_0.05'] = rej

# If you log10’d, report differences in log10-units; to interpret as ratios:
# ratio = 10**(diff_ASD_minus_TDC)
results.head(), results[results['sig_FDR_0.05']]

figA, axa = plt.subplots(figsize=(10,10))

freq = results['Frequency'].to_numpy()
tval = results['t'].to_numpy()

# Mask of FDR-significant bins (q ≤ 0.05). If you prefer, use: (results['p_FDR'] <= 0.05).to_numpy()
sig = results['sig_FDR_0.05'].to_numpy() if 'sig_FDR_0.05' in results.columns else (results['p_FDR'] <= 0.05).to_numpy()

# Base line + all points (non-sig in gray)
axa.plot(freq, tval, color='0.75', linewidth=1.5)
axa.scatter(freq[~sig], tval[~sig], s=70, color='r', alpha=0.8, label='NS (FDR>0.05)')

# Significant points, split by sign
pos = sig & (tval >= 0)
neg = sig & (tval < 0)

axa.scatter(freq[pos], tval[pos], s=90, color='tab:orange',  edgecolor='k', linewidth=0.5,
            label='FDR q≤0.05 (ASD > TDC)')
axa.scatter(freq[neg], tval[neg], s=90, color='tab:blue', edgecolor='k', linewidth=0.5,
            label='FDR q≤0.05 (ASD < TDC)')

# Cosmetics
ylim = 1.05 * np.nanmax(np.abs(tval))
axa.set_ylim([-ylim, ylim])
axa.axhline(0, color='0.8', linewidth=1)
axa.tick_params(axis='both', labelsize=20)
axa.set_ylabel('t', fontsize=30, rotation=0)
axa.set_xlabel('Frequency', fontsize=30)
axa.legend(frameon=False, fontsize=12, loc='best')

plt.show()


# %% 5 - Calculate and t-test Slope/Offset

import numpy as np
from scipy.stats import ttest_ind

# Ensure Dx is categorical (if it's object/string, that's fine)
# Filter by diagnosis
asd_df = df_fooof_params_reg[df_fooof_params_reg['Dx'] == 'ASD']
tdc_df = df_fooof_params_reg[df_fooof_params_reg['Dx'] == 'TDC']

regions = df_fooof_params_reg['Region'].unique()
regions.sort()  # optional: to keep a consistent order

# Arrays to hold results
asd_mean = np.zeros(len(regions))
tdc_mean = np.zeros(len(regions))
ttest_p  = np.zeros(len(regions))
ttest_t  = np.zeros(len(regions))

for i, reg in enumerate(regions):
    # select slope values for this region
    asd_slopeoff = asd_df.loc[asd_df['Region'] == reg, 'Offset']
    tdc_slopeoff = tdc_df.loc[tdc_df['Region'] == reg, 'Offset']

    # mean slope per region per group
    asd_mean[i] = asd_slopeoff.mean()
    tdc_mean[i] = tdc_slopeoff.mean()

    # independent-samples t-test
    t_stat, p_val = ttest_ind(asd_slopeoff, tdc_slopeoff, equal_var=False, nan_policy='omit')
    ttest_p[i] = p_val
    ttest_t[i] = t_stat


from statsmodels.stats.multitest import multipletests

# Benjamini–Hochberg FDR correction
reject, p_fdr, _, _ = multipletests(ttest_p, alpha=0.05, method='fdr_bh')


t_map_p = np.where(ttest_p<=0.05, ttest_t, 0)
t_map_fdr = np.where(reject,ttest_t, 0)


# %% 5.1 - Brain Plot Slope/Offset and tstat from above

apdatmin = np.min(np.hstack([asd_mean - min(asd_mean), tdc_mean - min(tdc_mean)]))
apdatmax = np.max(np.hstack([asd_mean - min(asd_mean), tdc_mean - min(tdc_mean)]))

labapdatmin = np.min(np.hstack([asd_mean, tdc_mean]))
labapdatmax = np.max(np.hstack([asd_mean, tdc_mean]))

fig1 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), asd_mean - min(asd_mean)),
                          r'C:\mne\fs', surf = "pial", fade = True,
                          cmap = "BuGn", 
                          symmetric=False,
                          cbar_label=fr"Offset ASD", datmin = apdatmin, datmax = apdatmax 
                          , labdatmin = labapdatmin, labdatmax = labapdatmax
                          ,setmin= True, setmax=True
                          #,threshold = 0.1
                          )
#fig1.savefig(os.path.join(base_dir,'slopeoff','ASD_Offset_bp.png'))

fig2 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), tdc_mean - min(tdc_mean)),
                          r'C:\mne\fs', surf = "pial", fade = True,
                          cmap = "BuGn", 
                          symmetric=False,
                          cbar_label=fr"Offset TDC", datmin = apdatmin, datmax = apdatmax
                          , labdatmin = labapdatmin, labdatmax = labapdatmax 
                          ,setmin= True, setmax=True
                          #,threshold = 0.1
                          )
#fig2.savefig(os.path.join(base_dir,'slopeoff','TDC_Offset_bp.png'))

fig3 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), t_map_p),
                          r'C:\mne\fs', surf = "pial", fade = True,
                          cmap = "coolwarm", 
                          symmetric=True,
                          cbar_label=fr"Offset t$_{{FDR}}$", datmin = np.min(t_map_fdr), datmax = -np.min(t_map_fdr)
                          #, labdatmin = -np.max(t_map_fdr), labdatmax = np.max(t_map_fdr) 
                          #cbar_label=fr"$\alpha$ - ASD", datmin = 0.00, datmax = 0.01
                          #,setmin= True, setmax=True
                          )
#fig3.savefig(os.path.join(base_dir,'slopeoff','ASDvTDC_Offset_t.png'))




# %% 6 - Determine t test for different bands for all regions

from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

# Expect these to already exist:
# - df_psd_reg_band  (columns: ['Region','Dx','Theta','Alpha','Beta','LowGamma','HighGamma', ...])
# - reg_name         (length = num_reg)
# - fdr_nan_safe     (your helper returning p_corr, reject_mask)

# Ensure Dx is clean strings
df_forthis = df_psd_reg_band.copy()  #>>>>>>>>df_psd_reg_band.copy() for fooof'd data
df_forthis['Dx'] = df_forthis['Dx'].astype(str).str.strip()

# Regions (use your atlas order)
regions = list(reg_name)
num_reg = len(regions)

# Pre-allocate arrays
theta_pval   = np.full(num_reg, np.nan)
theta_tstat  = np.full(num_reg, np.nan)

alpha_pval   = np.full(num_reg, np.nan)
alpha_tstat  = np.full(num_reg, np.nan)

beta_pval    = np.full(num_reg, np.nan)
beta_tstat   = np.full(num_reg, np.nan)

lowgamma_pval  = np.full(num_reg, np.nan)
lowgamma_tstat = np.full(num_reg, np.nan)

highgamma_pval  = np.full(num_reg, np.nan)
highgamma_tstat = np.full(num_reg, np.nan)

def welch_t_for_region(df, region, band_col):
    """Return (t, p) for ASD vs TDC at a single region and band using Welch t-test."""
    sub = df[df['Region'] == region]
    a = sub.loc[sub['Dx'] == 'ASD', band_col].to_numpy(dtype=float)
    b = sub.loc[sub['Dx'] == 'TDC', band_col].to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    t, p = ttest_ind(a, b, equal_var=False, nan_policy='omit')
    return float(t), float(p)

def fdr_nan_safe(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR while ignoring NaNs.
    Returns: corrected pvals (NaNs preserved), reject mask (False where NaN).
    """
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    p_corr = np.full_like(p, np.nan, dtype=float)
    reject = np.zeros_like(p, dtype=bool)
    if finite.sum() > 0:
        rej, p_c, _, _ = multipletests(p[finite], alpha=alpha, method='fdr_bh')
        p_corr[finite] = p_c
        reject[finite] = rej
    return p_corr, reject

# Loop regions once; compute for all bands
for i, reg in enumerate(regions):
    # Theta
    t, p = welch_t_for_region(df_forthis, reg, 'Theta')
    theta_tstat[i], theta_pval[i] = t, p
    # Alpha
    t, p = welch_t_for_region(df_forthis, reg, 'Alpha')
    alpha_tstat[i], alpha_pval[i] = t, p
    # Beta
    t, p = welch_t_for_region(df_forthis, reg, 'Beta')
    beta_tstat[i], beta_pval[i] = t, p
    # LowGamma
    t, p = welch_t_for_region(df_forthis, reg, 'LowGamma')
    lowgamma_tstat[i], lowgamma_pval[i] = t, p
    # HighGamma
    t, p = welch_t_for_region(df_forthis, reg, 'HighGamma')
    highgamma_tstat[i], highgamma_pval[i] = t, p

# --- FDR across regions (per band) ---
theta_pval_fdr,   theta_reject   = fdr_nan_safe(theta_pval,   alpha=0.05)
alpha_pval_fdr,   alpha_reject   = fdr_nan_safe(alpha_pval,   alpha=0.05)
beta_pval_fdr,    beta_reject    = fdr_nan_safe(beta_pval,    alpha=0.05)
lowgamma_pval_fdr,lowgamma_reject= fdr_nan_safe(lowgamma_pval,alpha=0.05)
highgamma_pval_fdr,highgamma_reject=fdr_nan_safe(highgamma_pval,alpha=0.05)

# --- Mask t-stats by FDR significance (keep sign/magnitude where significant, else 0) ---
theta_tstat_masked     = theta_tstat.copy()
theta_tstat_masked[~theta_reject] = 0

alpha_tstat_masked     = alpha_tstat.copy()
alpha_tstat_masked[~alpha_reject] = 0

beta_tstat_masked      = beta_tstat.copy()
beta_tstat_masked[~beta_reject] = 0

lowgamma_tstat_masked  = lowgamma_tstat.copy()
lowgamma_tstat_masked[~lowgamma_reject] = 0

highgamma_tstat_masked = highgamma_tstat.copy()
highgamma_tstat_masked[~highgamma_reject] = 0

# Optional: quick table of top regions per band (uncorrected p)
def top_table(pvals, tstats, regions, k=5):
    df = pd.DataFrame({"Region": regions, "t": tstats, "p": pvals})
    return df.sort_values("p", na_position="last").head(k)

print("Top Theta:\n", top_table(theta_pval, theta_tstat, regions))
print("Top Alpha:\n", top_table(alpha_pval, alpha_tstat, regions))
print("Top Beta:\n",  top_table(beta_pval,  beta_tstat,  regions))
print("Top LowG:\n",  top_table(lowgamma_pval, lowgamma_tstat, regions))
print("Top HighG:\n", top_table(highgamma_pval, highgamma_tstat, regions))

# %% 6.1 - Brain Plot the t-stat for each band from above
fig_theta = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), theta_tstat_masked),
                                 r"C:\mne\fs", surf="pial", fade=True, cort=(0.6,0.6,0.6),
                                 cmap="coolwarm", symmetric=True, cbar_label=fr"$\theta$ t-stat",
                                 datmin=-np.nanmax(np.abs(theta_tstat_masked)),
                                 datmax= np.nanmax(np.abs(theta_tstat_masked)))
#fig_theta.savefig(os.path.join(base_dir,'psd_bands','theta_t_bp.png'))


#adjusted = alpha_tstat_masked - np.sign(alpha_tstat_masked) * np.min(np.abs(alpha_tstat_masked[np.nonzero(alpha_tstat_masked)]))

fig_alpha = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), alpha_tstat_masked),
                                 r"C:\mne\fs", surf="pial", fade=True, cort=(0.9,0.9,0.9),
                                 cmap="coolwarm", symmetric=True, cbar_label=fr"$\alpha$ t-stat",
                                 datmin=-np.nanmax(np.abs(alpha_tstat_masked)),
                                 datmax= np.nanmax(np.abs(alpha_tstat_masked)) 
                                 ,threshold=1.5
                                 )
fig_alpha.savefig(os.path.join(base_dir,'psd_bands','rawalpha_t_bp.png'))



fig_beta = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), beta_tstat_masked),
                                r"C:\mne\fs", surf="pial", fade=True, cort=(0.6,0.6,0.6),
                                cmap="coolwarm", symmetric=True, cbar_label=fr"$\beta$ t-stat",
                                datmin=-np.nanmax(np.abs(beta_tstat_masked)),
                                datmax= np.nanmax(np.abs(beta_tstat_masked)))
#fig_beta.savefig(os.path.join(base_dir,'psd_bands','beta_t_bp.png'))



fig_lowgamma = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), lowgamma_tstat_masked),
                                    r"C:\mne\fs", surf="pial", fade=True, cort=(0.9,0.9,0.9),
                                    cmap="coolwarm", symmetric=True, cbar_label=fr"$\gamma$ 1 t-stat",
                                    datmin=-np.nanmax(np.abs(lowgamma_tstat_masked)),
                                    datmax= np.nanmax(np.abs(lowgamma_tstat_masked)))
#fig_lowgamma.savefig(os.path.join(base_dir,'psd_bands','lowgamma_t_bp.png'))



fig_highgamma = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), highgamma_tstat_masked),
                                     r"C:\mne\fs", surf="pial", fade=True, cort=(0.9,0.9,0.9),
                                     cmap="coolwarm", symmetric=True, cbar_label=fr"$\gamma$ 2 t-stat",
                                     datmin=-np.nanmax(np.abs(highgamma_tstat_masked)),
                                     datmax= np.nanmax(np.abs(highgamma_tstat_masked)))
#fig_highgamma.savefig(os.path.join(base_dir,'psd_bands','highgamma_t_bp.png'))


# %% prem testing.... irrelevant

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Build subject × region matrix for the band
theta_mat = (
    df_psd_reg_band.pivot_table(index='Subject', columns='Region', values='Theta')
    .sort_index()
)
subjects = theta_mat.index.to_numpy()
regions  = theta_mat.columns.to_numpy()

# Subject-level Dx vector aligned to theta_mat
dx_map = (
    df_psd_reg_band.drop_duplicates('Subject')[['Subject','Dx']]
    .set_index('Subject').loc[subjects, 'Dx'].to_numpy()
)

# Real t per region (Welch)
asd_mask = (dx_map == 'ASD')
tdc_mask = (dx_map == 'TDC')

theta_real = np.empty(len(regions), dtype=float)
for j, reg in enumerate(regions):
    a = theta_mat.iloc[asd_mask, j].to_numpy()
    b = theta_mat.iloc[tdc_mask, j].to_numpy()
    t, _ = ttest_ind(a, b, equal_var=False, nan_policy='omit')
    theta_real[j] = t

# Permutation null of max |t|
n_perm = 5000
rng = np.random.default_rng(42)
max_t_null = np.empty(n_perm, dtype=float)

for k in range(n_perm):
    # shuffle Dx across subjects
    dx_perm = rng.permutation(dx_map)
    a_mask = (dx_perm == 'ASD')
    b_mask = (dx_perm == 'TDC')

    # compute t per region under permuted labels
    t_perm = np.empty(len(regions), dtype=float)
    for j in range(len(regions)):
        a = theta_mat.iloc[a_mask, j].to_numpy()
        b = theta_mat.iloc[b_mask, j].to_numpy()
        t, _ = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        t_perm[j] = t

    max_t_null[k] = np.nanmax(np.abs(t_perm))

# Family-wise corrected p for each region via max-T
p_values = (np.abs(theta_real)[:, None] <= np.abs(max_t_null)[None, :]).mean(axis=1)

# Optional: tidy summary
theta_perm_table = pd.DataFrame({
    'Region': regions,
    't_real': theta_real,
    'p_maxT': p_values
}).sort_values('p_maxT')
print(theta_perm_table.head(10))


# %% plotting above result - irrelevant

dat = theta_perm_table['t_real']
fig3 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), dat),
                          r'C:\mne\fs', surf = "pial", fade = True,cort = (0.6,0.6,0.6),
                          cmap = "coolwarm", 
                          symmetric=True,
                          cbar_label=fr"t_real", datmin = -np.max(dat), datmax = np.max(dat)
                          #cbar_label=fr"$\alpha$ - ASD", datmin = 0.00, datmax = 0.01
                          #,setmin= True, setmax=True
                          )

dat2 = theta_perm_table['p_maxT']
fig4 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), dat2),
                          r'C:\mne\fs', surf = "pial", fade = True,cort = (0.6,0.6,0.6),
                          cmap = "coolwarm", 
                          symmetric=True,
                          cbar_label=fr"p_maxT", datmin = -np.max(dat2), datmax = np.max(dat2)
                          #cbar_label=fr"$\alpha$ - ASD", datmin = 0.00, datmax = 0.01
                          #,setmin= True, setmax=True
                          )

# %% 7 - Brain Plot for Average PSD

from functions.brainplots import make_4d_atlas_nifti, surface_brain_plotv2
from scipy.stats import zscore

# Group slices
asd = slice(0, 30)    # subjects 0..29
tdc = slice(30, 60)   # subjects 30..59

test = True
if test:

    # ---- From fooof_periodic_psd (periodic component) ----
    theta_psd_ind_asd    = np.mean(fooof_periodic_psd[asd,  0:5,  :], axis=1)
    theta_psd_ind_tdc    = np.mean(fooof_periodic_psd[tdc,  0:5,  :], axis=1)
    
    alpha_psd_ind_asd    = np.mean(fooof_periodic_psd[asd,  5:17, :], axis=1)
    alpha_psd_ind_tdc    = np.mean(fooof_periodic_psd[tdc,  5:17, :], axis=1)
    
    beta_psd_ind_asd     = np.mean(fooof_periodic_psd[asd, 17:46, :], axis=1)
    beta_psd_ind_tdc     = np.mean(fooof_periodic_psd[tdc, 17:46, :], axis=1)
    
    lo_beta_psd_ind_asd  = np.mean(fooof_periodic_psd[asd, 17:30, :], axis=1)
    lo_beta_psd_ind_tdc  = np.mean(fooof_periodic_psd[tdc, 17:30, :], axis=1)
    
    hi_beta_psd_ind_asd  = np.mean(fooof_periodic_psd[asd, 30:46, :], axis=1)
    hi_beta_psd_ind_tdc  = np.mean(fooof_periodic_psd[tdc, 30:46, :], axis=1)
    
    # ---- From prime_psd (raw PSD) ----
    theta_psd_raw_ind_asd   = np.mean(prime_psd[asd,  6:11,  :], axis=1)
    theta_psd_raw_ind_tdc   = np.mean(prime_psd[tdc,  6:11,  :], axis=1)
    
    alpha_psd_raw_ind_asd   = np.mean(prime_psd[asd, 11:23,  :], axis=1)
    alpha_psd_raw_ind_tdc   = np.mean(prime_psd[tdc, 11:23,  :], axis=1)
    
    beta_psd_raw_ind_asd    = np.mean(prime_psd[asd, 21:52,  :], axis=1)
    beta_psd_raw_ind_tdc    = np.mean(prime_psd[tdc, 21:52,  :], axis=1)
    
    lo_beta_psd_raw_ind_asd = np.mean(prime_psd[asd, 21:36,  :], axis=1)
    lo_beta_psd_raw_ind_tdc = np.mean(prime_psd[tdc, 21:36,  :], axis=1)
    
    hi_beta_psd_raw_ind_asd = np.mean(prime_psd[asd, 36:52,  :], axis=1)
    hi_beta_psd_raw_ind_tdc = np.mean(prime_psd[tdc, 36:52,  :], axis=1)
    
    lowgamma_psd_ind_asd    = np.mean(prime_psd[asd, 52:94,  :], axis=1)
    lowgamma_psd_ind_tdc    = np.mean(prime_psd[tdc, 52:94,  :], axis=1)
    
    highgamma_psd_ind_asd   = np.mean(prime_psd[asd,111:137, :], axis=1)
    highgamma_psd_ind_tdc   = np.mean(prime_psd[tdc,111:137, :], axis=1)
    

band = 'beta'
psd_asd = np.mean(beta_psd_raw_ind_asd, axis = 0)
psd_tdc= np.mean(beta_psd_raw_ind_tdc, axis = 0)
datamin = np.min(np.hstack([psd_asd, psd_tdc]))
datamax = np.max(np.hstack([psd_asd, psd_tdc]))

apdatmin = np.min(np.hstack([psd_asd - min(psd_asd), psd_tdc - min(psd_tdc)]))
apdatmax = np.max(np.hstack([psd_asd - min(psd_asd), psd_tdc - min(psd_tdc)]))


#Ok, plotting the brain 
fig9 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), psd_asd - np.min(psd_asd)),
                          r'C:\mne\fs', surf = "pial", fade = True,setmax = True, setmin = True,
                          cmap = "Greens", 
                          symmetric=False,
                          cbar_label=fr"$\{band}$ ASD (Hz$^{{-1}}$)", datmin = apdatmin, datmax = apdatmax #$\{band}$$\gamma_2$
                          #,threshold = 0.00005
                          , labdatmin = datamin, labdatmax = datamax
                          )
fig9.savefig(os.path.join(base_dir,'psd_bands',fr'rawASD_mean_{band}_bp.png'))


fig10 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), psd_tdc - np.min(psd_tdc)),
                          r'C:\mne\fs', surf = "pial", fade = True,setmax = True, setmin = True,
                          cmap = "Greens", 
                          symmetric=False,
                          cbar_label=fr"$\{band}$ TDC (Hz$^{{-1}}$)", datmin = apdatmin, datmax = apdatmax #$\{band}$$\gamma_2$
                          #,threshold = 0.00005
                          , labdatmin = datamin, labdatmax = datamax
                          )
fig10.savefig(os.path.join(base_dir,'psd_bands',fr'rawTDC_mean_{band}_bp.png'))



# %% 7.1 - Violin plot PSD values for spcified band and region w/ tstat (not FDR)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# -----------------------------
# Helpers
# -----------------------------
def group_vectors(arr_asd, arr_tdc, region=None, agg='mean'):
    """
    arr_*: shape (n_subjects, n_regions)
    region: None -> aggregate across regions per subject; int -> single region index
    agg: 'mean' or 'median' when region is None
    Returns: (asd_vec, tdc_vec) 1D arrays (length = subjects per group)
    """
    if region is None:
        if agg == 'median':
            asd_vec = np.nanmedian(arr_asd, axis=1)
            tdc_vec = np.nanmedian(arr_tdc, axis=1)
        else:
            asd_vec = np.nanmean(arr_asd, axis=1)
            tdc_vec = np.nanmean(arr_tdc, axis=1)
    else:
        if not (0 <= region < arr_asd.shape[1] and 0 <= region < arr_tdc.shape[1]):
            raise IndexError(f"region {region} is out of bounds for input arrays.")
        asd_vec = arr_asd[:, region]
        tdc_vec = arr_tdc[:, region]
    return asd_vec, tdc_vec

def cohen_d(x, y):
    """
    Cohen's d using pooled SD, NaN-safe, unequal n supported.
    Returns NaN if either group has <2 finite values or pooled SD is 0/NaN.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    
    mx, my = np.mean(x), np.mean(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    
    df = nx + ny - 2
    if df <= 0:
        return np.nan
    
    sp2 = ((nx - 1) * sx2 + (ny - 1) * sy2) / df
    if not np.isfinite(sp2) or sp2 <= 0:
        return np.nan
    
    sp = np.sqrt(sp2)
    return (mx - my) / sp


def welch_t_p(x, y):
    """Welch t-test with NaNs removed in a paired-wise way (independent groups)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    t, p = ttest_ind(x, y, equal_var=False, nan_policy='omit')
    return t, p

def p_to_stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "ns" if p >= 0.05 else ("*" if p >= 0.01 else ("**" if p >= 0.001 else ("***" if p >= 1e-4 else "****")))

def _jitter(n, width=0.08, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    return rng.uniform(-width, width, size=n)

# -----------------------------
# Main plotting function
# -----------------------------
def plot_group_compare(asd_vec, tdc_vec, title="Band (across regions)", kind="violin",
                       show_strip=True, strip_alpha=0.5, strip_size=28, jitter_width=0.08,
                       ylim=None, seed=42, ylabel = "Band power (a.u.)"):
    asd_clean = np.asarray(asd_vec, float)
    tdc_clean = np.asarray(tdc_vec, float)
    asd_clean = asd_clean[np.isfinite(asd_clean)]
    tdc_clean = tdc_clean[np.isfinite(tdc_clean)]

    # Welch t-test: ASD vs TDC
    t, p = ttest_ind(asd_clean, tdc_clean, equal_var=False, nan_policy='omit')
    d = cohen_d(asd_clean, tdc_clean)
    star = p_to_stars(p)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    data = [asd_clean, tdc_clean]
    positions = [1, 2]

    if kind.lower() == "violin":
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False)
        parts['cmeans'].set_color("black")
        ax.set_xticks(positions, ["ASD", "TDC"])
    else:
        bp = ax.boxplot(data, positions=positions, patch_artist=True, labels=["ASD", "TDC"], showmeans=True)
        colors = ["#ff7c00", "#023eff"]
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

    # Overlay scatter points
    if show_strip:
        x1 = np.full(len(asd_clean), positions[0]) + _jitter(len(asd_clean), jitter_width, seed)
        x2 = np.full(len(tdc_clean), positions[1]) + _jitter(len(tdc_clean), jitter_width, seed+1)
        ax.scatter(x1, asd_clean, s=strip_size, alpha=strip_alpha, edgecolor='none', color="orange")
        ax.scatter(x2, tdc_clean, s=strip_size, alpha=strip_alpha, edgecolor='none', color="steelblue")

    ax.set_ylabel(ylabel)
    #ax.set_yscale('log')
    #ax.set_title(f"{title}\nWelch t={t:.2f}  p={p:.3g} ({star})  d={d:.2f}")
    m_asd = np.nanmean(asd_clean)
    m_tdc = np.nanmean(tdc_clean)
    
    ax.set_title(
        f"{title}\n"
        f"ASD mean={m_asd:.4f}, TDC mean={m_tdc:.4f}\n"
        f"Welch t={t:.2f}  p={p:.3g} ({star})  d={d:.2f}"
    )

    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    plt.show()

    return {"t": t, "p": p, "d": d, "n_asd": len(asd_clean), "n_tdc": len(tdc_clean)}, fig


# -----------------------------
# Example usage
# -----------------------------
# Suppose your inputs are shaped (subjects, regions):
#   theta_psd_ind_asd: (N_asd, R)
#   theta_psd_ind_tdc: (N_tdc, R)

r = 16  # region index you want to inspect
reg_oi = reg_name_full[r]
band_up = 'Alpha'

# 1) Single region comparison
asd_r, tdc_r = group_vectors(alpha_psd_raw_ind_asd, alpha_psd_raw_ind_tdc, region=r)
stats_region = plot_group_compare(
    asd_r, tdc_r,
    title=f"{band_up} — {reg_oi}",  # <— make sure this matches your data (Theta/Alpha/etc.)
    kind="box",           # 'violin' also works
    show_strip=True       # show the scatter underlay
)

stats_all, fig13 = plot_group_compare(
    asd_r, tdc_r,
    title=fr"{band_up} — {reg_oi}",
    kind="violin",
    show_strip=True
)
fig13.savefig(os.path.join(base_dir,'psd_bands',fr'raw{band}_{reg_name[r]}.png'))

# %%  8 - Let's do something about some peaks

band_limsap = {
    'theta': (0, 5),
    'alpha': (5, 17),
    'beta': (17, 46),
    'lowgamma': (52, 94),
    'highgamma': (111, 137)
}

# loop over subjects
for i in range(30):
    # choose which band you want to plot
    band = 'beta'
    lo, hi = band_limsap[band]   # unpack start and stop indices
    plt.plot(freq_bins_low[lo:hi],fooof_periodic_psd[i, lo:hi,18], color = 'r')
    plt.plot(freq_bins_low[lo:hi],fooof_periodic_psd[i+30, lo:hi,18], color = 'b')


# %%

import numpy as np

band_limsap = {
    'theta': (0, 5),
    'alpha': (5, 17),
    'beta': (17, 46),
    'lowgamma': (52, 94),
    'highgamma': (111, 137),
}

band = 'theta'   # choose band
bandl = band.capitalize()
lo, hi = band_limsap[band]
band_slice = slice(lo, hi)

# PSD in the band: (n_ptp, band_len, n_regions)
band_psd = fooof_periodic_psd[:, band_slice, :]

# NaN-safe argmax
all_nan = np.all(np.isnan(band_psd), axis=1)                  # (n_ptp, n_regions)
band_psd_filled = np.where(np.isnan(band_psd), -np.inf, band_psd)
peak_idx_in_band = np.argmax(band_psd_filled, axis=1)         # (n_ptp, n_regions)

# Peak amplitudes
peak_amp = np.take_along_axis(
    band_psd, peak_idx_in_band[:, None, :], axis=1
).squeeze(1)                                                  # (n_ptp, n_regions)

# Peak frequencies from your freq array
freq_band = freq_bins_low[band_slice]                         # (band_len,)
peak_freq = freq_band[peak_idx_in_band]                       # (n_ptp, n_regions)

# Restore NaNs where the whole band is NaN
peak_amp[all_nan] = np.nan
peak_freq[all_nan] = np.nan

# Split into ASD (0–29) and TDC (30–59)
asd = slice(0, 30)
tdc = slice(30, 60)
band_peaks = {
    'asd': {'amp': peak_amp[asd], 'freq': peak_freq[asd]},    # (30, n_regions)
    'tdc': {'amp': peak_amp[tdc], 'freq': peak_freq[tdc]},    # (30, n_regions)
}

# Example for a single region
r = 30
reg_oi = reg_name_full[r]
amp_r  = peak_amp[:, r]
freq_r = peak_freq[:, r]
asd_amp_r, tdc_amp_r = amp_r[asd], amp_r[tdc]
asd_freq_r, tdc_freq_r = freq_r[asd], freq_r[tdc]

# Plot comparisons
amps, fig14 = plot_group_compare(
    asd_amp_r, tdc_amp_r,
    title=fr"{bandl} Amplitude — {reg_oi}",
    kind="violin",
    show_strip=True, ylabel = fr'Peak $\{band}$ Power (Hz$^{{-1}}$)'
)

freak, fig15 = plot_group_compare(
    asd_freq_r, tdc_freq_r,
    title=fr"Peak {bandl} Frequency — {reg_oi}",
    kind="violin",
    show_strip=True, ylabel = fr'Peak $\{band}$ Freq (Hz)'
)

asd = fooof_periodic_psd[0:30, band_slice, r]   # shape (30, n_freqs)
tdc = fooof_periodic_psd[30:, band_slice, r]    # shape (30, n_freqs)
freqs = freq_bins_low[band_slice]

# Mean and 95% CI
asd_mean = np.mean(asd, axis=0)
tdc_mean = np.mean(tdc, axis=0)

asd_ci = 1.96 * np.std(asd, axis=0) / np.sqrt(asd.shape[0])
tdc_ci = 1.96 * np.std(tdc, axis=0) / np.sqrt(tdc.shape[0])

fig, ax = plt.subplots(figsize=(11.25, 10))

ax.plot(freqs, asd_mean, color='#ff7c00', linewidth=4, label='ASD')
ax.fill_between(freqs, asd_mean - asd_ci, asd_mean + asd_ci, alpha=0.3, color='#ff7c00')

ax.plot(freqs, tdc_mean, color='#023eff', linewidth=4, label='TDC')
ax.fill_between(freqs, tdc_mean - tdc_ci, tdc_mean + tdc_ci, alpha=0.3, color='#023eff')


_shade(ax, -np.inf, 6,  col_redpurple)  # from min to 6
_shade(ax, 6,  13, col_purple)         # 6–13
_shade(ax, 13, 30, col_green)          # 13–30

ax.set_xlabel("Frequency (Hz) ", fontsize=70)
ax.set_ylabel(r'Power (Hz$^{{-1}}$)', fontsize=70) # (log $(\frac{fT^2}{Hz})$)
ax.tick_params(axis='both', which='major', labelsize=70)
ax.set_xlim([min(freq_bins_low[band_slice]), max(freq_bins_low[band_slice])])
ax.legend()
plt.show()


# %%
from scipy.signal import find_peaks
import numpy as np

def find_band_peaks_k(
    band_psd,                 # (n_ptp, band_len, n_regions)
    freq_band,                # (band_len,)
    prefer_hz=None,           # (lo, hi) to prefer; if None, use full band
    min_prominence=0.0,       # filter tiny bumps
    min_distance_bins=None,   # enforce spacing in bins, e.g., 2–3
    top_k=2                   # number of peaks to keep
):

    n_ptp, band_len, n_regions = band_psd.shape
    peak_amp  = np.full((n_ptp, n_regions, top_k), np.nan, float)
    peak_freq = np.full((n_ptp, n_regions, top_k), np.nan, float)
    peak_prom = np.full((n_ptp, n_regions, top_k), np.nan, float)

    if prefer_hz is None:
        prefer_hz = (freq_band[0], freq_band[-1])

    for i in range(n_ptp):
        for r in range(n_regions):
            y = band_psd[i, :, r]
            if np.all(np.isnan(y)):
                continue

            y_filled = np.where(np.isnan(y), -np.inf, y)
            kwargs = {"prominence": min_prominence}
            if min_distance_bins is not None:
                kwargs["distance"] = min_distance_bins

            peaks, props = find_peaks(y_filled, **kwargs)
            prominences = props.get("prominences", None)

            # If nothing qualifies, fall back to a single argmax
            if peaks.size == 0:
                k = np.nanargmax(y)
                peak_amp[i, r, 0]  = y[k]
                peak_freq[i, r, 0] = freq_band[k]
                # peak_prom remains NaN
                continue

            if prominences is None:
                # Shouldn’t happen when using 'prominence' arg, but be safe
                prominences = np.zeros(peaks.shape, dtype=float)

            f_peaks = freq_band[peaks]
            in_pref = (f_peaks >= prefer_hz[0]) & (f_peaks <= prefer_hz[1])

            # Prefer peaks inside the preferred Hz window if any exist
            if np.any(in_pref):
                peaks_use = peaks[in_pref]
                prom_use  = prominences[in_pref]
            else:
                peaks_use = peaks
                prom_use  = prominences

            # Sort candidate peaks by (prominence desc, amplitude desc) for stability
            amps_use = y[peaks_use]
            order = np.lexsort((-amps_use, -prom_use))  # secondary key: amplitude
            peaks_sorted = peaks_use[order]

            # Take up to top_k
            keep = peaks_sorted[:top_k]
            klen = keep.size

            peak_amp[i, r, :klen]  = y[keep]
            peak_freq[i, r, :klen] = freq_band[keep]
            peak_prom[i, r, :klen] = prominences[np.isin(peaks, keep, assume_unique=False)]

    return peak_amp, peak_freq, peak_prom



# Prepare band slices as you already do
band = 'theta'  # 13–30 Hz by your convention
bandl = band.capitalize()
lo, hi = band_limsap[band]
band_slice = slice(lo, hi)

band_psd  = fooof_periodic_psd[:, band_slice, :]  # (n_ptp, band_len, n_regions)
freq_band = freq_bins_low[band_slice]

# Prefer the full beta window; keep up to 2 peaks, require some spacing
peak_amp_k, peak_freq_k, peak_prom_k = find_band_peaks_k(
    band_psd, freq_band,
    prefer_hz = (freq_bins_low[lo], freq_bins_low[hi-1]),
    min_prominence=0.0,        # consider setting >0 to suppress noise
    min_distance_bins=2,       # or in Hz: convert to bins based on your freq grid
    top_k=2
)

# Example: primary vs secondary peaks for a region r
r = 30
reg_oi = reg_name_full[r]
# Correct splits
asd = slice(0, 30)
tdc = slice(30, 60)

primary_amp    = peak_amp_k[:, r, 0]
secondary_amp  = peak_amp_k[:, r, 1]
primary_freq   = peak_freq_k[:, r, 0]
secondary_freq = peak_freq_k[:, r, 1]


# For your arrays shaped (n_ptp,)
asd_primary_amp  = primary_amp[asd]
tdc_primary_amp  = primary_amp[tdc]
asd_primary_freq = primary_freq[asd]
tdc_primary_freq = primary_freq[tdc]

asd_secondary_amp  = secondary_amp[asd]
tdc_secondary_amp  = secondary_amp[tdc]
asd_secondary_freq = secondary_freq[asd]
tdc_secondary_freq = secondary_freq[tdc]


# Primary peak comparisons
amps1, fig1 = plot_group_compare(
    asd_primary_amp, tdc_primary_amp,
    title=fr"{bandl} Amplitude 1 — {reg_oi}",
    kind="violin",
    show_strip=True, ylabel = fr'Peak $\{band}$ Power (Hz$^{{-1}}$)'
)

freqs1, fig2 = plot_group_compare(
    asd_primary_freq, tdc_primary_freq,
    title=fr"Peak {bandl} Frequency 1 — {reg_oi}",
    kind="violin",
    show_strip=True, ylabel = fr'Peak $\{band}$ Freq (Hz)'
)

# Secondary peak comparisons (will drop NaNs independently per group)
amps2, fig3 = plot_group_compare(
    asd_secondary_amp, tdc_secondary_amp,
    title=fr"{bandl} Amplitude 2 — {reg_oi}",
    kind="violin",
    show_strip=True, ylabel = fr'Peak $\{band}$ Power (Hz$^{{-1}}$)'
)

freqs2, fig4 = plot_group_compare(
    asd_secondary_freq, tdc_secondary_freq,
    title=fr"Peak {bandl} Frequency 2 — {reg_oi}",
    kind="violin",
    show_strip=True, ylabel = fr'Peak $\{band}$ Freq (Hz)'
)

# %%


band_limsap = {
    'theta': (0, 5),
    'alpha': (5, 17),
    'beta': (17, 46)
}

for band in band_limsap:
    band = band  # 13–30 Hz by your convention
    bandl = band.capitalize()
    lo, hi = band_limsap[band]
    band_slice = slice(lo, hi)
    
    reg_dif = np.zeros([38,len(freq_bins_low[band_slice])])
    for i, reg in enumerate(reg_name):
        asd_val = np.mean(fooof_periodic_psd[0:30,band_slice,i], axis = 0)
        tdc_val = np.mean(fooof_periodic_psd[30:,band_slice,i], axis = 0)
        
        diff = asd_val - tdc_val
        
        reg_dif[i,:] = diff
        
    dat = np.mean(reg_dif, axis = 1)
    datmax = np.max(abs(dat))
    datmin = -datmax
    
    fig16 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), dat - np.min(0)),
                              r'C:\mne\fs', surf = "pial", fade = True,#setmax = True, setmin = True,
                              cmap = "coolwarm", 
                              symmetric=True,
                              cbar_label=fr"$\{band}$ diff (Hz$^{{-1}}$)", datmin = datmin, datmax = datmax
                              )
    fig16.savefig(os.path.join(base_dir,fr'psd_bands',fr'{band}_psd_diff_bp.png'))
    plt.figure()
    plt.plot(dat)

# %%

reg_dif = np.zeros([38,len(freq_bins_low)])
for i, reg in enumerate(reg_name):
    asd_val = np.mean(fooof_periodic_psd[0:30,:,i], axis = 0)
    tdc_val = np.mean(fooof_periodic_psd[30:,:,i], axis = 0)
    
    diff = asd_val - tdc_val
    
    reg_dif[i,:] = diff
    
dat = np.mean(reg_dif, axis = 1)
datmax = np.max(abs(dat))
datmin = -datmax

fig16 = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg3d), dat - np.min(0)),
                          r'C:\mne\fs', surf = "pial", fade = True,#setmax = True, setmin = True,
                          cmap = "coolwarm", 
                          symmetric=True,
                          cbar_label=fr"PSD diff (Hz$^{{-1}}$)", datmin = datmin, datmax = datmax
                          )
#fig16.savefig(os.path.join(base_dir,fr'psd_bands',fr'{band}_psd_diff_bp.png'))
plt.figure()
plt.plot(dat)

