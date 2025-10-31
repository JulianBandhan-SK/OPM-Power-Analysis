# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:26:51 2025

@author: Julian Bandhan
"""
from functions.common_imports import np, plt, random, stats, scipy, pg, sns, pd, loadmat, Axes3D, image, Line2D
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2
import numpy as np
atlasimg = fr"Y:/projects/OPM/opm_pipeline_templates/1-5YR_M1p5_4YO/meg38_lang.nii.gz"

%matplotlib inline

# %% Custom Color map
from matplotlib.colors import ListedColormap

values = np.linspace(0,1,38)  # 1 to 38 for 38 regions

# Use a qualitative colormap or generate your own
cmap = plt.get_cmap('tab20b')  # Has 20 distinct colors, or combine two maps
colors = [cmap(i) for i in range(20)] + [plt.get_cmap('tab20c')(i) for i in range(18)]
custom_cmap = ListedColormap(colors)


# %%
reg_name = pd.read_csv(r'Y:/projects/OPM/opm_pipeline_templates/Adult/meg38_lang.txt', header=None, delimiter='\t')[0]


data = values-0.02702703

fig1 = surface_brain_plotv3(
    make_atlas_nifti(image.load_img(atlasimg), data),
    r'C:\mne\fs',
    surf = 'pial',
    cmap=custom_cmap,
    symmetric=False,
    cbar_label=f'')
    
    #fig1.savefig(fr'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\analysis\atlas\revised_bp\{i}_{region}.png', bbox_inches = 'tight')
    #plt.close("all")
    
# %%
reg_name = pd.read_csv(r'Y:/projects/OPM/opm_pipeline_templates/Adult/meg38_lang.txt', header=None, delimiter='\t')[0]

for i in range(38,42):
    num = i+1
    
    values = np.zeros(len(reg_name))
    values[i] = 1
        
    fig = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg), values),
                              r'C:\mne\fs',# surf = "inflated",
                              cmap = "Blues",
                              symmetric=False,
                              cbar_label=fr"{reg_name[i]}", datmin = 0, datmax = 1
                              )
    
    filepath = fr'Y:\projects\OPM-Analysis\OPM_rest_babyeinsteins\analysis\meg38reg\nummed_meg38lang\{i}_{reg_name[i]}.png'
    #fig.savefig(filepath)
    #plt.close('all')