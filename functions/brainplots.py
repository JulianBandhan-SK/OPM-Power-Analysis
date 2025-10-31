# %% Brain Plot: by Dr. Sebastian Coleman
"""

Ensure that atlas_img is set to the nifti file for the atlas you want to use in your script
MEG38 - atlas_img = "Y:\projects\OPM\opm_pipeline_templates\Adult\MEGatlas_38reg\meg38_1mm.nii.gz"
AAL90 - atlas_img =  'X:\toolboxes\brainnetviewer\2019-10-31\Data\ExampleFiles\AAL90\aal.nii'

"""
import mne
import os.path as op

from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps import transforms
from surfplot import Plot, utils
import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from nilearn import image

import mne
import numpy as np
from matplotlib import pyplot as plt


def make_atlas_nifti(atlas_img, values):

    # some package imports inside function - ignore my bad practice
    from nilearn import image, datasets
    from nibabel import Nifti1Image
    
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()
    atlas_new = np.zeros(np.shape(atlas_data))
    indices = np.unique(atlas_data[atlas_data>0])
    for reg in range(len(values)):
        reg_mask = atlas_data == indices[reg]
        atlas_new[reg_mask] = values[reg]
        
    #print(np.min(atlas_new))
       
    new_img = Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    img = image.resample_img(new_img, mni.affine)
    
    return img

def make_4d_atlas_nifti(atlas_img, values):
    from nilearn import image, datasets
    from nibabel import Nifti1Image
 
    # load fsaverage and atlas   

    mni = datasets.load_mni152_template()

    atlas_data = atlas_img.get_fdata()
 
    # place values in each parcel region

    regs = []

    for reg in range(atlas_data.shape[-1]):

        atlas_reg = atlas_data[:,:,:,reg]

        atlas_reg /= np.max(atlas_reg)

        regs.append(atlas_reg * values[reg])

    atlas_new = np.sum(regs, 0)
 
 
    # make image from new atlas data

    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)

    # interpolate image

    img_interp = image.resample_img(new_img, mni.affine)

    return img_interp
 

    
# Plots 3 views
def surface_brain_plotv2(img, subjects_dir, surf='pial', cmap='Reds', symmetric=True, 
                       threshold=0, fade=True, cbar_label=None, figsize=(10,7), setmax = False, setmin = False, setmid = False, threshold_div = 3,
                       datmax = 1, datmin = 0, datmid = 0.5, cort = (0.9,0.9,0.9)):
    
    # some package imports inside function - ignore my bad practice
    from nilearn import surface
    import matplotlib as mpl
    
    # make MNE stc out of nifti
    lh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
    lh = surface.vol_to_surf(img, lh_surf)
    rh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
    rh = surface.vol_to_surf(img, rh_surf)
    data = np.hstack([lh, rh])
    vertices = [np.arange(len(lh)), np.arange(len(rh))]
    stc = mne.SourceEstimate(data, vertices, tmin=0, tstep=1)

    # set up axes
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_axes([0, 0.2, 0.25, 0.5])  # left
    ax2 = fig.add_axes([0.5, 0.2, 0.25, 0.5])  #right
    # ax3 = fig.add_axes([0.0, 0.15, 0.35, 0.35])  # bottom-left
    # ax4 = fig.add_axes([0.65, 0.15, 0.35, 0.35])  # bottom-right
    ax3 = fig.add_axes([0.25, 0.2, 0.25, 0.6])  # center 
    cax = fig.add_axes([0.77, 0.2, 0.03, 0.6]) # colorbar ax
    for ax in [ax1, ax2, ax3]:#, ax4, ax5]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax =np.max(np.abs(data)) #0.158 or 0.097
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/2)
        if fade:
            clim = {'kind': 'value', 'pos_lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'pos_lims':(threshold, threshold, vmax)}
    else:
        if setmax:
            vmax = datmax#np.max(data)
        else:
            vmax = np.max(data)
            print("VMAXXXXXX",vmax)
            
        if setmin:
            vmin = datmin
        else:
            vmin = np.min(data)
            print("VMINNNNN",vmin)
            
        if setmid:
            mid = datmid
        else:
            mid = threshold + ((vmax-threshold)/threshold_div)
            print("VMIDDDDD",mid)
            
        if fade:
            clim = {'kind': 'value', 'lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'lims':(threshold, threshold, vmax)}
    
        
    if surf=='inflated':
        cortex='low_contrast'
    elif surf=='pial':
        cortex= cort # was (0.6, 0.6, 0.6)
    else:
        cortex=cort 
    plot_kwargs = dict(subject='fsaverage',
                       subjects_dir=subjects_dir,
                       surface=surf,
                       cortex=cortex,
                       background='white',
                       colorbar=False,
                       time_label=None,
                       time_viewer=False,
                       transparent=True,
                       clim=clim,
                       colormap=cmap,
                       )
    
    def remove_white_space(imdata):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped

    # top left
    views = ['lat']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax1.imshow(screenshot)

    # top right
    views = ['lat']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax2.imshow(screenshot)

    # # bottom left
    # views = ['med']
    # hemi = 'lh'
    # brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    # screenshot = brain.screenshot()
    # brain.close()
    # screenshot = remove_white_space(screenshot)
    # ax3.imshow(screenshot)

    # # bottom right
    # views = ['med']
    # hemi = 'rh'
    # brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    # screenshot = brain.screenshot()
    # brain.close()
    # screenshot = remove_white_space(screenshot)
    # ax4.imshow(screenshot)

    # middle
    views = ['dorsal']
    hemi = 'both'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    background = np.sum(screenshot, -1) == 3*255
    alpha = np.ones(screenshot.shape[:2])  
    alpha[background] = 0
    ax3.imshow(screenshot, alpha=alpha)

    # colorbar
    # cmap = plt.get_cmap(cmap)
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #          cax=cax, orientation='horizontal', label=cbar_label)
    # cbar.ax.tick_params(labelsize=25)
    # cbar.set_label(cbar_label, fontsize=30, labelpad=20)
    
    
    # Colorbar
    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    
    # Set ticks for min, midpoint, and max values
    tick_values = np.linspace(vmin, vmax, 2)  # [vmin, midpoint, vmax]
    if symmetric == False:
        tick_values2 = np.linspace(datmin, datmax, 2)
    else:
        tick_values2 = np.linspace(datmin, datmax, 2)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation='vertical', label=cbar_label, ticks=tick_values)
    
    # Format value conditionally
    ticks = []
    for i in tick_values2:
        if i ==0:
            ticks.append('0')
        elif i < 0.0000001 and i >0:
            # Scientific notation with superscript
            i = f"{i:.1e}".split('e')
            base, exponent = i[0], int(i[1])
            ticks.append(rf"{base} × 10$^{{{exponent}}}$")
        else:
            # Regular decimal format
            ticks.append(f"{i:.0e}")
    
    # Adjust tick labels and font size
    print(ticks)
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label(cbar_label, fontsize=40, labelpad=18)
    
    # Optional: Format tick labels for clarity
    cbar.ax.set_yticklabels(ticks)#[f"{tick}" for tick in ticks])  # Adjust decimal places if needed
    plt.tight_layout()

    return fig



def make_atlas_niftiSC(atlas_img, values):
    """
    Dr. Sebastian Coleman's original Make Atlas Script
    """

    # some package imports inside function - ignore my bad practice
    from nilearn import image, datasets
    from nibabel import Nifti1Image
    
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()
    atlas_new = np.zeros(np.shape(atlas_data))
    indices = np.unique(atlas_data[atlas_data>0])
    for reg in range(len(values)):
        reg_mask = atlas_data == indices[reg]
        atlas_new[reg_mask] = values[reg]
    new_img = Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    img = image.resample_img(new_img, mni.affine)
    
    return img

def surface_brain_plotSC(img, subjects_dir, surf='inflated', cmap='cold_hot', symmetric=True, 
                       threshold=0, fade=True, cbar_label=None, figsize=(10,7)):
    

    """
    Dr. Sebastian Coleman's original Brain Plotting Script
    """
    # some package imports inside function - ignore my bad practice
    from nilearn import surface
    import matplotlib as mpl
    
    # make MNE stc out of nifti
    lh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
    lh = surface.vol_to_surf(img, lh_surf)
    rh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
    rh = surface.vol_to_surf(img, rh_surf)
    data = np.hstack([lh, rh])
    vertices = [np.arange(len(lh)), np.arange(len(rh))]
    stc = mne.SourceEstimate(data, vertices, tmin=0, tstep=1)

    # set up axes
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_axes([0, 0.60, 0.35, 0.35])  # top-left
    ax2 = fig.add_axes([0.65, 0.60, 0.35, 0.35])  # top-right
    ax3 = fig.add_axes([0.0, 0.15, 0.35, 0.35])  # bottom-left
    ax4 = fig.add_axes([0.65, 0.15, 0.35, 0.35])  # bottom-right
    ax5 = fig.add_axes([0.32, 0.3, 0.36, 0.5])  # center 
    cax = fig.add_axes([0.25, 0.1, 0.5, 0.03]) # colorbar ax
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/2)
        if fade:
            clim = {'kind': 'value', 'pos_lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'pos_lims':(threshold, threshold, vmax)}
    else:
        vmax = np.max(data)
        vmin = np.min(data)
        mid = threshold + ((vmax-threshold)/3)
        if fade:
            clim = {'kind': 'value', 'lims':(threshold, 0.9*vmax, vmax)}
        else:
            clim = {'kind': 'value', 'lims':(threshold, threshold, vmax)}
            
    print("VMAXXX: ", vmax)
    print("VMIDDDD: ", mid)
    print("VMINNN: ", vmin)
        
    if surf=='inflated':
        cortex='low_contrast'
    elif surf=='pial':
        cortex=(0.6, 0.6, 0.6)
    else:
        cortex=(0.6, 0.6, 0.6)
    plot_kwargs = dict(subject='fsaverage',
                       subjects_dir=subjects_dir,
                       surface=surf,
                       cortex=cortex,
                       background='white',
                       colorbar=False,
                       time_label=None,
                       time_viewer=False,
                       transparent=True,
                       clim=clim,
                       colormap=cmap,
                       )
    
    def remove_white_space(imdata):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped

    # top left
    views = ['lat']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax1.imshow(screenshot)

    # top right
    views = ['lat']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax2.imshow(screenshot)

    # bottom left
    views = ['med']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax3.imshow(screenshot)

    # bottom right
    views = ['med']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax4.imshow(screenshot)

    # middle
    views = ['dorsal']
    hemi = 'both'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    background = np.sum(screenshot, -1) == 3*255
    alpha = np.ones(screenshot.shape[:2])  
    alpha[background] = 0
    ax5.imshow(screenshot, alpha=alpha)

    # colorbar
    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)
    
    return fig

