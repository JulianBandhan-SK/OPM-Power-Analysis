
"""
Additional stuff:
    plt.rcParams['font.family'] = 'Arial'  # Set global font to Arial
    
"""
from .common_imports import np, stats, sns, plt, Line2D, pd




# %%
import seaborn as sns
import matplotlib.pyplot as plt

def plot_stuff_vs_stuffv2(values1, values2, plt_title, 
                        plot_type="lineplot", 
                        cmap='red', lmap='darkred', 
                        ylabel="PSD", xlabel="Age (years)", alpha = 0.7, lalpha = 0.4,figsize = (12,12)):

    # Seaborn styling
    sns.set(style="whitegrid")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    #fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)


    # Choose plot type
    if plot_type == "lineplot":
        sns.lineplot(x=values1, y=values2, color=cmap, ax=ax, linewidth = 3)
    elif plot_type == "scatterplot":
        sns.scatterplot(x=values1, y=values2, color=cmap, ax=ax)
    elif plot_type == "barplot":
        sns.barplot(x=values1, y=values2, color=cmap, ax=ax)
    elif plot_type == 'regplot':
        sns.regplot(x=values1, y=values2, ci=None, line_kws={"linewidth":5,"color": lmap, "alpha":lalpha}, scatter_kws={"s": 100, "alpha": alpha, "color": cmap})
    else:
        raise ValueError("plot_type must be one of: 'lineplot', 'scatterplot', 'barplot'")

    # Formatting
    #ax.set_title(plt_title, fontsize=50)
    ax.set_xlabel(xlabel, fontsize=70)
    ax.set_ylabel(ylabel, fontsize=70)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='both', which='major', labelsize=75)

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    return fig,ax


# %%


def plot_stuff_vs_stuffv3(values1, values2, plt_title, 
                        plot_type="lineplot", 
                        cmap='red', lmap='darkred', 
                        ylabel="PSD", xlabel="Age (years)", alpha = 0.7, lalpha = 0.4,figsize = (12,50)):

    # Seaborn styling
    sns.set(style="whitegrid")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    #fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)


    # Choose plot type
    if plot_type == "lineplot":
        sns.lineplot(x=values1, y=values2, color=cmap, ax=ax, linewidth = 3)
    elif plot_type == "scatterplot":
        sns.scatterplot(x=values1, y=values2, color=cmap, ax=ax)
    elif plot_type == "barplot":
        sns.barplot(x=values1, y=values2, color=cmap, ax=ax)
    elif plot_type == 'regplot':
        sns.regplot(x=values1, y=values2, ci=None, line_kws={"linewidth":5,"color": lmap, "alpha":lalpha}, scatter_kws={"s": 100, "alpha": alpha, "color": cmap})
    else:
        raise ValueError("plot_type must be one of: 'lineplot', 'scatterplot', 'barplot'")

    # Formatting
    #ax.set_title(plt_title, fontsize=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='both', which='major')

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    return fig,ax
# %% Test

# Function to extract PSD for each band
def extract_band_psd(prime_psd, freq, bands):
    band_indices = {band: np.where((freq >= low) & (freq <= high))[0] for band, (low, high) in bands.items()}
    print(band_indices)
    band_psd = {band: np.mean(prime_psd[:, indices[0]:indices[-1] + 1, :], axis=1) for band, indices in band_indices.items()}
    return band_psd

# def plot_bandpsd_vs_age(df, x, y, plt_title, p_value = None, showpval = True, cmap='red', lmap='darkred'):

#     # Plot using Seaborn son, mwahahahaha
#     sns.set(style="whitegrid")
#     fig = plt.figure(figsize=(15, 20))  # Keep original plot size
    
    
#     if showpval:
#         if p_value == None:
#             # Calculate linear regression
#             slope, intercept, r_value, p_value, _ = stats.linregress(df[x], df[y])
#         else:
#             p_value = p_value
        
#         # Format p-value conditionally
#         if p_value < 0.001:
#             # Scientific notation with superscript
#             p_value_sci = f"{p_value:.1e}".split('e')
#             base, exponent = p_value_sci[0], int(p_value_sci[1])
#             p_formatted = rf"{base} × 10$^{{{exponent}}}$"
#         else:
#             # Regular decimal format
#             p_formatted = f"{p_value:.3f}"
        
#         # Custom legend handles for R and p values (line only)
#         legend_handles = [
#             Line2D([0], [0], color=lmap, lw=5, label=fr'p = {p_formatted}')
#         ]
#         plt.legend(handles=legend_handles, loc='upper right', fontsize=40)  # Massive legend text


#     sns.regplot(df, x=x, y=y, ci=None, line_kws={"linewidth":5,"color": lmap}, scatter_kws={"s": 100, "alpha": 0.7, "color": cmap})
    
    

#     # Increase font sizes without resizing plot
#     plt.title(f'{plt_title}', fontsize=60)  # Massive title
#     plt.xlabel('Age (years)', fontsize=55)  # Massive axis labels
#     plt.ylabel('PSD', fontsize=55)
#     plt.xticks(fontsize=50)  # Massive tick labels
#     plt.yticks(fontsize=40)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return fig

# def plot_avgbandpsd_vs_age(ages, band_psd, band, cmap='red', lmap='darkred'):
#     psd_values = band_psd
    
#     # Calculate linear regression
#     slope, intercept, r_value, p_value, _ = stats.linregress(ages, psd_values)

#         # Format p-value conditionally
#     if p_value < 0.001:
#         # Scientific notation with superscript
#         p_value_sci = f"{p_value:.1e}".split('e')
#         base, exponent = p_value_sci[0], int(p_value_sci[1])
#         p_formatted = rf"{base} × 10$^{{{exponent}}}$"
#     else:
#         # Regular decimal format
#         p_formatted = f"{p_value:.3f}"

#     # Plot using Seaborn son, mwahahahaha
#     sns.set(style="whitegrid")
#     fig = plt.figure(figsize=(20, 15))  # Keep original plot size
#     sns.regplot(x=ages, y=psd_values, ci=None, line_kws={"linewidth":5,"color": lmap}, scatter_kws={"s": 100, "alpha": 0.7, "color": cmap})
    
#     # Custom legend handles for R and p values (line only)
#     legend_handles = [
#         Line2D([0], [0], color=lmap, lw=5, label=fr'R = {r_value:.2f}, p = {p_formatted}')
#     ]
#     plt.legend(handles=legend_handles, loc='upper right', fontsize=40)  # Massive legend text

#     # Increase font sizes without resizing plot
#     plt.title(f'Average {band} across all regions', fontsize=60)  # Massive title
#     plt.xlabel('Age (years)', fontsize=55)  # Massive axis labels
#     plt.ylabel('Average PSD', fontsize=55)
#     plt.xticks(fontsize=50)  # Massive tick labels
#     plt.yticks(fontsize=40)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return fig

# def plot_indpsd_vs_age(ages, band_psd, freq_idx, region_index, plt_title, cmap='red', lmap='darkred', ylabel = "PSD"):
#     """
#     Hopefully plots each frequency value vs age
#     Parameters
#     ----------
#     ages : TYPE
#         DESCRIPTION.
#     band_psd : array - [n_points x n_regions]
#         DESCRIPTION.
#     region_index : TYPE
#         DESCRIPTION.
#     plt_title : TYPE
#         DESCRIPTION.
#     cmap : TYPE, optional
#         DESCRIPTION. The default is 'red'.
#     lmap : TYPE, optional
#         DESCRIPTION. The default is 'darkred'.

#     Returns
#     -------
#     fig : TYPE
#         DESCRIPTION.

#     """
#     # Ensure band_psd is correctly indexed
#     if band_psd.ndim == 3:
#         psd_values = band_psd[:, freq_idx, region_index]  # 3D: (n_points, n_freqs, n_regions)
#     else:
#         psd_values = band_psd[:, region_index]  # 2D: (n_points, n_regions)

#     # Seaborn styling
#     sns.set(style="whitegrid")

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(15, 10))  # Adjusted for better aspect ratio

#     # Scatter plot
#     #sns.scatterplot(x=ages, y=psd_values, color=cmap, s=100, alpha=0.7, ax=ax)
#     sns.regplot(x=ages, y=psd_values, ci=None, line_kws={"linewidth":5,"color": lmap}, scatter_kws={"s": 100, "alpha": 0.7, "color": cmap})
    


#     # Formatting
#     ax.set_title(plt_title, fontsize=30)
#     ax.set_xlabel("Age (years)", fontsize=25)
#     ax.set_ylabel(ylabel, fontsize=25)
#     ax.tick_params(axis='both', which='major', labelsize=20)

#     # Grid and layout
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return fig

def plot_stuff_vs_stuff(values1, values2, plt_title, cmap='red', lmap='darkred', ylabel = "PSD", xlabel = "Age (years)"):

    # Seaborn styling
    sns.set(style="whitegrid")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjusted for better aspect ratio

    # Lineplot
    sns.lineplot(x=values1, y=values2, color=cmap, ax=ax)


    # Formatting
    ax.set_title(plt_title, fontsize=30)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fig
def plot_stuff_vs_stuffv6(df, x, y, subject_id,
                          plt_title="",
                          ylabel="PSD", xlabel="Age (years)",
                          palette="husl"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 10))

    # Unique colors per subject
    unique_subjects = df[subject_id].unique()
    color_palette = sns.color_palette(palette, len(unique_subjects))
    color_map = dict(zip(unique_subjects, color_palette))

    # Plot each subject with lines connecting their visits
    for subject in unique_subjects:
        sub_df = df[df[subject_id] == subject].sort_values(x)
        ax.plot(sub_df[x], sub_df[y], color=color_map[subject], alpha=0.5, linewidth=2)
        ax.scatter(sub_df[x], sub_df[y], color=color_map[subject], s=100, edgecolor='k', alpha=0.8)

    # Add a regression line (across all data)
    sns.scatter(data=df, x=x, y=y, ci=None,
                scatter=False,
                #line_kws={"color": "darkred", "linewidth": 4},
                ax=ax)

    # Formatting
    ax.set_title(plt_title, fontsize=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', labelsize=25)
    plt.grid(True)
    plt.tight_layout()

    return fig, ax

def plot_with_focus_subject(df, x, y, subject_id, focus_subject,
                            plt_title="", ylabel="Value", xlabel="X",
                            base_color="lightblue", focus_color="blue"):

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1️⃣ Plot all subjects in the background
    ax.scatter(df[x], df[y], color=base_color, s=80, alpha=0.9, label="All Subjects")

    # 2️⃣ Highlight one subject with line + larger points
    sub_df = df[df[subject_id] == focus_subject].sort_values(x)
    sub_df = sub_df.dropna()
    ax.plot(sub_df[x], sub_df[y], color=focus_color, linewidth=3, label=f"{focus_subject}")
    ax.scatter(sub_df[x], sub_df[y], color=focus_color, s=150, edgecolor='k', alpha=0.9)

    # Formatting
    ax.set_title(plt_title, fontsize=40)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    return fig, ax





def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
