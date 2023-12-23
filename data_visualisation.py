#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:55:07 2023

  @author: clarariegis

   AUTHOR: Clara Riégis

     NAME: abide_data.py

  PURPOSE: Visualise data and phenotypes of the participants, 
           Undergraduate project supervised by Dr. František Váša

QUESTIONS: Do graph theoretic measures depend on overall mean functional connectivity? 
           Is this affected by different pre-processing pipelines ? 
           Which of NR, MR and GSR lead to the best classification? 
           Is this effect specific to one atlas/ datasets? 


    OVERVIEW
    –––––––––––––––––––––––––––––––––––––––––––––––
    - 0 - Libraries
    - 1 - Descriptive stats
    - 2 - Prepare the data for analysis     
    - 3 - Distance matrix         
    - 4 - Data visualisation (before exclusions) 
    - 5 - Balanced framewise displacement
    –––––––––––––––––––––––––––––––––––––––––––––––
"""

# %%                - 0 - Libraries

from scipy.spatial import distance
import nibabel as nib
import pandas as pd
import scipy.stats as stats
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ––– Custom functions (analysis) –––
# Set the directory to where the functions are being stored:
import os
os.chdir('/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/code')
from graph_measures import graph_measures
   

# %%                - 1 - Descriptive stats             


# Loading the phenotypic file to know which ID are in the ASD or TC groups.
phenotypic = pd.read_csv('participants/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')


# Male / females without quality check 
available = phenotypic[phenotypic["subject"].isin(remaining_pcp_id)]
sum(available["SEX"] == 1)
sum(available["SEX"] == 2)
np.min(available["AGE_AT_SCAN"])

# ASD / TC without quality check 
sum(available["DX_GROUP"] == 1)
sum(available["DX_GROUP"] == 2)
np.min(available["AGE_AT_SCAN"])

# %%                - 2 - Prepare the data for analysis        

# - Separate ASD and TC pcps

# The TC group is categorised as "2", which I switched to "0".
phenotypic['DX_GROUP'][phenotypic['DX_GROUP'] == 2] = 0

# Separate/store the IDs of HC and ASD participants:
id_tc2 = phenotypic.loc[phenotypic['DX_GROUP'] == 0, 'SUB_ID'].tolist()
id_asd2 = phenotypic.loc[phenotypic['DX_GROUP'] == 1, 'SUB_ID'].tolist()


# REMINDER: data for all participants in the NR group was previoulsy stored
# in: "all_pcp_nr" while the GSR group was stored in: "all_pcp_gsr" (see I.2.a).


# Dictionary with NR and GSR time series for TC.
all_pcp_tc = {"nr": all_pcp_nr[all_pcp_nr["ID"].isin(id_tc2)],
              "gsr": all_pcp_gsr[all_pcp_gsr["ID"].isin(id_tc2)]}

# Making sure the index is a list of consecutive numbers.
all_pcp_tc["nr"].index = np.arange(0, len(all_pcp_tc["nr"]))
all_pcp_tc["gsr"].index = np.arange(0, len(all_pcp_tc["gsr"]))


# Dictionary with NR and GSR time series for ASD.
all_pcp_asd = {"nr": all_pcp_nr[all_pcp_nr["ID"].isin(id_asd2)],
               "gsr": all_pcp_gsr[all_pcp_gsr["ID"].isin(id_asd2)]}

# Making sure the index is a list of consecutive numbers.
all_pcp_asd["nr"].index = np.arange(0, len(all_pcp_asd["nr"]))
all_pcp_asd["gsr"].index = np.arange(0, len(all_pcp_asd["gsr"]))


    ## Exclusion
graph_measures(all_pcp_tc["nr"], all_pcp_tc["gsr"], prop_threshold = 0.15, phenotypic=phenotypic, exclusion=True)
graph_measures(all_pcp_asd["nr"], all_pcp_asd["gsr"], prop_threshold = 0.15, phenotypic=phenotypic, exclusion=True)


# %%%                - 3 - distance matrix           


# DISTANCE MATRIX
# It will be used to plot the correlation between edges and mFD as a function
# of distance.
# Getting the coordinated of the desikan ROIs in MNI-152 space (excluding the
# background image):
ho_atlas = nib.load("participants/atlas/ho_roi_atlas.nii")
ho_coord = plotting.find_parcellation_cut_coords(ho_atlas, background_label=0)
ho_coord = np.delete(ho_coord, 82, axis=0)  # Delete the "fake" ROI.
# Getting the condensed distance matrix:
cond_dist_mat = distance.pdist(ho_coord, metric="euclidean")


# %%%                - 4 - Data visualisation (before exclusions)         


# Loading the phenotypic file to know which ID are in the ASD or TC groups.
phenotypic = pd.read_csv(
    'participants/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
phenotypic = phenotypic.set_index("SUB_ID")



tc_meas_noexclusion = graph_measures( all_pcp_tc["nr"], all_pcp_tc["gsr"], prop_threshold=0.05, phenotypic=phenotypic, exclusion=False)
asd_meas_noexclusion = graph_measures(all_pcp_asd["nr"], all_pcp_asd["gsr"], prop_threshold=0.05, phenotypic=phenotypic, exclusion=False)

mfc_tc = tc_meas_noexclusion["group_nr"]["mean FC"]
mfc_asd = asd_meas_noexclusion["group_nr"]["mean FC"]


size_plt = 23
size_pcp = 18
col_all = "darkslateblue"
col_asd = "lightcoral"
col_tc = "cornflowerblue"
col_fem = "darkgoldenrod"
col_mal = "wheat"

fig, ax = plt.subplot_mosaic("""
                              CC
                              AB
                              DD
                              EE
                              """, figsize=(13, 27))

plt.subplots_adjust(top=0.7, bottom=0.01, hspace=0.7, wspace=0.3)


# –––––––––  MEAN FC DISTRBUTION –––––––––

# Mean FC distribution for TC group - before exclusion :
ax["A"].hist(mfc_tc, bins=80, color=col_tc)
ax["A"].set_xlabel("mean FC", size=size_plt)          # x label.
ax["A"].set_ylabel("Frequency",  size=size_plt)       # y label.
# set the size of the ticks.
ax["A"].tick_params(axis="both", labelsize=size_plt)
# participants number.
# ax["A"].text(0.62, 16, f"n = {len(mfc_tc)}", size=size_pcp)
# Add median and exclusion boundaries:
ax["A"].axvline(np.median(mfc_tc), color='black', linestyle='-', linewidth=1.4)
upper_boudary = np.median(mfc_tc) + 3 * stats.median_abs_deviation(mfc_tc)
ax["A"].axvline(upper_boudary, color='grey', linestyle='dashed', linewidth=1)
lower_boudary = np.median(mfc_tc) - 3 * stats.median_abs_deviation(mfc_tc)
ax["A"].axvline(lower_boudary, color='grey', linestyle='dashed', linewidth=1)


# Mean FC distribution for ASD patients - before exclusion :
ax["B"].hist(mfc_asd, bins=80, color=col_asd)
ax["B"].set_xlabel("mean FC", size=size_plt)
ax["B"].set_ylabel("Frequency",  size=size_plt)
# set the size of the ticks.
ax["B"].tick_params(axis="both", labelsize=size_plt)
ax["B"].set_ylim(ax["A"].get_ylim())
# ax["B"].text(0.62, 16, f"n = {len(mfc_asd)}", size=size_pcp)
ax["B"].axvline(np.median(mfc_asd), color='black', linestyle='-', linewidth=1.2)
upper_boudary = np.median(mfc_asd) + 3 * stats.median_abs_deviation(mfc_asd)
ax["B"].axvline(upper_boudary, color='black', linestyle='dashed', linewidth=1)
lower_boudary = np.median(mfc_asd) - 3 * stats.median_abs_deviation(mfc_asd)
ax["B"].axvline(lower_boudary, color='black', linestyle='dashed', linewidth=1)


# –––––––––  MEAN FD DISTRBUTION –––––––––

# At this stage only pcp with 0s at ts were exluded (but NONE based on mFD or mFC).
all_pcp_fd = phenotypic.loc[all_pcp_nr["ID"], "func_mean_fd"]
ax["C"].hist(all_pcp_fd, bins=90, color=col_all)
ax["C"].set_xlabel("mean FD",  size=size_plt)
ax["C"].set_ylabel("Frequency",  size=size_plt)
# set the size of the ticks.
ax["C"].tick_params(axis="both", labelsize=size_plt)
# ax["C"].set_title("Distribution of mean FD before exclusion", size = 25, fontweight = "bold")
# ax["C"].text(1.2, 130, f"n = {len(all_pcp_nr)}", size=size_pcp)
ax["C"].axvline(0.4, color='black', linestyle='dashed', linewidth=1)


barWidth = 0.30


all_meas_noexclusion = pd.concat([tc_meas_noexclusion["group_nr"], asd_meas_noexclusion["group_nr"]], axis=0)
all_meas_noexclusion = all_meas_noexclusion.set_index('ID')
rem_pheno = phenotypic.loc[all_meas_noexclusion.index]
rem_ev_noexcl = pd.concat([all_meas_noexclusion, rem_pheno], axis=1)
rem_ev_noexcl["SUB_ID"] = rem_ev_noexcl.index
rem_ev_noexcl.index = np.arange(0, len(rem_ev_noexcl))

TC = rem_ev_noexcl.loc[rem_ev_noexcl["DX_GROUP"] == 2]
ASD = rem_ev_noexcl.loc[rem_ev_noexcl["DX_GROUP"] == 1]

TC_plt = []
ASD_plt = []

for this_site in rem_ev_noexcl["SITE_ID"].unique():

    TC_plt.append(len(TC[TC["SITE_ID"] == this_site]))
    ASD_plt.append(len(ASD[ASD["SITE_ID"] == this_site]))


# Set position of bar on X axis
br0 = np.arange(len(rem_ev_noexcl["SITE_ID"].unique()))
br1 = [x + 0.25 for x in br0]
br2 = [x + barWidth for x in br1]

# Make the plot

ax["D"].bar(br1, TC_plt, color=col_tc, width=barWidth,
            edgecolor='grey', label='TC')
ax["D"].bar(br2, ASD_plt, color=col_asd, width=barWidth,
            edgecolor='grey', label='ASD')


# Adding Xticks
#ax["D"].set_xlabel('Scanning sites', fontweight ='bold', fontsize = 15)
#plt.ylabel('Students passed', fontweight ='bold', fontsize = 15)
ax["D"].set_xticks([r + barWidth for r in range(len(rem_ev_noexcl["SITE_ID"].unique()))],
                   rem_ev_noexcl["SITE_ID"].unique().tolist(), rotation=80)
# set the size of the ticks.
ax["D"].tick_params(axis="both", labelsize=size_plt - 4)
# ax["D"].text(15, 120, f"n = {len(rem_ev_noexcl)}", size=size_pcp)
# ax["D"].text(15, 105, f"n TC = {len(TC)}", size=size_pcp)
# ax["D"].text(15, 90, f"n ASD = {len(ASD)}", size=size_pcp)

# ax["D"].legend()


FEM = rem_ev_noexcl[rem_ev_noexcl["SEX"] == 2]
MAL = rem_ev_noexcl[rem_ev_noexcl["SEX"] == 1]

FEM_plt = []
MAL_plt = []

i = 0
for this_site in rem_ev_noexcl["SITE_ID"].unique():

    FEM_plt.append(len(FEM[FEM["SITE_ID"] == this_site]))
    MAL_plt.append(len(MAL[MAL["SITE_ID"] == this_site]))


# Set position of bar on X axis
br0 = np.arange(len(rem_ev_noexcl["SITE_ID"].unique()))
br1 = [x + 0.25 for x in br0]
br2 = [x + barWidth for x in br1]

# Make the plot

ax["E"].bar(br1, FEM_plt, color=col_fem, width=barWidth,
            edgecolor='grey', label='FEM')
ax["E"].bar(br2, MAL_plt, color=col_mal, width=barWidth,
            edgecolor='grey', label='MAL')


# Adding Xticks
#ax["E"].set_xlabel('Scanning sites', fontweight ='bold', fontsize = 15)
#plt.ylabel('Students passed', fontweight ='bold', fontsize = 15)
ax["E"].set_xticks([r + barWidth for r in range(len(rem_ev_noexcl["SITE_ID"].unique()))],
                   rem_ev_noexcl["SITE_ID"].unique().tolist(), rotation=80)
# set the size of the ticks.
ax["E"].tick_params(axis="both", labelsize=size_plt - 4)
# ax["E"].text(15, 120, f"n = {len(rem_ev_noexcl)}", size=size_pcp)
# ax["E"].text(15, 105, f"n fem = {len(FEM)}", size=size_pcp)
# ax["E"].text(15, 90, f"n mal = {len(MAL)}", size=size_pcp)
# ax["E"].legend()
ax["D"].set_ylim(ax["E"].get_ylim())


all_patch = mpatches.Patch(color = col_all, label= f'All participants (n = {len(all_pcp_nr)})')
asd_patch = mpatches.Patch(color = col_asd, label=f'ASD (n = {len(mfc_asd)})')
tc_patch = mpatches.Patch(color = col_tc, label= f'TC (n = {len(mfc_tc)})')
fem_patch = mpatches.Patch(color = col_fem, label= f'Females (n = {len(FEM)})')
mal_patch = mpatches.Patch(color = col_mal, label= f'Males (n = {len(MAL)})')


fig.legend(handles=[all_patch, asd_patch, tc_patch, fem_patch, mal_patch], loc = (0.63,0.813), fontsize = size_plt)

ax["C"].set_title("Mean FD distribution", size = size_plt)
ax["A"].text(x = 0.5, y = 25, s = "Mean FC distribution in TC and ASD", size = size_plt)
ax["D"].set_title("ASD and TC distribution across scanning sites", size = size_plt)
ax["E"].set_title("Female and male distribution across scanning sites", size = size_plt)


plt.show()





# %%                - 5 - Balanced framewise displacement


# NR measures were generated without balancing mean FD and stored under
# "rem_everything_un", which contains the graph measures and
# phenotypic data after participant exclusion for FD exceeding 0.4-0.1.


with open(r"clean_unbalanced_dat.pickle", "rb") as input_file:
    rem_everything_un2 = cPickle.load(input_file)

# Extracting TC participant:
tc_everyt_fd04 = rem_everything_un2.loc[rem_everything_un2["DX_GROUP"] == 0]
# Put the IDs back into their original column.
tc_everyt_fd04["SUB_ID"] = tc_everyt_fd04.index
# Make the index a list of consecutive numbers.
tc_everyt_fd04.index = np.arange(0, len(tc_everyt_fd04))

# Extracting ASD participant:
asd_everyt_fd04 = rem_everything_un2.loc[rem_everything_un2["DX_GROUP"] == 1]
# Put the IDs back into their original column.
asd_everyt_fd04["SUB_ID"] = asd_everyt_fd04.index
# Make the index a list of consecutive numbers.
asd_everyt_fd04.index = np.arange(0, len(asd_everyt_fd04))


i = 0
p = 0

asd_everyt_fd04_bal = asd_everyt_fd04.copy()

while p < 0.05:  # while there is a significant difference in mFD between ASD and TC...

    # ... exclude ASD participants that have the highest mFD.
    asd_everyt_fd04_bal = asd_everyt_fd04_bal.drop(
        asd_everyt_fd04_bal['func_mean_fd'].idxmax())

    # Run a Mann Whitney U test to see if the difference is still significant.
    mann_whi = stats.mannwhitneyu(asd_everyt_fd04_bal['func_mean_fd'],  # ASD FD
                                  tc_everyt_fd04['func_mean_fd'])  # TC FD

    # Store the p-value to assess whether it is still below the 0.05 threshold.
    p = mann_whi[1]

    i += 1

    print(p)


# Dataframe containing the NR measures and the phenotypic information of the
# participants remaining the balanced FD group.
rem_everyt_bal = pd.concat((tc_everyt_fd04, asd_everyt_fd04_bal))
print(len(rem_everyt_bal))

with open(r"clean_balanced_dat.pickle", "wb") as output_file:
    cPickle.dump(rem_everyt_bal, output_file)

# with open(r"clean_balanced_dat.pickle", "rb") as input_file:
#     rem_everything_bal2 = cPickle.load(input_file)


fig, ax = plt.subplots(1, 2, figsize=(13, 5))
plt.subplots_adjust(top=0.8, bottom=0.01, hspace=0.5, wspace=0.3)
i = 0
list_plots = ["func_mean_fd", "mean FC"]

for this_plot in list_plots:

    asd = asd_everyt_fd04_bal[this_plot]

    tc = tc_everyt_fd04[this_plot]

    label = "mean FD"

    if this_plot == "vectorised matrix":

        asd = asd_everyt_fd04_bal[this_plot]

        tc = tc_everyt_fd04[this_plot]

        label = "mean FC"

    data = [tc, asd]

    mann_whi = stats.mannwhitneyu(tc, asd)

    p_value = mann_whi[1]

    if p_value < 0.01:
        p_value = "< 0.01"

    else:
        p_value = f" = {round(p_value, 3)}"

    raincloud(data, feature1="TC", feature2="ASD",  x_label=label,
              title=f"p {p_value}", colors="on", ax=ax[i], size=size_plt)

    i += 1

