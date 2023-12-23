#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
    - 1 - Main figures
        - FRAMEWISE DISPLACEMENT AND FUNCTIONAL CONNECTIVITY
        - EDGES WEIGHTS FOR NR, MR and GSR
        - STAT ANALYSIS
    –––––––––––––––––––––––––––––––––––––––––––––––

"""

#%%                - 0 - Libraries


import _pickle as cPickle
from decimal import Decimal
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import matplotlib.colors as colors
from numpy import unravel_index
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# ––– Custom functions (analysis) –––
# Set the directory to where the functions are being stored:
import os
os.chdir('/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/code')
from raincloud import raincloud
from graph_measures import graph_measures

#%%                - 1 - Analyses


size_plt = 20
plt_size = 20
size_pcp = 13
col_all = "darkslateblue"
col_asd = "lightcoral"
col_tc = "cornflowerblue"
n_round = 3

plt.rcParams.update({"axes.grid" : False})

# list_fd_max = [0.4]  # , 0.3, 0.2, 0.1
# , "maximum mean FD : 0.3", "maximum mean FD : 0.2", "maximum mean FD : 0.1"
# rem_everything_fd01 = rem_everything_un2[rem_everything_un2["func_mean_fd"] <= 0.1]


with open(r"clean_unbalanced_dat.pickle", "rb") as input_file:
    rem_everything_un2 = cPickle.load(input_file)

with open(r"clean_balanced_dat.pickle", "rb") as input_file:
    rem_everything_bal2 = cPickle.load(input_file)
    
phenotypic = pd.read_csv('participants/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')


# Replace the 2 by 0 for the typical controls
phenotypic2 = pd.read_csv('participants/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
phenotypic2["DX_GROUP"].loc[phenotypic2["DX_GROUP"] == 2] = 0

# FULL SAMPLE DATA 
id_full = rem_everything_un2.index.tolist() # ID of the participants in the balanced group (full sample).

ts_nr_full = all_pcp_nr.loc[all_pcp_nr["ID"].isin(id_full)] # selecting the time series of the participants.
ts_nr_full.index = np.arange(0, len(ts_nr_full))            # making sure that the index is made of consecutive numbers. 

ts_gsr_full = all_pcp_gsr.loc[all_pcp_gsr["ID"].isin(id_full)] # doing the same thing with the GSR time series. 
ts_gsr_full.index = np.arange(0, len(ts_gsr_full))

meas_full = graph_measures(ts_nr_full,   # Calculating graph measures for these participants. 
                           ts_gsr_full,  # this is because their mean FC is needed later in the loop. 
                           prop_threshold = 0.25, 
                           phenotypic = phenotypic, 
                           exclusion = False)

meas_nr_full = meas_full["group_nr"].copy() # Just retrieving one of the dataframes out of the three (nr, gsr or mreg). It doesn't matter which one because the mean FC is the same for all of them. 
meas_nr_full = meas_nr_full.rename(columns = {"ID" : "SUB_ID"})

rem_full = phenotypic2.merge(meas_nr_full, on = "SUB_ID" )


# UNBALANCED DATA
id_balanced = rem_everything_bal2["SUB_ID"].tolist()

ts_nr_balanced = all_pcp_nr.loc[all_pcp_nr["ID"].isin(id_balanced)]
ts_nr_balanced.index = np.arange(0, len(ts_nr_balanced))

ts_gsr_balanced = all_pcp_gsr.loc[all_pcp_gsr["ID"].isin(id_balanced)]
ts_gsr_balanced.index = np.arange(0, len(ts_gsr_balanced))

meas_balanced = graph_measures(ts_nr_balanced, 
                            ts_gsr_balanced, 
                            prop_threshold = 0.25, 
                            phenotypic = phenotypic, 
                            exclusion = False)

meas_nr_balanced = meas_balanced["group_nr"].copy() # Just retrieving one of the dataframes out of the three (nr, gsr or mreg). It doesn't matter which one because the mean FC is the same for all of them. 
meas_nr_balanced = meas_nr_balanced.rename(columns = {"ID" : "SUB_ID"})

rem_balanced = phenotypic2.merge(meas_nr_balanced, on = "SUB_ID" )


list_plots = ["eigen_c", "assortativity", "clustering","transitivity", 
              "char_path", "efficiency", "betweenness", "diameter", 
              "radius", "s_information" , "m_f_p_t"]



group_labels = ["Full sample"] # Balanced FD
data_2 = [rem_full] # rem_balanced


for (this_label, this_dat) in zip(group_labels, data_2):
    col_all = "darkslateblue"
    phenotypic = pd.read_csv('participants/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')

    #     # TC outliers:
    # funct_measures_tc = graph_measures(all_pcp_tc["nr"], all_pcp_tc["gsr"], prop_threshold = 0.15, phenotypic = phenotypic, fd_lim = this_fd_max, exclusion = True)

    # # Number of participants after non finite exclusion: 454
    # # Number of mFC outliers excluded: 22
    # # Number of mFD outliers excluded: 6

    #     # ASD outliers:
    # funct_measures_asd = graph_measures(all_pcp_asd["nr"], all_pcp_asd["gsr"], prop_threshold = 0.15, phenotypic = phenotypic, fd_lim = this_fd_max, exclusion = True)

    # # Number of participants after non finite exclusion: 390
    # # Number of mFC outliers excluded: 23
    # # Number of mFD outliers excluded: 10

    rem_pheno_tc = this_dat.loc[this_dat["DX_GROUP"] == 0]
    rem_pheno_asd = this_dat.loc[this_dat["DX_GROUP"] == 1]

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––– FRAMEWISE DISPLACEMENT AND FUNCTIONAL CONNECTIVITY ––––––––––
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # ________ Framewise Displacement Raincloud ________

    # Compare the framewise displacement of both groups
    # just putting the FD in a variable to make it more handy in the mann whitney U
    rem_fd_tc = rem_pheno_tc["func_mean_fd"]
    rem_fd_asd = rem_pheno_asd["func_mean_fd"]
    # len(rem_fd_tc) # checking the the number of participants.
    # len(rem_fd_asd)
    
    # assessing whether the difference is significant.
    mann_fd = stats.mannwhitneyu(rem_fd_tc, rem_fd_asd)
    rbc = float(pg.mwu(rem_fd_tc, rem_fd_asd)["RBC"])
    # Reporting a few descriptive stats and the p-value.
    # print(f"""mean FD TC: {np.median(rem_fd_tc)} and ASD: {np.median(rem_fd_asd)}; 
    # median absolute deviation TC: {stats.median_abs_deviation(rem_fd_tc)} and ASD : {stats.median_abs_deviation(rem_fd_asd)} 
    # p-value: {mann_fd[1]} """)

    rem_fd = (rem_fd_tc, rem_fd_asd)

    fig, ax = plt.subplots(4, 3, figsize=(15, 14))
    plt.subplots_adjust(top=0.8, bottom=0.01, hspace=0.5, wspace=0.4)
    plt.grid(False)
    #plt.tick_params(axis = "both" , labelsize=size_plt)

    # Add a general title to the figure (specifying which mean FD cutoff is used).
    fig.suptitle(this_label,  fontsize=20, fontweight="bold", y=0.86)

    axx = ax[0, 0]

    p_value = mann_fd[1]
    if p_value < 0.01:
        p_value = f"= {Decimal(p_value):.2e}"
    else:
        p_value = f" = {round(p_value, 3)}"
    
    title = r"r$_{rb}$ = " + f"{round(rbc, n_round)}, p {p_value}"
    raincloud(rem_fd, feature1="TC", feature2="ASD",  x_label="Mean FD (mm)",
              title= title, colors="on", ax=axx, size=size_plt)

    # ________ Famewise Displacement Histogram ________

    # FD distribution after balancing : ASD
    ax[1, 0].hist(rem_pheno_asd["func_mean_fd"], bins=80, color=col_asd)
    ax[1, 0].set_xlabel("Mean FD (mm)", fontsize=size_plt)
    ax[1, 0].set_ylabel("Frequency", fontsize=size_plt)
    # ax[1, 0].text(0.8, 0.9, f'n = {len(rem_pheno_asd["func_mean_fd"])}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[1, 0].transAxes)

    # FD distribution after balancing : TC
    ax[2, 0].hist(rem_pheno_tc["func_mean_fd"], bins=80, color=col_tc)
    ax[2, 0].set_xlabel("Mean FD (mm)", fontsize=size_plt)
    ax[2, 0].set_ylabel("Frequency", fontsize=size_plt)
    # ax[2, 0].text(0.8, 0.9, f'n = {len(rem_pheno_tc["func_mean_fd"])}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[2, 0].transAxes)
    ax[1, 0].set_ylim(ax[2, 0].get_ylim())

    # FD distribution after balancing : ALL
    all_mFD = rem_pheno_tc["func_mean_fd"].tolist(
    ) + rem_pheno_asd["func_mean_fd"].tolist()
    ax[3, 0].hist(all_mFD, bins=80, color=col_all)
    ax[3, 0].set_xlabel("Mean FD (mm)", fontsize=size_plt)
    ax[3, 0].set_ylabel("Frequency", fontsize=size_plt)
    # ax[3, 0].text(0.8, 0.9, f'n = {len(all_mFD)}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[3, 0].transAxes)

    # ________ Functional Connectivity Raincloud ________
    axx = ax[0, 1]

    mann_whi = stats.mannwhitneyu(rem_pheno_tc["mean FC"], rem_pheno_asd["mean FC"])
    u, p = mann_whi
    # makes the titles cleaner on the plot (removes the "Name" and "dtype")
    p_value = float(p)
    
    # Rank biserial correlation:
    rbc = float(pg.mwu(rem_pheno_tc["mean FC"], rem_pheno_asd["mean FC"])["RBC"])
    title = r"r$_{rb}$ = " + f"{round(rbc, n_round)}, p = {round(p_value, n_round)}"
    
    mFC = [rem_pheno_tc["mean FC"], rem_pheno_asd["mean FC"]]
    raincloud(mFC, feature1="TC", feature2="ASD", x_label="Mean FC",
              title= title, colors="on", ax=axx, size=size_plt)

    # ________ Functional Connectivity Histogram ________

    #ax.get_shared_y_axes().join(ax[1,1], ax[2,1], ax[3,1])

    # FC distribution : ASD
    ax[1, 1].hist(rem_pheno_asd["mean FC"], bins=80, color=col_asd)
    ax[1, 1].set_xlabel("Mean FC", fontsize=size_plt)
    ax[1, 1].set_ylabel("Frequency", fontsize=size_plt)
    # ax[1, 1].text(0.8, 0.9, f'n = {len(rem_pheno_asd["mean FC"])}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[1, 1].transAxes)

    # FC distribution : TC
    ax[2, 1].hist(rem_pheno_tc["mean FC"], bins=80, color=col_tc)
    ax[2, 1].set_xlabel("Mean FC", fontsize=size_plt)
    ax[2, 1].set_ylabel("Frequency", fontsize=size_plt)
    # ax[2, 1].text(0.8, 0.9, f'n = {len(rem_pheno_tc["mean FC"])}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[2, 1].transAxes)
    ax[1, 1].set_ylim(ax[2, 1].get_ylim())

    # FC distribution : ALL
    # convert into list because it was in pandas core series so it wasn't working.
    all_mFC = rem_pheno_tc["mean FC"].tolist(
    ) + rem_pheno_asd["mean FC"].tolist()
    ax[3, 1].hist(all_mFC, bins=80, color=col_all)
    ax[3, 1].set_xlabel("Mean FC", fontsize=size_plt)
    ax[3, 1].set_ylabel("Frequency", fontsize=size_plt)
    # ax[3, 1].text(0.8, 0.9, f'n = {len(all_mFC)}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[3, 1].transAxes)

    # ________ Correlation FD and FC ________
    # ASD
    sns.regplot(x=rem_pheno_asd["func_mean_fd"], y=mFC[1], scatter_kws={
                's': 8}, ax=ax[1, 2], color=col_asd)
    ax[1, 2].set_xlabel("Mean FD", fontsize=size_plt)
    ax[1, 2].set_ylabel("Mean FC", fontsize=size_plt)
    r, p = stats.pearsonr(x=rem_pheno_asd["func_mean_fd"], y=mFC[1])
    # round them cause we don't need a tone of decimals
    ax[1, 2].set_title(
        f"r = {round(r, 3)}, p = {round(p, 3)}", fontsize=size_plt)

    if p < 0.01:
        p = f"= {Decimal(p):.2e}"
    else:
        p = f" = {round(p, 3)}"

    ax[1, 2].set_title(f"r = {round(r, 3)}, p {p}", fontsize=size_plt)

    # TC
    sns.regplot(x=rem_pheno_tc["func_mean_fd"], y=mFC[0], scatter_kws={
                's': 8}, ax=ax[2, 2], color=col_tc)
    ax[2, 2].set_xlabel("Mean FD", fontsize=size_plt)
    ax[2, 2].set_ylabel("Mean FC", fontsize=size_plt)
    # ax[2, 2].text(0.8, 0.9, f'n = {len(mFC[0])}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[2, 2].transAxes)
    r, p = stats.pearsonr(x=rem_pheno_tc["func_mean_fd"], y=mFC[0])

    if p < 0.01:
        p = f"= {Decimal(p):.2e}"
    else:
        p = f" = {round(p, 3)}"

    ax[2, 2].set_title(f"r = {round(r, 3)}, p {p}", fontsize=size_plt)

    # ALL
    sns.regplot(x=all_mFD, y=all_mFC, scatter_kws={
                's': 8}, ax=ax[3, 2], color=col_all)
    ax[3, 2].set_xlabel("Mean FD", fontsize=size_plt)
    ax[3, 2].set_ylabel("Mean FC", fontsize=size_plt)
    # ax[3, 2].text(0.8, 0.9, f'n = {len(all_mFC)}', horizontalalignment='center',
    #               verticalalignment='center', fontsize=size_pcp, transform=ax[3, 2].transAxes)
    r, p = stats.pearsonr(x=all_mFD, y=all_mFC)

    if p < 0.01:
        p = f"= {Decimal(p):.2e}"
    else:
        p = f" = {round(p, 3)}"

    ax[3, 2].set_title(f"r = {round(r, 3)}, p {p}", fontsize=size_plt)

    # ________ Correlation FD and FC as a function of distance ________

    # FD from all remaining participants.
    rem_pcp_fd = this_dat["func_mean_fd"]

    # Vectorised upper triangular of remaining participants:
    rem_vec_all = np.array(this_dat["vectorised matrix"].values.tolist())

    # Calculate the correlation between edges and FD accross participants.
    corr_fd_fc = np.zeros(rem_vec_all.shape[1])
    for i in range(rem_vec_all.shape[1]):
        corr_fd_fc[i] = stats.pearsonr(rem_vec_all[:, i], rem_pcp_fd)[0]

    # Put the correlation coefficients and the distances in one dataframe.
    corr_dist = pd.DataFrame({"corr_fd_fc": corr_fd_fc, "distance": cond_dist_mat})
    corr_dist = corr_dist.dropna()  # Get rid of the rows that have NaN values.
    
    values = np.vstack([corr_dist["corr_fd_fc"], corr_dist["distance"]])
    kernel = stats.gaussian_kde(values)(values)

    sns.scatterplot(corr_dist,
                    x="distance",
                    y="corr_fd_fc",
                    c=kernel,
                    cmap="viridis",
                    ax=ax[0, 2])

    sns.regplot(data=corr_dist, x="distance", y="corr_fd_fc",
                scatter=False, ax=ax[0, 2], color="red")
    
    ax[0, 2].set_xlabel("Distance (mm)", fontsize=size_plt)
    ax[0, 2].set_ylabel("Corr. FC and FD (r)", fontsize=size_plt)
    
    r, p = stats.pearsonr(x=corr_dist["distance"], y=corr_dist["corr_fd_fc"])

    if p < 0.01:
        p = f"= {Decimal(p):.2e}"
    else:
        p = f" = {round(p, 3)}"

    # round them cause we don't need a tone of decimals
    ax[0, 2].set_title(f"r = {round(r, 3)}, p {p}", fontsize=size_plt)
    
    
    red_line = Line2D([0], [0], linewidth=3,  label=f'ASD (n = {len(rem_pheno_asd["mean FC"])})', color='lightcoral')
    blue_line = Line2D([0], [0], linewidth=3,  label=f'TC (n = {len(rem_pheno_tc["mean FC"])})', color='cornflowerblue')
    purple_line =  Line2D([0], [0], linewidth=3,  label=f'All participants (n = {len(all_mFC)})', color = col_all)
    
    fig.legend(handles=[red_line, blue_line, purple_line], fontsize = plt_size)

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # –––––––––––––––––– EDGES WEIGHTS FOR NR, MR and GSR –––––––––––––––––––––
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    col_all = "darkslateblue"
    id_asd = rem_pheno_asd["subject"]
    id_tc = rem_pheno_tc["subject"]
    id_all = pd.concat([id_asd , id_tc])
    
    ts_all_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(id_all)]
    ts_all_nr = all_pcp_nr[all_pcp_nr["ID"].isin(id_all)]
    ts_all_gsr.index = np.arange(len(ts_all_gsr))
    ts_all_nr.index = np.arange(len(ts_all_nr))

    all_measures = graph_measures(ts_all_nr, ts_all_gsr, prop_threshold = 0.15,  # THE MEASURES CALCULATES PREVIOUSLY COULD BE REUSED TO SAVE TIME INSTEAD OF RE-RUNNING THIS
                                  phenotypic=phenotypic, exclusion=False)

    # Average the vectorised upper triangulars of the connectivity matrices
    # for all three conditions (NR, MR and GSR).
    vec_nr = all_measures["group_nr"]["vectorised matrix"].tolist()
    vec_gsr = all_measures["group_gsr"]["vectorised matrix"].tolist() 
    vec_mreg = all_measures["group_mreg"]["vectorised matrix"].tolist() 


    mfc_nr = pd.DataFrame(np.mean(vec_nr, axis=0))
    mfc_gsr = pd.DataFrame(np.mean(vec_gsr, axis=0))
    mfc_mreg = pd.DataFrame(np.mean(vec_mreg, axis=0))


    # Create a figure with six subplots.
    fig2, ax2 = plt.subplots(2, 3, figsize=(14, 10))
    fig2.subplots_adjust(top=0.6, bottom=0.01, hspace=0.5, wspace=0.3)
    # Add a general title to the figure.
    fig2.suptitle(this_label,  fontsize=20, fontweight="bold", y=0.73)
    # Number of participants
    fig2.text(x=0.2, y=0.65,  s=f"n = {len(vec_gsr)}", size = size_plt )
    plt.grid(False)
    
    plt_size = 20
    
    # GSR in function of NR:
    ax2[0, 0].scatter(x=mfc_nr, y=mfc_gsr, s=2, c=col_all)
    ax2[0, 0].set_xlabel("NR edge weight", fontsize=plt_size)
    ax2[0, 0].set_ylabel("GSR edge weight", fontsize=plt_size)

    # MR in function of NR:
    ax2[0, 1].scatter(x=mfc_nr, y=mfc_mreg, s=2, c=col_all)
    ax2[0, 1].set_xlabel("NR edge weight", fontsize= plt_size)  # averaged weights
    ax2[0, 1].set_ylabel("MR edge weight ", fontsize= plt_size)  # averaged weights

    # MR in function of GSR:
    ax2[0, 2].scatter(x=mfc_gsr, y=mfc_mreg, s=2, c=col_all)
    ax2[0, 2].set_xlabel("GSR edge weight", fontsize=plt_size)
    ax2[0, 2].set_ylabel("MR edge weight", fontsize=plt_size)

    # ____ edge distribution ____

    # NR 
        # averaged edges
    mfc_nr.plot(kind = "kde", ax = ax2[1, 0], legend=False, color = col_all)
    
        # edges of pcp with max mean FC
    meas = all_measures["group_nr"]
    maxi = pd.DataFrame(meas["vectorised matrix"][meas['mean FC'].idxmax()])
    maxi.plot(kind = "kde", ax = ax2[1, 0], legend=False, color = "red")
    
        # edges of pcp with min mean FC
    mini = pd.DataFrame(meas["vectorised matrix"][meas['mean FC'].idxmin()])
    mini.plot(kind = "kde", ax = ax2[1, 0], legend=False, color = "blue")
     
    
    # MR
    mfc_mreg.plot(kind = "kde", ax = ax2[1, 1], legend=False , color = col_all)

        # edges of pcp with max mean FC
    meas = all_measures["group_mreg"]
    maxi = pd.DataFrame(meas["vectorised matrix"][meas['mean FC'].idxmax()])
    maxi.plot(kind = "kde", ax = ax2[1, 1], legend=False, color = "red")
    
        # edges of pcp with min mean FC
    mini = pd.DataFrame(meas["vectorised matrix"][meas['mean FC'].idxmin()])
    mini.plot(kind = "kde", ax = ax2[1, 1], legend=False, color = "blue")


    # MR
        # averaged edges
    mfc_gsr.plot(kind = "kde", ax = ax2[1, 2], legend=False , color = col_all)

        # edges of pcp with max mean FC
    meas = all_measures["group_gsr"]
    maxi = pd.DataFrame(meas["vectorised matrix"][meas['mean FC'].idxmax()])
    maxi.plot(kind = "kde", ax = ax2[1, 2], legend=False, color = "red")
    
        # edges of pcp with min mean FC
    mini = pd.DataFrame(meas["vectorised matrix"][meas['mean FC'].idxmin()])
    mini.plot(kind = "kde", ax = ax2[1, 2], legend=False, color = "blue")

    ax2[1, 0].set_ylim(ax2[1, 2].get_ylim())
    ax2[1, 1].set_ylim(ax2[1, 2].get_ylim())

    ax2[1, 0].set_xlabel("NR edge weight", fontsize=plt_size)
    ax2[1, 1].set_xlabel("MR edge weight", fontsize=plt_size)
    ax2[1, 2].set_xlabel("GSR edge weight", fontsize=plt_size)
    
    ax2[1, 0].set_ylabel("Density", fontsize=plt_size)
    ax2[1, 1].set_ylabel("Density", fontsize=plt_size)
    ax2[1, 2].set_ylabel("Density", fontsize=plt_size)
    
    red_line = Line2D([0], [0], linewidth=3,  label='Max mean FC', color='red')
    blue_line = Line2D([0], [0], linewidth=3,  label='Min mean FC', color='blue')
    purple_line =  Line2D([0], [0], linewidth=3,  label=f'Average (n = {len(meas)})', color= col_all)
    
    fig2.legend(handles=[red_line, blue_line, purple_line], fontsize = plt_size)

    plt.show()
    
    
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # –––––––––––––––––– STATISTICAL ANALYSIS –––––––––––––––––––––––––––––––––
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    
    # Corresponding NR time series.
    ts_asd_nr = all_pcp_nr[all_pcp_nr["ID"].isin(id_asd)]
    # Corresponding GSR ts.
    ts_asd_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(id_asd)]
    ts_asd_nr.index = np.arange(len(ts_asd_nr))
    ts_asd_gsr.index = np.arange(len(ts_asd_gsr))

    # Corresponding NR time series.
    ts_tc_nr = all_pcp_nr[all_pcp_nr["ID"].isin(id_tc)]
    # Corresponding GSR ts.
    ts_tc_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(id_tc)]
    ts_tc_nr.index = np.arange(len(ts_tc_nr))
    ts_tc_gsr.index = np.arange(len(ts_tc_gsr))

    size_plt = 20
    n_meas = 11
    
    # Store the results of the Mann Whit U test (Rank biserial and p-val).
    pval_frame1 = np.zeros((9,n_meas)) #.astype(str)
    rrb_frame = np.zeros((9,n_meas))
    combination1 = np.zeros((9,n_meas)).astype(str)
    
    # Store the results of the Correlation test (Pearson and p-val).
    pval_frame2 = np.zeros((9,n_meas))
    r_frame = np.zeros((9,n_meas))
    combination2 = np.zeros((9,n_meas)).astype(str)
    
    
    two_fig = ["mann", "linreg"]

    for figure in two_fig:


        

        s = ["eigen_c", "assortativity", "clustering","transitivity", 
                      "char_path", "efficiency", "betweenness", "diameter", 
                      "radius", "s_information" , "m_f_p_t"]
        
        groups = [ "group_nr", "group_mreg", "group_gsr"]
        thresholds = [0.05, 0.15, 0.25]

        

    # –––––––––––––––––

        i1 = 0

        y = -1 
        
        tc_measures = {}
        asd_measures = {}
        all_measures = {}
        
        for this_thresh in thresholds:

            if figure == "mann":

                tc_measures[this_thresh] = graph_measures(ts_tc_nr, ts_tc_gsr, 
                                                 prop_threshold=this_thresh, 
                                                 phenotypic=phenotypic, 
                                                 exclusion=False)
                    
                asd_measures[this_thresh] = graph_measures(ts_asd_nr, ts_asd_gsr, 
                                                  prop_threshold=this_thresh, 
                                                  phenotypic=phenotypic, 
                                                  exclusion=False)

            else: 

                all_measures[this_thresh] = graph_measures(ts_all_nr, ts_all_gsr, 
                                                  prop_threshold = this_thresh, 
                                                  phenotypic=phenotypic, 
                                                  exclusion=False)
                    
                
        for this_group in groups:
            
            for this_thresh in thresholds:
            
                y += 1
                
                z = 0
                        
            
                for this_plot in list_plots: # Loop through the graph measures.    
                
                    
                    
                    if figure == "mann":
    
                        tc = tc_measures[this_thresh][this_group][this_plot].tolist()
    
                        asd = asd_measures[this_thresh][this_group][this_plot].tolist()
    
                        label = " "
    
                        data = [tc, asd]
    
                        mann_whi = stats.mannwhitneyu(asd, tc)
                        rbc = float(pg.mwu(asd, tc)["RBC"])
                        rbc = round(rbc, n_round)
                        
                        p_value = mann_whi[1]
    
                        col = "on"
    
                        if p_value > 0.1:
                            col = "off"
    
                        if p_value < 0.01:
                            p_value = f"{Decimal(p_value):.2e}"  # "< 0.01"
    
                        else:
                            p_value = f"{round(p_value, 3)}"
                            
                        combination1[y,z] = this_group + "_" + str(this_thresh) + "_" + this_plot
                        pval_frame1[y,z] = float(p_value)
                        rrb_frame[y,z] = rbc
                        
                        
                            
                    else:
                        all_fc = all_measures[this_thresh][this_group]["mean FC"]
                        all_meas = all_measures[this_thresh][this_group][this_plot]
    
                        label = " "  # "Global clustering"
    
                        spearmanr = stats.spearmanr(all_fc, all_meas)
    
                        r_value = spearmanr[0]
                        r_value = round(r_value, n_round)
                        p_value = spearmanr[1]
    
                        if p_value > 0.05:
                            col_all = "darkgrey"
                            
                        else: col_all = "darkslateblue"
    
                        if p_value < 0.01:
    
                            p_value = f"{Decimal(p_value):.2e}"  # " < 0.01"
    
                        else:
                            p_value = f"{round(p_value, 3)}"
                            
                            
                        combination2[y,z] = this_group + "_" + str(this_thresh) + "_" + this_plot
                        pval_frame2[y,z] = float(p_value)
                        r_frame[y,z] = r_value
                        
                        
                        
                        
                    i1 += 1
                    z += 1
                    
                    print(f"y = {y}")
                    print(f"z = {z}")
                    
                    print(f"_____________ ITERATION {i1} _____________")
    
        plt.show()
        
        
        # __________ HEATMAP RESULTS __________
        
        
        if figure == "mann": 
            result_frame = rrb_frame.copy()
            p_val_frame = pval_frame1.copy()
            combin_frame = combination1.copy()
        else: 
            result_frame = r_frame.copy()
            p_val_frame = pval_frame2.copy()
            combin_frame = combination2.copy()
            
           
        # ––––––––––––––––   statistic   ––––––––––––––––
        
        # Reverse the arrays along the Y axis such that they are in the desired 
        # order in the plot (because plt.colormesh reverses it, so it ends up 
        # being in the original order).
        result_frame = np.flip(result_frame, axis = 0)
        p_val_frame = np.flip(p_val_frame, axis = 0)
        combin_frame = np.flip(combin_frame, axis = 0)
        
        plt.rcParams["figure.figsize"] = (25,20)
        
        # Visualise the colormap for the group differences 
        cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
        pc = plt.pcolormesh(result_frame,  # Plot the Rank biserial values. 
                            norm=colors.CenteredNorm(), # Center the colorbar around 0.
                            cmap=cmap)  # Colorbar previoulsy selected. 
        
        
        
        # Show ticks and labels.
        measures = ["eigen. centrality", "assortativity", "clustering","transitivity", 
                    "char. path length", "efficiency", "bet. centrality",
                      "diameter", "radius", "search info." , "first passage t."]
        
        n_thresh = 3
        l_width = 0.6
        plt.axhline(n_thresh, color="black", linestyle="-", linewidth = l_width)
        plt.axhline(n_thresh*2, color="black", linestyle="-", linewidth = l_width)
        plt.axhline(n_thresh*3, color="black", linestyle="-", linewidth = l_width)
        s = 31
        
        
        
        # Adding the values of the Rrb corrlation value on top of the colormap. 
        for i in range(9):
            for j in range(len(list_plots)):
                text = plt.text(j+0.55, i+0.5, round(result_frame[i, j],2), size = s, 
                                ha="center", va="center", color="black")
                
                # Adding a "*" if the p-value is significant. 
                if 0.0055 < float(p_val_frame[i,j]) < 0.05: 
                    text = plt.text(j+0.55, i+0.75, "*", size = s, 
                                    ha="center", va="center", color="black")
                    
                elif 0.0019  < float(p_val_frame[i,j]) <= 0.0055: 
                    text = plt.text(j+0.55, i+0.75, "**", size = s, 
                                    ha="center", va="center", color="black")
                    
                elif float(p_val_frame[i,j]) <= 0.0019: 
                    text = plt.text(j+0.55, i+0.75, "***", size = s, 
                                    ha="center", va="center", color="black")
                    
            
        
        
        cbar = plt.colorbar()                         # Add colorbar. 
        cbar.ax.tick_params(labelsize=35)             # Make colobar ticks bigger.
        
        if figure == "mann": 
            cbar.set_label(label = r"r$_{rb}$", size=50)  # Add colorbar label. 
            plt.title( 'Group differences', y = 1.1, size = 30)
            
        else:
            cbar.set_label(label = "r$_{s}$", size = 50)        # Add colorbar label.
            plt.title( 'Correlation', y = 1.1, size = 30)

        
        plt.show()
        
        # ––––––––––––––––   p-values   ––––––––––––––––
        plt.rcParams["figure.figsize"] = (50,25)
        
        # p_signif = np.zeros((9,n_meas)
        
        # for i in range(9):
        #     for j in range(len(list_plots)):
                
        #         if p_val_frame[i, j] < 0.05: 
        #             p_signif[i, j] = 1/4
            
        
        # Visualise the colormap for the group differences 
        cmap = plt.get_cmap('binary') # The colormap (red/blue)
        pc = plt.pcolormesh(np.zeros((9,n_meas)),  # Plot the Rank biserial values. 
                            norm=colors.CenteredNorm(vcenter = 0.5), # Center the colorbar around 0.
                            cmap=cmap)  # Colorbar previoulsy selected. 
        
        
    
        
        s = 36
        
        plt.axhline(n_thresh, color="black", linestyle="--", linewidth = l_width)
        plt.axhline(n_thresh*2, color="black", linestyle="--", linewidth = l_width)
        plt.axhline(n_thresh*3, color="black", linestyle="--", linewidth = l_width)
        
        
        # Adding the values of the Rrb corrlation value on top of the colormap. 
        for i in range(9):
            for j in range(len(list_plots)):
                text = plt.text(j+0.55, i+0.5, p_val_frame[i, j], size = s, 
                                ha="center", va="center", color="black")
                
                # Adding a "*" if the p-value is significant. 
                if 0.0055 < p_val_frame[i,j] < 0.05: 
                    text = plt.text(j+0.55, i+0.75, "*", size = s, 
                                    ha="center", va="center", color="black")
                    
                elif 0.0019  < p_val_frame[i,j] <= 0.0055: 
                    text = plt.text(j+0.55, i+0.75, "**", size = s, 
                                    ha="center", va="center", color="black")
                    
                elif p_val_frame[i,j] <= 0.0019: 
                    text = plt.text(j+0.55, i+0.75, "***", size = s, 
                                    ha="center", va="center", color="black")
                    
                    
                
        cbar = plt.colorbar()                         # Add colorbar. 
        cbar.ax.tick_params(labelsize=35)             # Make colobar ticks bigger.
        plt.xticks(np.arange(len(measures)), measures, rotation = 70, size = 40)
        plt.yticks(color = 'w')
         
        if figure == "mann": 
            cbar.set_label(label = r"r$_{rb}$", size=50)  # Add colorbar label. 
            plt.title( 'Group differences (p-values)', y = 1.1, size = 30)
             
        else:
            cbar.set_label(label = "r$_{s}$", size = 50)        # Add colorbar label.
            plt.title( 'Correlation (p-values)', y = 1.1, size = 30)
                
        plt.show()
                
                
                    
                    
        # ___________ Measures of relative contributions ___________
        
        result_frame = np.flip(result_frame, axis = 0)
        p_val_frame = np.flip(p_val_frame, axis = 0)
        combin_frame = np.flip(combin_frame, axis = 0)
        
    
        # Calculate the difference between the average Rrb and every Rrb.
        abs_result_frame = np.absolute(result_frame.astype(float)) # Taking the absolute value.
        avr_result = np.mean(abs_result_frame, axis = 0)   # Average accross groups and thresholds.
        difference = pd.DataFrame(abs_result_frame - avr_result) # Take the diff between the average and every row.
        combin_frame = pd.DataFrame(combin_frame) # Putting the combination into a dataframe.
        
        listi = [0,3,6]
        dat_box = pd.DataFrame() 
        combin_box = pd.DataFrame()
        marker_dict = {'5%': 'o', '15%': 's', '25%': '^'}
        marker_list = ['o', 's', '^']

        mark_size = 18
        j=0
        font_size = 20
        
        plt.figure(figsize=(6.4,4.8))
        col_list = ['darkorange', 'indianred', 'seagreen']
        color_dic = {'NR': u'darkorange', 'MR': u'indianred', 'GSR': u'seagreen'}


        # - Group by standardisation method. 
        for i in listi:
            # Put the values of several rows in the same row (3 first rows, then 3:5, 
            # then 6:8 = NR together, same for MR and GSR).
            dat_box[i] = sum(difference[i:i + 3].values.tolist(), []) # Store this in a dataframe (so it'll only have three rows instead of 9)
            combin_box[i] = sum(combin_frame[i:i + 3].values.tolist(), []) 
            
            # Now for every group (NR, MR and GSR), add the scatter of the datapoints for every 
            # threshold separately (row 0,1,2 of the original dataset, then 3,4,5 and 6,7,8)
            plt.scatter(x = difference.loc[[0 + i]], y = np.repeat(3.1 - j, difference.loc[[0 + i]].shape[1]), marker = marker_dict["5%"], s = mark_size, color = col_list[j])
            plt.scatter(x = difference.loc[[1 + i]], y = np.repeat(3 - j, difference.loc[[0 + i]].shape[1]), marker = marker_dict["15%"], s = mark_size, color = col_list[j])
            plt.scatter(x = difference.loc[[2 + i]], y = np.repeat(2.9 - j, difference.loc[[0 + i]].shape[1]), marker = marker_dict["25%"], s =mark_size, color = col_list[j])
            
            j+=1
            print(i)
            
        medprops = dict(linewidth = 1.7)
        #dat_box[0] =1
        plt.boxplot(dat_box[dat_box.columns[::-1]], medianprops = medprops, vert = False, showfliers=False)
        plt.yticks(color = 'w')
        plt.xticks(size = font_size - 4, rotation = 25)
        
        mark_size = 7
        black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                  markersize=mark_size, label='5%')
        black_square = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                                  markersize=mark_size, label='15%')
        black_triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                  markersize=mark_size, label='25%')
        
        nr_patch = mpatches.Patch(color=col_list[0], label='NR')
        mr_patch = mpatches.Patch(color=col_list[1], label='MR')
        gsr_patch = mpatches.Patch(color=col_list[2], label='GSR')

        
        plt.legend(handles=[black_dot, black_square, black_triangle], 
                   prop={'size': 13}, loc='center left', bbox_to_anchor=(1, 0.88))
        plt.axvline(0, color="grey", linestyle="--", linewidth = l_width)
        
        
        if figure == "mann": 
            plt.title( 'Group differences', y = 1.1, size = font_size)
            plt.xlabel(r'$\Delta$ |$r_{rb}$|', size = font_size)
            
        else:
            plt.title( 'Correlation', y = 1.1, size = font_size)
            plt.xlabel(r'$\Delta$ |$r_{s}$|', size = font_size)
        
        plt.show()
        
        mean_diff_groups = {"NR" : np.mean(dat_box[0]), "MR": np.mean(dat_box[3]), "GSR": np.mean(dat_box[6])}
        
        # - Group by threshold. 
        plt.figure(figsize=(6.4,4.8))
        dat_box = pd.DataFrame()
        combin_box = pd.DataFrame()
        color_dic = {'NR': u'darkorange', 'MR': u'indianred', 'GSR': u'seagreen'}
        thresh_list = ['5%', "15%", "25%"]
        mark_size = 18
        j=0
        font_size = 20
        

        for i in range(3):
            # Put the values of several rows in the same row (3 first rows, then 3:5, 
            # then 6:8 = NR together, same for MR and GSR).
            dat_box[i] = sum(difference.loc[[0 + i]].values.tolist(), []) + sum(difference.loc[[3 + i]].values.tolist(), []) + sum(difference.loc[[6 + i]].values.tolist(), []) # Store this in a dataframe (so it'll only have three rows instead of 9)
            combin_box[i] = sum(combin_frame.loc[[0 + i]].values.tolist(), [])+ sum(combin_frame.loc[[3 + i]].values.tolist(), []) + sum(combin_frame.loc[[6 + i]].values.tolist(), []) # !!!!!!!!!!!!!!!!!!!!!!!!!! find right way to reverse this !!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # Now for every group (NR, MR and GSR), add the scatter of the datapoints for every 
            # threshold separately (row 0,1,2 of the original dataset, then 3,4,5 and 6,7,8)
            plt.scatter(x = difference.loc[[0 + i]], y = np.repeat(3.1 - i, difference.loc[[0 + i]].shape[1]),  marker = marker_list[j], s = mark_size, color = 'darkorange') # marker = marker_dict[thresh_list[i]],
            plt.scatter(x = difference.loc[[3 + i]], y = np.repeat(3 - i, difference.loc[[0 + i]].shape[1]), marker = marker_list[j], s = mark_size, color = 'indianred')
            plt.scatter(x = difference.loc[[6 + i]], y = np.repeat(2.9 - i, difference.loc[[0 + i]].shape[1]), marker = marker_list[j], s =mark_size, color = 'seagreen')
            
            j+=1
            print(0+i)
            
        medprops = dict(linewidth = 1.7)
        #dat_box[0] =1
        plt.boxplot(dat_box[dat_box.columns[::-1]], medianprops = medprops, vert = False, showfliers=False)
        plt.yticks(color = 'w')
        plt.xticks(size = font_size - 4, rotation = 25)
        #mark_size = 30
        plt.legend(handles=[nr_patch, mr_patch, gsr_patch], 
                   prop={'size': 13}, loc='center left', bbox_to_anchor=(1, 0.88))        
        plt.axvline(0, color="grey", linestyle="--", linewidth = l_width)
        
        
        if figure == "mann": 
            plt.title( 'Group differences', y = 1.1, size = font_size)
            plt.xlabel(r'$\Delta$ |$r_{rb}$|', size = font_size)
            
        else:
            plt.title( 'Correlation', y = 1.1, size = font_size)
            plt.xlabel(r'$\Delta$ |$r_{s}$|', size = font_size)
        
        plt.show()
        
        mean_diff_thresh = {"5%" : np.mean(dat_box[0]), "15%": np.mean(dat_box[1]), "25%": np.mean(dat_box[2])}




        # __________ RAINCLOUD PLOTS __________
        # Raincloud of the groups with highest and lowest effect sizes
        
            
        if figure == "mann": 
            result_frame = rrb_frame.copy()
            p_val_frame = pval_frame1.copy()
            combin_frame = combination1.copy()
        else: 
            result_frame = r_frame.copy()
            p_val_frame = pval_frame2.copy()
            combin_frame = combination2.copy()
            
            
        max_ind = unravel_index(result_frame.argmax(), result_frame.shape)
        max_comb = combin_frame[max_ind[0],max_ind[1]]
        max_meas = "_".join(max_comb.split("_")[3:]) # [3:] -> exclude the first 3 items ("group", "nr", "thresholds")
                                                     # "_".join -> join the items in the list into one string with "_" instead of the comas. 
        
        min_ind = unravel_index(result_frame.argmin(), result_frame.shape)
        min_comb = combin_frame[min_ind[0],min_ind[1]]
        min_meas = "_".join(min_comb.split("_")[3:]) 
        
        zero_ind = unravel_index(np.absolute(result_frame).argmin(), result_frame.shape)
        zero_comb = combin_frame[zero_ind[0],zero_ind[1]]
        zero_meas = "_".join(zero_comb.split("_")[3:]) 
        
        
        i1 = 0
        
        ii = np.arange(start=0, stop=3, step=1)  # Range of values of eta.
        jj = np.arange(start=0, stop=9, step=1)  # Range of values of gamma.
    
        # Repeating the ranges, so that we'll get every possible combination of eta-gamma
        # there are XXXX values of each, so we end up with an array of XXXX values
        iis = np.tile(ii, len(jj))  # ([XXXXX]).
        jjs = np.repeat(jj, len(ii))  # ([XXXXX])].
        axes = list(zip(jjs, iis))
        fig, ax = plt.subplots(9, 3, figsize=(20, 35))
        plt.subplots_adjust(top=0.7, bottom=0.02, hspace=0.55, wspace=0.2)

        y=0.73
        
        fig.text(s= min_meas,  x=0.28,  y=y, fontsize=25)
        fig.text(s=zero_meas,  x=0.55,  y=y, fontsize=25)
        fig.text(s=max_meas,  x=0.83,  y=y, fontsize=25)

        x = 0.055
        fig.text(s="5% ",  x=x,  y=0.67, rotation=90, fontsize=25)
        fig.text(s="15% ",  x=x,  y=0.59, rotation=90, fontsize=25)
        fig.text(s="25%",  x=x,  y=0.51, rotation=90, fontsize=25)
        fig.text(s="NR",  x=0.01,  y=0.593, rotation=90, fontsize=25)

        fig.text(s="5% ",  x=x,  y=0.43, rotation=90, fontsize=25)
        fig.text(s="15% ",  x=x,  y=0.352, rotation=90, fontsize=25)
        fig.text(s="25% ",  x=x,  y=0.274, rotation=90, fontsize=25)
        fig.text(s="MR",  x=0.01,  y=0.355, rotation=90, fontsize=25)

        fig.text(s="5%",  x=x,  y=0.198, rotation=90, fontsize=25)
        fig.text(s="15%",  x=x,  y=0.12, rotation=90, fontsize=25)
        fig.text(s="25%",  x=x,  y=0.04, rotation=90, fontsize=25)
        fig.text(s="GSR",  x=0.01,  y=0.12, rotation=90, fontsize=25)
        
        
                    
                
        for this_group in groups:
            
            for this_thresh in thresholds:
            
                for this_plot in [min_meas,zero_meas, max_meas]:
        
                    if figure == "mann":    
                        
                        
                        
                        tc = tc_measures[this_thresh][this_group][this_plot].tolist()
            
                        asd = asd_measures[this_thresh][this_group][this_plot].tolist()
            
                        label = " "
            
                        data = [tc, asd]
            
                        mann_whi = stats.mannwhitneyu(tc, asd)
                        rbc = float(pg.mwu(tc, asd)["RBC"])
                        rbc = round(rbc, n_round)
                        
                        p_value = mann_whi[1]
            
                        col = "on"
            
                        if p_value > 0.1:
                            col = "off"
            
                        if p_value < 0.01:
                            p_value = f"{Decimal(p_value):.2e}"  # "< 0.01"
            
                        else:
                            p_value = f"{round(p_value, 3)}"
                            
                        
                        
                            
                        axx = axes[i1]
                                
                        title = r"r$_{rb}$ = " + f"{rbc}, p = {p_value}"
                                
                        raincloud(data_x=data, feature1="TC", feature2="ASD",  x_label=label,
                                  title = title, colors=col, ax=ax[axx], size=size_plt)
            
                        i1 += 1
                            
                                                    
                        
                        
                    else:
                        
                        
                        all_fc = all_measures[this_thresh][this_group]["mean FC"]
                        all_meas = all_measures[this_thresh][this_group][this_plot]
            
                        label = " "  # "Global clustering"
            
                        spearmanr = stats.spearmanr(all_fc, all_meas)
            
                        r_value = spearmanr[0]
                        r_value = round(r_value, n_round)
                        p_value = spearmanr[1]
                        
                        if p_value > 0.05:
                            col_all = "darkgrey"
                            
                        else: col_all = "darkslateblue"
            
                        if p_value < 0.01:
            
                            p_value = f"{Decimal(p_value):.2e}"  # " < 0.01"
            
                        else:
                            p_value = f"{round(p_value, 3)}"
                            
                        
                        
                        
                        axx = axes[i1]
                         
                            
                        sns.regplot(x=all_fc, y=all_meas, scatter_kws={'s': 5},
                                    color=col_all, ax=ax[axx]).set_title(label= r"r$_{rb}$" + f" = {r_value}, p = {p_value}", fontsize = size_plt)
        
                        ax[axx].tick_params(axis="both", labelsize=20)
                        ax[axx].set_xlabel(label)
                        ax[axx].set_ylabel(label)
                        
                        i1 += 1
                        
                        
                        
        plt.show()

        # __________ Correlation between effect sizes __________
        # Calculate the correlation between the rank biserial stats and the 
        # spearman correlations.
        
        colors1 = ["black", "darkorange","indianred","seagreen"]
        
        
        # Overall ("all"), nr (0:3), mr (3:6), gsr (6:9).
        
        j = 0
        for i in ["all", 0,3,6]:
            
            if i == "all": 
                vect_rrb = rrb_frame.flatten() # Flatten the rank biserial correlations. 
                vect_rs = r_frame.flatten() # Flatten the spearman correlations.
                print(combination1)
                
            else:
                vect_rrb = rrb_frame[i:i+3].flatten() # Flatten the rank biserial correlations. 
                vect_rs = r_frame[i:i+3].flatten() # Flatten the spearman correlations.
                print(combination1[i:i+3])
            
            plt.figure(figsize=(6.4,4.8))
            sns.regplot(x=vect_rrb, y=vect_rs, scatter_kws={  # Plot the linear reg between the two. 
                        's': 14},  color=colors1[j])
            
            s = 20
            rs_stat = stats.spearmanr(vect_rrb, vect_rs)[0] # Spearman corr between the two effect sizes. 
            p_stat = stats.spearmanr(vect_rrb, vect_rs)[1] # The corresponding p-value.
            if p_stat < 0.01 : p_stat = f"{Decimal(p_stat):.2e}"
            else: p_stat = f"{round(p_stat, 2)}"
            plt.title(r"r$_{s}$" + f" = {round(rs_stat, 2)}, p = {p_stat}", size = s)
            
            plt.yticks(size = 15)
            plt.ylabel(r"r$_{s}$", size = s)
            plt.xticks(size = 15, rotation = 30)
            plt.xlabel(r"r$_{rb}$", size = s)
            
            
            plt.show()
        
            j+=1
        
        # Plot the legend separately.
        line_w = 8
        black_line = Line2D([0], [0], linewidth=3,  label='Overall', color='black', lw = line_w)
        red_line = Line2D([0], [0], linewidth=3,  label=f'NR', color='darkorange', lw = line_w)
        blue_line = Line2D([0], [0], linewidth=3,  label=f'MR', color='indianred', lw = line_w)
        purple_line =  Line2D([0], [0], linewidth=3,  label=f'GSR', color = "seagreen", lw = line_w)
        
        plt.figure(figsize=(10,10))
        plt.legend(handles=[black_line, red_line, blue_line, purple_line], fontsize = plt_size, ncol=4) # "ncol=4" to make 4 colomns -> horizontal alignment. 
        
        
    # __________ Correlation between graph measures __________
    # Get the correlation between each pair of graph measures accross the NR, 
    # MR and GSR connectomes, for one threshold (15%).
        
        
        
    for group in groups:  # "groups" has been defined higher up -> list of groups (NR, MR, GSR).
        
        s = 18
        
        df = np.corrcoef(all_measures[0.15][group][list_plots].transpose()) # Correlation between all the graph measures.
        headers =  measures.copy() # measures was defined higher -> list of the measures.
        
        # following lines: stackoveflow "Itachi" (see references)
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
    
        # # Want diagonal elements as well
        # mask[np.diag_indices_from(mask)] = False
    
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
    
        # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        cmap = plt.get_cmap('coolwarm')
        
        # Draw the heatmap with the mask and correct aspect ratio
        y_headers = headers.copy()
        y_headers[0] = " " # hide the first header of the y axis since we're hiding the diagonal.
        ax2 = sns.heatmap(df, mask=mask, cmap=cmap, xticklabels = headers[:-1], yticklabels = y_headers, vmin=-1, vmax=1, cbar_kws={'label':r"r$_{s}$"})
        ax2.figure.axes[-1].yaxis.label.set_size(s)
        cax = ax2.figure.axes[-1]
        cax.tick_params(labelsize=s)
        plt.tick_params(labelsize = s, color = "w")
        plt.title(group)
        
        s = 14
        
        for i in range(11):
            for j in range(11):
                if mask[i,j] == False:
                    
                    x = np.round(df[i, j],2)
                    text = plt.text(j+0.55, i+0.5, f'{x:.2f}' , size = s, 
                                    ha="center", va="center", color="black")
                    
                        
                        
                    
                    
                    
