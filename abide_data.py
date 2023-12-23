#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   AUTHOR: Clara Riégis
     NAME: abide_data.py
  PURPOSE: data download and inspection, 
           Undergraduate project supervised by Dr. František Váša
QUESTIONS: Do graph theoretic measures depend on overall mean functional connectivity? 
           Is this affected by different pre-processing pipelines ? 
           Which of NR, MR and GSR lead to the best classification? 
           Is this effect specific to one atlas/ datasets? 


    OVERVIEW
    –––––––––––––––––––––––––––––––––––
    - 0 - Libraries
    - 1 - Dowload data 
    - 2 - Retrieve data   
    - 3 - time series inspection  
    - 4 - check parcellation process 
    - 5 - exclude ts causing issue   
    - 6 - exclude remaining pcps = 0 
    - 7 - Save data 
    –––––––––––––––––––––––––––––––––––
"""

# %%                - 0 - Libraries

import nibabel as nib
import _pickle as cPickle
from nilearn.datasets import fetch_abide_pcp
import pandas as pd
import glob
import numpy as np
from nilearn import image as nimg
from nilearn import input_data
from nilearn import plotting
import matplotlib.pyplot as plt
import os


# %%                - 1 - Dowload data 

# Using the 'fetch_abide_pcp()" function to download the ROIs time-series
# into a specified directory. Here, the ROIs parcellated according to the
# Harvard-Oxford atlad that passed the quality control.
# Can run the function twice: once to download with GSR, and once withOUT.
# https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_abide_pcp.html

abide = fetch_abide_pcp(data_dir="participants", derivatives=['rois_ho'],
                        global_signal_regression=False, pipeline='ccs',
                        band_pass_filtering=True, quality_checked=True)



# %%                - 2 - Retrieve data                

# Setting a path to access the GSR data:
path_gsr = 'participants/ABIDE_pcp/ccs/filt_global/*_*_rois_ho.1D'

# ____________ INITIALIZE VARIABLES ____________


all_loaded_id = []   # Store the IDs for all participants.
all_loaded_gsr = []  # Store the GSR time series for all participants.
all_loaded_nr = []   # Store the NR time series for all participants.

os.chdir('/Users/clarariegis/Downloads')
for filename in glob.glob(path_gsr):

    # ____________ PCP IDs ____________

    # Get the ID of the participant whose file is being loaded:
    loaded_id = int(filename.split('_')[-3][2:])
    # [-3] -> "-" is to start from the end of the split string. This is because
    #         the first part of the file names where not always the same.
    # [2:] -> Not take into account the first two numbers (00).

    all_loaded_id.append(loaded_id)

    # ____________ STORE GSR & NR time series ____________

    # GSR times series:
    all_loaded_gsr.append(np.loadtxt(filename))

    # NR path - based on the ID of the participant that was loaded
    # from the GSR data:
    path_nr = (filename.split('filt_global/')[0] +
               'filt_noglobal/' +
               filename.split('/')[4])
    # NR times series:
    all_loaded_nr.append(np.loadtxt(path_nr))


# GSR - storing the retrieved IDs and time series in one data frame:
all_pcp_gsr = pd.DataFrame({"ID": all_loaded_id,   # Participants IDs.
                           "ts": all_loaded_gsr})  # Times series corresponding
                                                   # to each ID.

# NR - storing the retrieved IDs and time series in one data frame:
all_pcp_nr = pd.DataFrame({"ID": all_loaded_id,    # Participants IDs.
                           "ts": all_loaded_nr})   # Times series corresponding
                                                   # to each ID.

# clean variable explorer:
del (all_loaded_id, all_loaded_gsr, all_loaded_nr, filename, loaded_id,
     path_gsr, path_nr)



# %%                - 3 - time series inspection        

# ISSUE: The number of time series (111) in each participants' file (parcellated
# using the Harvard-Oxford atlas) did not match the number of region labels (110).
# As raised by Zhongxing Zhou: https://www.nitrc.org/forum/message.php?msg_id=30606

# Loading the file with the ROIs labels obtained from the ABIDE dataset - it is
# the same on Cyberduck and on the ABIDE webpage.
ho_labels = pd.read_csv("participants/atlas/ho_labels.csv")  # -> 110 labels.

# Checking how many files has 111 time series <=> 111 ROIs.
length = []
for i in range(len(all_pcp_nr["ts"])):
    if all_pcp_gsr["ts"][0].shape[1] == 111:
        length.append(i)

if len(length) == len(all_pcp_gsr["ts"]):
    print("All the participants have 111 ROIs")



# It also seems that some of the time series only have "0" as values:
zeros_ts = []
for i in range(len(all_pcp_nr["ts"])):

    for j in range(all_pcp_nr["ts"][i].shape[1]):

        if np.all(all_pcp_nr["ts"][i][:, j] == 0):
            # print(i,j)
            zeros_ts.append([i, j])




# %%                - 4 - check parcellation process   


# - Conducting the parcellation on 1 participants to see if the result is the same.

# Load a nifti file for a random participant:
nii_nr_1pcp = nib.load("participants/func_preproc_noglobal/Yale_0050607_func_preproc.nii")


# Load the Harvard-Oxford Atlas:
ho_atlas = nib.load("participants/atlas/ho_roi_atlas.nii")

# Get the image labels. When return_label_names is True, the function returns
# the coordinates + the labels. Here I only stored the labels, hence the "[1]".
ho_img_labels = np.array(plotting.find_parcellation_cut_coords(
    ho_atlas, return_label_names=True)[1])

# Some online documentation pointed out that the "3455" label does not correspond
# to any brain region in the csv label file. Here we check where it is located:
# Index 82 -> consistent with online forum posts.
np.where(ho_img_labels == 3455)
# See Reference section at the end of the code to see the relevant documentation.

# Resampling the atlas to match the dimensions of the nifti file.
ho_resampled = nimg.resample_to_img(
    ho_atlas, nii_nr_1pcp, interpolation='nearest')

# Creating a mask with the H-O atlas.
masker_filt = input_data.NiftiLabelsMasker(labels_img=ho_resampled, detrend=False,
                                           low_pass=0.1, high_pass=0.01, t_r=2)


# Using the mask to average the ts for each ROI
ts_nr_1pcp = masker_filt.fit_transform(nii_nr_1pcp)
# Unsurprisingly the number of ts is the same (111).
print(ts_nr_1pcp.shape[1])

# Storing the time series of the same particpant but pre-parcellated (from ABIDE).
ts_abide_nr_1pcp = all_pcp_nr.loc[all_pcp_nr["ID"] == 50607, "ts"].tolist()[0]

fig, axs = plt.subplots((2))
fig.tight_layout(pad=4)
axs[0].plot(ts_nr_1pcp[:100, 82])
axs[0].set_title("ts 82 (my parcelation)")
axs[1].plot(ts_abide_nr_1pcp[:100, 82])
axs[1].set_title("ts 82 (abide parcelation)")


# %%                - 5 - exclude ts causing issue    


# - Exlude time series 82.

for i in range(len(all_pcp_nr["ts"])):

    all_pcp_nr["ts"][i] = np.delete(all_pcp_nr["ts"][i], 82, axis=1)

    all_pcp_gsr["ts"][i] = np.delete(all_pcp_gsr["ts"][i], 82, axis=1)



# Let's check again how many ts with only zeros there are after removing the
# ts 82 (which seemed to not match any ROI label).
print(f" There were {len(zeros_ts)} with only 0s.")
zeros_ts = []
for i in range(len(all_pcp_nr["ts"])):

    for j in range(all_pcp_nr["ts"][i].shape[1]):

        if np.all(all_pcp_nr["ts"][i][:, j] == 0):
            # print(i,j)
            zeros_ts.append([i, j])

print(
    f"After removing the ts 82, there are only {len(zeros_ts)} with only zeros left")




# %%                - 6 - exclude remaining pcps = 0    

# - Excluded the participants that have time series with
# only zeros despite having excluded the ts 82 that seemed to not correspond to
# any ROI.

# REMINDER: • so far all the participants are stores in two data frames
# containing there ID and the ts ("all_pcp_nr" = No Regression; "all_pcp_gsr" =
# Global Signal Regression).
# • "zeros_ts" contains the pcp index and the ts index of the ts with only 0s. .
zeros_ts2 = []
for i in range(len(zeros_ts)):
    zeros_ts2 += [zeros_ts[i][0]]

# Just get the single values. This is necessary because some pcps have more than
# one ts causing issue, but we can only excluded each participant once.
zeros_ts2 = list(set(zeros_ts2))

# Excluding the participants:
for i in range(len(zeros_ts2)):
    all_pcp_nr.drop(labels=zeros_ts2[i], axis=0, inplace=True)
    all_pcp_gsr.drop(labels=zeros_ts2[i], axis=0, inplace=True)


# Put the index back to being consecutive numbers - which is not the case since
# some participants were just excluded:
all_pcp_nr.index = np.arange(0, len(all_pcp_nr))
all_pcp_gsr.index = np.arange(0, len(all_pcp_gsr))


# Checking again how many participants have some time series equal to 0s - just
# to make sure none is remaining.
zeros_ts = []
for i in range(len(all_pcp_nr["ts"])):

    for j in range(all_pcp_nr["ts"][i].shape[1]):

        if np.all(all_pcp_nr["ts"][i][:, j] == 0):
            # print(i,j)
            zeros_ts.append([i, j])

print(f"Now {len(zeros_ts)} participant have time series with only zeros.")

# %%                - 7 - Save data  


with open(r"ts_nr.pickle", "wb") as output_file:
    cPickle.dump(all_pcp_nr, output_file)
    
    
with open(r"ts_gsr.pickle", "wb") as output_file:
    cPickle.dump(all_pcp_gsr, output_file)
    







