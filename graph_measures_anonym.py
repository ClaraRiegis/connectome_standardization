#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:37:16 2023

   @author: k20045146

     NAME: graph_measures.py

  PURPOSE: Function definition; build connectomes and generate graph measures. 


"""



import numpy as np
import bct as bct
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from mst_threshold import mst_threshold
none = pd.DataFrame()

def graph_measures (group_nr, group_gsr, prop_threshold, phenotypic, exclusion, fd_lim = 0.4 , group_pred = none):
    
    """
    
    PARAMETERS:
    ——————————
    group_nr : [DataFrame] IDs and time series of the raw 
                (Not Regressed - NR) data. 
    
    group_gsr : [DataFrame] IDs and time series of the GSR data.
    
    prop_threshold: [Float] Indicate the percentage of edges that should be 
                     retained (e.g. 0.05). 
                  
    group_pred : [DataFrame] Optional -  If a dataframe with IDs and ts added, 
                 then the slope and mean matrice of the group_nr are used to 
                 compute the regression. 
                 
    fd_lim :  max mean FD used to exclude participants. Default is 0.4
                 
    COMMENTS:
    ––––––––
    - The exclusion of mFD and mFC outliers only occurs when the optional 
      argument (group_pred) is NOT specified. In the case where it IS specified
      the data should have already been cleaned from outliers and be ready to 
      use. 
      
      
    OUTPUT:
    ––––––
    Dictionary containing a data frame for each of the pipelines. 
    Each data frame includes: ID, mean FC, vectorised matrix, global clustering, 
    assortativity and characteristic path length (for all participants in that 
    group). 
    
    """
    

    # The MR group (here "mreg") is originally the same as the NR group (and 
    # then the regression happens). 
    group_mreg = group_nr 
    
    
    if len(group_pred) == 0 : # If group_pred is not specified...`
        # ... we will be looping through / return the following groups:
            
        # ------------------- uncom ----------------------------------------------
        groups = ["group_nr", "group_gsr", "group_mreg"]
        
    else: # If group_pred is specified...
        # ...will be looping through / return the following groups:
        groups = ["group_gsr" , "group_gsr", "group_mreg", "group_pred"]
    
        
    # Initializing variables that should only be initialized ONCE 
    # (and NOT every time we loop through a new group). 
    dictionary = {}
    slopes = np.zeros((len(group_nr["ts"][0][0]),len(group_nr["ts"][0][0])))
    intercepts = np.zeros((len(group_nr["ts"][0][0]),len(group_nr["ts"][0][0])))
    
    
    
    for this_group in groups: # Looping through the different groups. 
    
        # Convert the name of the group (string) into an actual variable 
        # (dataframe):
        group = locals()[this_group] 
        # print(this_group)
        
        
        # ____________ INITIALIZE VARIABLES ____________
        
        all_final_id = []   # ID of every participant /that will be in the analysis/. 
        all_corr = []       # Correlation matrix of every participant //. 
        all_mn_fc = []      # Mean of the upper triangular of the matrix for every 
                            # participant //.
                            
        
        vec_triu = []       # Vectorised upper triu for every participant.    
        
        all_mn_fc = []      # Mean functional connectivity for every participant.
        
        
        cluster = []        # Store the graph measures for every participant. 
        charpath = []
        assort = []
        
        
        
        # ____________ CORRELATION MATRIX ____________
        
        # for every participant in the group. 
        for i in range(len(group)):  
            # print(i)
            # Create a correlation matrix. 
            # Taking the transpose of the array so that it correlates along the 
            # right axis. 
            corr = np.corrcoef(group["ts"][i].T)
            
            
            # Getting rid of participants with non finite values. 
            # QUESTION: Is that the right thing to do ?
            #   ANSWER: We have a lot more pcps than ROIs, so it's better to get 
            #           rid of the pcps rather than excluding NaN regions. 
            
            
            # Only keep participants that have finite values. 
            if np.all(np.isfinite(corr)): 
                
            
                
                all_final_id.append(group["ID"][i]) # Storing the IDs.
                
                all_corr.append(corr) # Storing the matrices. 
                
                
                # Average the upper triangular of the matrix - excluding the 
                # diagonal - and storing it for every participant.
                up_triu = corr[np.triu_indices(corr.shape[0], k = 1)]
                mn_uptriu = np.mean(up_triu)
                all_mn_fc.append(mn_uptriu)
    
        all_corr = np.stack(all_corr, axis = 0) # Put all the matrices in one 3D array. 
                                                # participant x edge row x edge column.
        
    
        # ____________ MEAN FC REGRESSION ____________
    
        # If specified, regress the mean FC of each pcps from the same edge of each pcp.
        # (see notes).
        if this_group == "group_mreg" or this_group == "group_pred": 
            mreg_resid = np.zeros((all_corr.shape[0],all_corr.shape[1],all_corr.shape[2]))
            
            
            # Loop through the rows of the correlation matrix...
            for j in range(all_corr.shape[1] - 1): # Starts at row 0 etc.
                       
                # ... and columns excluding the diagonal.
                for k in range(j + 1, all_corr.shape[2]):  # Starts at col 1.
                        
                    if this_group == "group_mreg":
                        
                        #  Compute the regression.
                        # (1 edge x all participants VS mn FC x all participants).
                        X = sm.add_constant(all_mn_fc) # Adding an intercept. 
                        mreg = sm.OLS(all_corr[:,j,k], X).fit() # Linear reg. 
                       
                        mreg_resid[:,j,k] = mreg.resid   # Store the residuals.
                                           
                        slopes[j,k] = mreg.params[1]     # Store the slopes. 
                        intercepts[j,k] = mreg.params[0] # Store the intercepts. 
                       
                       
                    elif this_group == "group_pred": # If "group_pred"...
                            
                        #... the linear regression is conducted using the slopes
                        # and intercepts of the "mreg" (presumably training) group.
                        
                        # Initialize a variable to store the predicted residuals. 
                        pred_resid = np.zeros((all_corr.shape[0],all_corr.shape[1],
                                                all_corr.shape[2]))
                        
                        # Using the "y = b0 + b1x + e" to get the residuals. 
                        mreg_resid[:,j,k] = (all_corr[:,j,k] - (slopes[j,k] * 
                                              np.array(all_mn_fc)) - intercepts[j,k])
                        
                        # NOTE: used "np.array" to avoid "can't multiply sequence 
                        # by non-int of type 'numpy.float64'" error (see Resources
                        # section at the end of the script). 
                      
                        
                        
            
             
            #  We use the mean matrix of the training group.
            if this_group == "group_mreg": 
                mean_corr = np.mean(all_corr, axis = 0)
                
            
            
            # Reinitialize all_corr so that all the matrices are 0s, and we can 
            # overwrite the upper triu and then sum the matrice and its transpose 
            # to fill in the lower triu. 
            
            all_corr = np.zeros((mreg_resid.shape[0],mreg_resid.shape[1],mreg_resid.shape[2]))
            
            
            for i in range(all_corr.shape[0]): # Loop along the number of participants. 
            # for i in range(all_corr.shape[0]): # Loop along the number of participants. 
                all_corr[i,:,:] = mreg_resid[i,:,:] + mreg_resid[i,:,:].T # Make it symmetrical.
            
                all_corr[i,:,:] = all_corr[i,:,:] + mean_corr
                
        
        #print(all_corr.shape)
            
        all_corr = np.split(all_corr, all_corr.shape[0], axis= 0)    
        
    
        # ____________ THRESHOLD & VECTORISED TRIU ____________
        
        for i in range(len(all_corr)):
            all_corr[i] = np.squeeze(all_corr[i])
            
            # plt.imshow(all_corr[i])
            # plt.show()
            
            #print(len(all_corr[0]))
            # Threshold the connectivity matrix.                                                                       
            #print("threshold")
            
            # VECTORISED TRIANGULAR: 
            # Should k = 0 or 1 ???????????????????????????????????????????????
            upp_triu = all_corr[i][np.triu_indices(all_corr[i].shape[0], k = 1)]
            vec_triu.append(upp_triu)
            
            # bct.threshold_proportional(all_corr[i], prop_threshold, copy = False)
            all_corr[i] = mst_threshold(W = all_corr[i], dens = prop_threshold)[0]
            
            
            # tree, clus  = backbone_wu2(all_corr[i], prop_threshold)
            # # plt.show()
            
            # all_corr[i] = clus + clus.T
            
            
            
            # comps,comp_sizes = bct.get_components(all_corr[i])
            # if (comps!=1).all(): print("more than one component !!!!!!!!!!")
            
            # plt.imshow(all_corr[i])
            # plt.show()
            
            all_corr[i][all_corr[i] > 0] = 1
            
            # plt.imshow(all_corr[i])
            # plt.show()
            
            # comps,comp_sizes = bct.get_components(all_corr[i])
            # if (comps==1).all() == False : print("more than one component !!!!!!!!!!")
            
            
            # if (comps==1).all(): print("only one component :)")
                
            # ____________ NETWORK MEASURES ____________
            
            # CLUSTERING COEFFICIENT:
            # Taking the average because it's a local measure.                            
            cluster.append(np.mean(bct.clustering_coef_bu(all_corr[i])))
                        
            # CHARACTERISTIC PATH LENGTH:
            # No need to average it since it's a global measure. 
            # plt.imshow(bct.distance_bin(all_corr[i]))
            # plt.show()
            charpath.append(bct.charpath(bct.distance_bin(all_corr[i]))[0])
            # plt.imshow(all_corr[i])
            # plt.show()
            
            
            # print(all_corr[i])
            # print(bct.charpath(all_corr[i])[0])
            # plt.imshow(all_corr[i])
            # plt.show()
            
            # ASSORTATIVITY:
            # No need to average it since it's a global measure. 
            assort.append(bct.assortativity_bin(all_corr[i]))
                    
        print("measures have been calculated")
    
        # ____________ DATAFRAME TO RETURN ____________
        
        # Putting the measures in a data frame. 
        dictionary[this_group] = pd.DataFrame({"ID" : all_final_id,
                                                "mean FC" : all_mn_fc, 
                                                "vectorised matrix" : vec_triu, 
                                                "cluster" : cluster, 
                                                "charpath" : charpath, 
                                                "assort" : assort})
        
        
        
    # ____________ FC OUTLIERS EXCLUSION ____________
    
    # Excluding mean FC and mean Framewise displacement (FD) outliers. 
    # This step only occurs if "group_pred" is NOT specified and IF 
    # "exclusion" = True:
    if exclusion:
        if len(group_pred) == 0:
                
            
            # Number of participants left in the group after removing non-finite 
            # values and ts that only contained zeroes. 
            n_all_pcp = len(dictionary["group_nr"])
            print(f"Number of participants after non finite exclusion: {n_all_pcp}")
            
            
            # Calculating the median mFC and the Median Absolute Deviation (MAD).
            median_fc = np.median(dictionary["group_nr"]["mean FC"])
            mad_fc = stats.median_abs_deviation(dictionary["group_nr"]["mean FC"])
            
            # Getting rid of FC outliers (if the value is more or less than median ± 
            # the median absolute deviation).
            for i in range(len(dictionary["group_nr"]["mean FC"])):
                    
                if dictionary["group_nr"]["mean FC"][i] < median_fc - 3*mad_fc or dictionary["group_nr"]["mean FC"][i] > median_fc + 3*mad_fc:
                    dictionary["group_nr"].drop(labels= i, axis=0, inplace=True)
                    dictionary["group_gsr"].drop(labels= i, axis=0, inplace=True)
                    dictionary["group_mreg"].drop(labels= i, axis=0, inplace=True)
            
            
            # Number of mFC outliers excluded. 
            n_after_exc_mfc = len(dictionary["group_nr"])
            print(f"Number of mFC outliers excluded: {n_all_pcp - n_after_exc_mfc}")
            
            # The index has been modified (by droping the participants), 
            # so I put it back to a list of consecutive numbers to loop through them 
            # more easily. 
            dictionary["group_nr"].index = np.arange(0, len(dictionary["group_nr"]))
            dictionary["group_gsr"].index = np.arange(0, len(dictionary["group_nr"]))
            dictionary["group_mreg"].index = np.arange(0, len(dictionary["group_nr"]))
            
            
            
            # ____________ FD OUTLIERS EXCLUSION ____________
            
            # Repeting the same steps but excluding participants with a mFD 
            # exceeding 0.5.
            for i in range(len(dictionary["group_nr"]["mean FC"])):
                if dictionary["group_nr"]["ID"][i] in phenotypic.loc[phenotypic['func_mean_fd'] > fd_lim, 'SUB_ID'].tolist(): 
                    dictionary["group_nr"].drop(labels= i, axis=0, inplace=True)
                    dictionary["group_gsr"].drop(labels= i, axis=0, inplace=True)
                    dictionary["group_mreg"].drop(labels= i, axis=0, inplace=True)
            
            
            # Number of mFD outliers excluded. 
            n_after_exc_mfd = len(dictionary["group_nr"])
            print(f"Number of mFD outliers excluded: {n_after_exc_mfc - n_after_exc_mfd}")
                
            # Setting all the indeces back to being a list of consecutive numbers. 
            dictionary["group_nr"].index = np.arange(0, len(dictionary["group_nr"]))
            dictionary["group_gsr"].index = np.arange(0, len(dictionary["group_gsr"]))
            dictionary["group_mreg"].index = np.arange(0, len(dictionary["group_mreg"]))
            
    
    return(dictionary)
    
