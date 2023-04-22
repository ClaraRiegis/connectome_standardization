#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: k20045146


"""

import matplotlib.pyplot as plt 
import numpy as np



# CODE ADAPTED FROM : Alex Belengeanu
# https://medium.com/mlearning-ai/getting-started-with-raincloud-plots-in-python-2ea5c2d01c11

# I just made a some modifications & turned it into a function. 

def raincloud(data_x, feature1, feature2, x_label, title, colors, ax, size):
    
    # fig, ax = plt.subplots(figsize=(8, 4))
    
    if colors == "on" :
        # Create a list of colors for the boxplots based on the number of features you have
        boxplots_colors = ['cornflowerblue', 'lightcoral']
        
        violin_colors = ['cornflowerblue', 'lightcoral']
    
        # Create a list of colors for the scatter plots based on the number of features you have
        scatter_colors = ['cornflowerblue', 'lightcoral']
    
    
        
    else: 
        
        # Create a list of colors for the boxplots based on the number of features you have
        boxplots_colors = ['dimgrey', 'darkgrey']
        
        violin_colors = ['dimgrey', 'darkgrey']
    
        # Create a list of colors for the scatter plots based on the number of features you have
        scatter_colors = ['dimgrey', 'darkgrey']
    
    
    # Boxplot data
    bp = ax.boxplot(data_x, patch_artist = True, vert = False)
        
    
    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    
    
    for median, color in zip(bp['medians'], violin_colors) :
        median.set_color(color)
    
    # Create a list of colors for the violin plots based on the number of features you have
    
    
    # Violinplot data
    vp = ax.violinplot(data_x, points=500, 
                   showmeans=False, showextrema=False, showmedians=False, vert=False)
    
    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
        # Change to the desired color
        b.set_color(violin_colors[idx])
    
    
    # Scatterplot data
    for idx, features in enumerate(data_x):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax.scatter(features, y, s=.3, c=scatter_colors[idx])
    
    plt.sca(ax)
    plt.yticks(np.arange(1,3,1), [feature1, feature2])  # Set text labels.
    plt.xlabel(x_label, fontsize = size)
    ax.tick_params(axis = "both" , labelsize=size)         # x label. 
    plt.title(title, fontdict={'fontsize': size})
  