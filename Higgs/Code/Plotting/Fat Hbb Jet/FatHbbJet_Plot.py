

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import colorsys
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
def filter_df(x):
    # Perform the calculation
    
    # Filter the DataFrame to only include rows where x >= 0.1
    mask = x >= 800

    # Apply the mask to the DataFrame
    df_filtered = x.loc[mask]

    return df_filtered


path = '/Users/stavrosklaoudatos/Desktop/HiggsData/Code'

dict ={}
per = (-2*np.pi, 2*np.pi)
dirs = os.listdir(path)
dirs = sorted(dirs)
p = {
    "fathbbjet_pt":[(0,2000), 'Fat H to bb Jet Transverse Momentum $p_T$ [GeV]', 'Transverse Momentum $p_T$ [GeV]'],
    "fathbbjet_eta":[per, 'Fat H to bb Jet Pseudorapidity $\eta$ [rad]', 'Rapidity $\eta$ [rad]'],
    "fathbbjet_phi":[per,'Fat H to bb Jet Azimuthal Angle $\phi$ [rad]', 'Azimuthal Angle $\phi$ [rad]'],
    "fathbbjet_mass":[(0,1000),'Fat H to bb Jet Invariant Mass $m$ [GeV]', 'Invariant Mass $m$ [GeV]'],
    "fathbbjet_deepTagMD_HbbvsQCD":[(-1,1),'Mass-decorrelated DeepBoostedJet tagger H to bb vs QCD discriminator', ''],
    "fathbbjet_msoftdrop":[(0,1000), 'Fat H to bb Jet Soft Drop Mass [GeV]', 'Soft Drop Mass [GeV]']
}

keys = list(p.keys())

values = list(p.values())

ranges= []
titles = []
xaxis = []

for l in range(len(values)):

    ranges.append(values[l][0])
    titles.append(values[l][1])
    xaxis.append(values[l][2])


MX = 'MX1000'


files = []

for dir in dirs:
    if '.parquet' in dir:
        files.append(dir)

for f in files:
    name = f.split('_')
    name = name[1] + '_' +name[2]
    if MX in name:
        
        

        df = pd.read_parquet(f, engine='pyarrow')

        dict.update({name:df})
        print(name, '\n')

# Create a new figure


# Define the colors for the histograms
num_plots = len(dict)
mode= 0
modes=[]


for k in range(len(p)):
    plt.figure(figsize=(15, 6))
    for i, (name, df) in enumerate(dict.items()):
        
        hue = i / num_plots
        color = colorsys.hsv_to_rgb(hue, 1, 1)
        # Plot histogram for each DataFrame

        x = df[keys[k]]
        
        x = filter(x)
        
        
        
        y,b,m = plt.hist(x, bins=500, alpha=0.5, linewidth=2, color=color, label=name,histtype='step', range = ranges[k])


        y=y.max()
        mode = x.mode()[0]
        modes.append(round(mode,-3))

        #if mode not in modes[:-2]:
            #plt.axvline(mode, color='k', linestyle='dashed', linewidth=2)
            #plt.text(200, 7*(1+i/2), 'Mode: {:.2f}'.format(mode))

           # ticks = plt.xticks()
           # print(ticks[0])
           # plt.xticks(list(ticks[0])+modes)

    
    # if 'MY125'in name:
            


    #mode = mode/num_plots
    # Set plot title and labels


    
    plt.title(titles[k]+ ' in ' + MX, loc='left')
    plt.title('2017 (13 TeV)', loc='right')
    plt.xlabel(xaxis[k])
    plt.ylabel('Events')


    # Add a legend
    plt.legend()
    plt.yscale('log')
   # plt.rc('text', usetex=True)

    # Show the plot
    plt.savefig('/Users/stavrosklaoudatos/Desktop/HiggsData/Plots/FatHbbJets/' + MX +'/'+keys[k]+MX+'img.png')
    plt.clf()




