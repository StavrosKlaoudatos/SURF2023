import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import colorsys



path = '/Users/stavrosklaoudatos/Desktop/HiggsData/Code'

dict ={}
    
dirs = os.listdir(path)
dirs = sorted(dirs)




files = []

for dir in dirs:
    if '.parquet' in dir:
        files.append(dir)

for f in files:
    name = f.split('_')
    name = name[1] + '_' +name[2]
    if 'MX1000' in name:
        
        

        df = pd.read_parquet(f, engine='pyarrow')

        dict.update({name:df})
        print(name, '\n')

# Create a new figure
plt.figure(figsize=(10, 6))

# Define the colors for the histograms
num_plots = len(dict)
mode= 0
modes=[]
p= [0]
for i, (name, df) in enumerate(dict.items()):
    
    hue = i / num_plots
    color = colorsys.hsv_to_rgb(hue, 1, 1)
    # Plot histogram for each DataFrame

    x = df['LeadPhoton_pt']
    

    
    
    y,b,m = plt.hist(x, bins=200, alpha=0.5, linewidth=2, color=color, label=name,histtype='step', range = (-3000,3000))


    y=y.max()
    mode = x.mode()[0]
    modes.append(round(mode,-3))

    if mode not in modes[:-2]:
        plt.axvline(mode, color='k', linestyle='dashed', linewidth=2)
        plt.text(200, 7*(1+i/2), 'Mode: {:.2f}'.format(mode))

        ticks = plt.xticks()
        print(ticks[0])
        plt.xticks(list(ticks[0])+modes)

 
   # if 'MY125'in name:
        


mode = mode/num_plots
# Set plot title and labels



plt.title('Lead Photon Tranverse Momentum $p_T$ in MX1000', loc='left')
plt.title('2017 (13 TeV)', loc='right')
plt.xlabel('Lead Photon Tranverse Momentum $p_T$ [GeV]')
plt.ylabel('Events')


# Add a legend
plt.legend()
plt.yscale('log')
plt.rc('text', usetex=True)

# Show the plot
plt.show()





