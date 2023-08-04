import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import colorsys
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)




def filter_df(x,a,b):
    # Perform the calculation
    
    # Filter the DataFrame to only include rows where x >= 0.1
    mask = x < b 

    # Apply the mask to the DataFrame
    df_filtered = x.loc[mask]

    mask2 = x > a

    df_filtered = df_filtered.loc[mask2]

    return df_filtered




path = '/Users/stavrosklaoudatos/Desktop/HiggsData/Code'

dict ={}
    
dirs = os.listdir(path)
dirs = sorted(dirs)

per = (0, 1)
p = [
    "fathbbjet_tau1", "fathbbjet_tau2","fathbbjet_tau3"
  ]






ranges= [per,per]
titles = ["Tau 2 Over Tau 1 Ratio for Subjetiness","Tau 3 Over Tau 2 Ratio for Subjetiness"]
xaxis = ["Tau2/Tau1","Tau3/Tau2"]
location = ['T2T1','T3T2']
temp_vals =[]
vals =[[p[1],p[0]],[p[2],p[1]]]





MX = 'MX2000'


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


for k in range(len(titles)):
    modes = []
    plt.figure(figsize=(15, 6))
    for i, (name, df) in enumerate(dict.items()):
        
        hue = i / num_plots
        color = colorsys.hsv_to_rgb(hue, 1, 1)
        # Plot histogram for each DataFrame

        
        
        
        x = df[vals[k][0]]/ df[vals[k][1]] 
        
    
        
        
        x= filter_df(x)


        y, bin_edges = np.histogram(x, bins=500, range=ranges[k])
        max_y_idx = np.argmax(y)  # Index of the maximum y value
        max_x_val = (bin_edges[max_y_idx] + bin_edges[max_y_idx + 1]) / 2  # Center of the bin

        
        y,b,m = plt.hist(x, bins=500, alpha=0.5, linewidth=1,color=color, label=name,histtype='step', range = ranges[k])





        
        modes.append(round(max_x_val, -3))
       
        
        if max_x_val not in modes[:-2]:
            plt.axvline(max_x_val, color='k', linestyle='dashed', linewidth=0.5)
            plt.text(max_x_val, 10**(0.5+i/4), 'Max freq: {:.2f}'.format(max_x_val))

            ticks = plt.xticks()
            plt.xticks(list(ticks[0]) + modes)


    
    # if 'MY125'in name:
            


    #mode = mode/num_plots
    # Set plot title and labels


    
    plt.title(titles[k]+ ' in ' + MX, loc='left')
    plt.title('2017 (13 TeV)', loc='right')
    plt.xlabel(xaxis[k])
    plt.ylabel('Events')


    # Add a legend
    plt.legend()
  
   # plt.rc('text', usetex=True)

    # Show the plot
    plt.yscale('log')
    plt.savefig('/Users/stavrosklaoudatos/Desktop/HiggsData/Plots/TauPlots/' + MX +'/'+location[k]+MX+'img.png')
    plt.show()
    plt.clf()




