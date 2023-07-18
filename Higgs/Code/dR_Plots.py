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
    mask = x >= 0.1

    # Apply the mask to the DataFrame
    df_filtered = x.loc[mask]

    return df_filtered




path = '/Users/stavrosklaoudatos/Desktop/HiggsData/Code'

dict ={}
    
dirs = os.listdir(path)
dirs = sorted(dirs)

per = (0, 2*np.pi)
p = {
    "Leading_Photons_dR":[per, 'Angular Distance between two leading Photons [rad]', 'Angular Distnace [rad]','LeadPhoton_eta','SubleadPhoton_eta','LeadPhoton_phi','SubleadPhoton_phi'],
    "Diphoton_fathbbjet_dR":[per, 'Angular Distance between the Diphoton System and the Fat Hbb Jet [rad]', 'Angular Distance','Diphoton_eta','fathbbjet_eta','Diphoton_phi','fathbbjet_phi'],
    "Lead_fathbbjet_dR":[per,'Angular Distance between the Lead Photon and the Fat Hbb Jet [rad]', 'Angular Distance','LeadPhoton_eta','fathbbjet_eta','LeadPhoton_phi','fathbbjet_phi'],
    "Sublead_fathbbjet_dRs":[per,'Angular Distance between the Sublead Photon and the Fat Hbb Jet [rad]', 'Angular Distance','SubleadPhoton_eta','fathbbjet_eta','SubleadPhoton_phi','fathbbjet_phi']
}
keys = list(p.keys())

values = list(p.values())



ranges= []
titles = []
xaxis = []
temp_vals =[]
vals =[]

for l in range(len(values)):

    ranges.append(values[l][0])
    titles.append(values[l][1])
    xaxis.append(values[l][2])
    temp_vals.append(values[l][3:])
    vals.append(temp_vals)
    temp_vals =[]



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

        
        
        
        x = np.sqrt((df[vals[k][0][0]]- df[vals[k][0][1]] )**2 + (df[vals[k][0][2]] - df[vals[k][0][3]] )**2)
        
    
        
        
        x= filter_df(x)
        
        y,b,m = plt.hist(x, bins=500, alpha=0.5, linewidth=1,color=color, label=name,histtype='step', range = ranges[k])





        y=y.max()
        mode = x.mode()[0]
        modes.append(round(mode,-3))
       
        
        #if mode not in modes[:-2]:
            #plt.axvline(mode, color='k', linestyle='dashed', linewidth=0.5)
            #plt.text(mode, 7*(1+i/2), 'Mode: {:.2f}'.format(mode))

            #ticks = plt.xticks()
            
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
  
   # plt.rc('text', usetex=True)

    # Show the plot
    #plt.yscale('log')
    plt.savefig('/Users/stavrosklaoudatos/Desktop/HiggsData/Plots/dR_Distributions/' + MX +'/'+keys[k]+MX+'img.png')
    
    plt.clf()




