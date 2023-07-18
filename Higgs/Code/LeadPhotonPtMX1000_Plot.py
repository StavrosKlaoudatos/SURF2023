import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import colorsys
from statistics import mean



path = '/Users/stavrosklaoudatos/Desktop/HiggsData/Code'

Dict ={}
    
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

        Dict.update({name:df})
        print(name, '\n')

# Create a new figure
plt.figure(figsize=(10, 6))

# Define the colors for the histograms
num_plots = len(Dict)
mode= 0
modes =[]
modes=[]
names = list(Dict.keys())
p= [0]
for i, (name, df) in enumerate(Dict.items()):
    
    hue = i / num_plots
    color = colorsys.hsv_to_rgb(hue, 1, 1)
    # Plot histogram for each DataFrame

    x = np.sqrt((df['LeadPhoton_eta'] - df['SubleadPhoton_eta'])**2 + (df['LeadPhoton_phi'] - df['SubleadPhoton_phi'])**2)


    
    
    y,b,m = plt.hist(x, bins=50, alpha=0.5, linewidth=2, color=color, label=name,histtype='step', range = (0,10))





    print(y.argmax())
    mode = b[y.argmax()]
    modes.append(round(mode,3))

    plt.axvline(mode, color='k', linestyle='dashed', linewidth=2)

    
        
print(names)

mean_mode = mean(modes)



ticks = list(plt.xticks()[0])
try:
    ticks.remove(400)
except:
    pass

plt.xticks(ticks+[mean_mode])
plt.axvline(mean_mode, color='red', linestyle='dashed', linewidth=3)

modes_str='Modes \n'
for i in range(len(modes)):
    string = str(names[i].split('_')[1]) + ': ' + str(modes[i]) + '\n'
    modes_str+= string




plt.text(-170,2500, modes_str, bbox=dict(facecolor='blue', alpha=0.5))




plt.title('Lead Photon Transverse Momentum $p_T$ in MX1000', loc='left')
plt.title('2017 (13 TeV)', loc='right')
plt.xlabel('Lead Photon Transverse Momentum $p_T$ [GeV]')
plt.ylabel('Events')


# Add a legend
plt.legend()



plt.rc('text', usetex=True)
#plt.yscale('log')

# Show the plot
plt.show()





