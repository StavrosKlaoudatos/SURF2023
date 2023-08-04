import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import colorsys
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


path_to_parquets = '/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/data/Parquets'


 

p = [[['LeadPhoton_eta', 'SubleadPhoton_eta', 'LeadPhoton_phi', 'SubleadPhoton_phi']], [['Diphoton_eta', 'fathbbjet_eta', 'Diphoton_phi', 'fathbbjet_phi']], [['LeadPhoton_eta', 'fathbbjet_eta', 'LeadPhoton_phi', 'fathbbjet_phi']], [['SubleadPhoton_eta', 'fathbbjet_eta', 'SubleadPhoton_phi', 'fathbbjet_phi']]]

ranges= [(0, 8), (0, 8), (0, 8), (0, 8)]

titles = ['Angular Distance between two leading Photons [rad]', 'Angular Distance between the Diphoton System and the Fat Hbb Jet [rad]', 'Angular Distance between the Lead Photon and the Fat Hbb Jet [rad]', 'Angular Distance between the Sublead Photon and the Fat Hbb Jet [rad]']

xaxis = ['Angular Distnace [rad]', 'Angular Distance [rad]', 'Angular Distance [rad]', 'Angular Distance [rad]']

location = ['Leading_Photons_dR', 'Diphoton_fathbbjet_dR', 'Lead_fathbbjet_dR', 'Sublead_fathbbjet_dRs']





def GeneratePlots(quants, ranges, titles, xaxis, location,bins=300,MX = ['MX1000','MX2000'],path='/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/dR_Distributions/'):

    def filter_df(x):
    

        # Filter the DataFrame to only include rows where x >= 0.1
        mask = x > 0.7

        # Apply the mask to the DataFrame
        df_filtered = x.loc[mask]

        mask2 = x < 30

        df_filtered = df_filtered.loc[mask2]

        return df_filtered

    def CreateFolders(path,MX=MX):
        try:
            for mx in MX:
                directory = path + '/' +mx
                os.mkdir(directory)
        except:
            print("Directory: {} already exists \n ======================================".format(directory))
            pass

    def AddModeLine(mode, y):
        # Add vertical line
        plt.axvline(mode, color='k', linestyle='dashed', linewidth=0.5)
        # Add text
        plt.text(mode*1.1, y, 'Mode: {:.2f}'.format(mode))


    def CreateDataFrame(MX,path = path_to_parquets):




        d ={}
        files = [] 
        dirs = os.listdir(path)
        dirs = sorted(dirs)




        for dir in dirs:
            if '.parquet' in dir:
                files.append(dir)
        

        for f in files:
            name = f.split('_')
            name = name[1] + '_' +name[2]
            if MX in name:
                
                

                df = pd.read_parquet(path+'/'+f, engine='pyarrow')

                d.update({name:df})
                print(name, '\n')

        return d


    def GenerateMode(data,bins:int,range):
        y, bin_edges = np.histogram(data, bins=bins, range=range)
        max_y_idx = np.argmax(y)  # Index of the maximum y value
        mode = (bin_edges[max_y_idx] + bin_edges[max_y_idx + 1]) / 2  # Center of the bin

        return mode


    CreateFolders(path=path)

    

    

    for mx in MX:

        dict = CreateDataFrame(mx)
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

                
                
                #Setup Plotting
                x = np.sqrt((df[quants[k][0][0]]- df[quants[k][0][1]] )**2 + (df[quants[k][0][2]] - df[quants[k][0][3]] )**2)
                #========================================================    




            
                
                x= filter_df(x)
                y,b,m = plt.hist(x, bins=bins, alpha=0.5, linewidth=1,color=color, label=name,histtype='step', range = ranges[k])


                mode = GenerateMode(x,bins,ranges[k])
                modes.append(round(mode, -3))
                AddModeLine(mode, 10**(0.5+i/4))

            ticks = plt.xticks()
            plt.xticks(list(ticks[0]) + modes)

            
            plt.title(titles[k]+ ' in ' + mx, loc='left')
            plt.title('2017 (13 TeV)', loc='right')
            plt.xlabel(xaxis[k])
            plt.ylabel('Events')


            # Add a legend
            plt.legend(loc='best')
        
            # plt.rc('text', usetex=True)

            # Show the plot
            plt.yscale('log')
        
            plt.savefig(path + mx +'/'+location[k]+mx+'img.png')
            #plt.show()

            plt.clf()


GeneratePlots(p,ranges,titles,xaxis,location)

