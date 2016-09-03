# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:36:10 2016

@author: dillon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:08:18 2016

@author: dillon
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcdefaults
from matplotlib import colors
from matplotlib import cm

from numba import jit

rcdefaults()

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)


#%%
# we read in the two csv files, noting that each row is labeled in column 0 of the file
df1 = pd.read_csv("/home/dillon/Dropbox/Dillon/Observations/GreeneAndHoSample.csv",index_col=0)
df2 = pd.read_csv("/home/dillon/Dropbox/Dillon/Observations/ReverreberationMappedSample.csv",index_col=0)

# we add a new column to df2 that duplicates an existing column but has a name that matches with df1
df2['logMBH'] = df2['log M_BH']


# we make a new dataframe consisting of only the columns we care about from df1
finaldf = df1[['BPTxaxis','BPTyaxis','logMBH']]
# then we add the information about those variables from df2 as rows below
finaldf = finaldf.append(df2[['BPTxaxis','BPTyaxis','logMBH']])


#%%

# for partitioning, better to use either pd.qcut
# or use selection as below


def partition(data,massheading):
    datalist=[]
    minmass=min(data[massheading])
    maxmass=max(data[massheading])
    stepsize=(maxmass-minmass)/5
    mass=minmass
    while mass<maxmass:
        zaxis=data[massheading]
        parsedlist=data[(zaxis>mass) & (zaxis<mass+stepsize)]
        datalist.append(parsedlist)
        mass=mass+stepsize
    return datalist
#%%
def alt_partition(data, heading, bins=6):
    # takes the data and splits up using even sized intervals
    # for the data variable in question, and creates a new 'group' variable
    # with labels specifying the intervals that have precision of default 2
    data['groups'] = pd.cut(data[heading], bins=bins, precision=2)

def alt_partition_2(data, massheading, bins=6):
    '''
    splits using quantiles; bins specifies the number of groups to use
    '''
    data['groups'] = pd.qcut(data[massheading], q=bins, precision=2)

#%%

def BPTlineplot(data, label='', clr='k',xaxis='BPTxaxis',yaxis='BPTyaxis'):
    return plt.plot(data[xaxis],data[yaxis],color=clr,label=str(label),lw=2)

def BPTscatterplot(data, label='', clr='k',xaxis='BPTxaxis',yaxis='BPTyaxis'):
    plt.scatter(data[xaxis],data[yaxis],color=clr,label=str(label),lw=1)
    

#for bpt plots filtered by mass label
def observed_data_BPT_scatter(file_path, mass_label):
    data=pd.read_csv(file_path,index_col=0)
    sorted_data=partition(data, mass_label)
    norm = colors.Normalize(vmin=np.log(min(np.array(data[mass_label]))),
    vmax=np.log(max(np.array(data[mass_label]))))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    for part in sorted_data:
        uppermass=max(np.array(part[mass_label]))
        lowermass=min(np.array(part[mass_label]))
        BPTscatterplot(part,clr=mapper.to_rgba(np.log(np.array(part[mass_label])
        [0])),label=str(uppermass)+'-'+str(lowermass))


def alt_observed_data_BPT_scatter(file_path, mass_label):
    data = finaldf.dropna() # we drop the rows with missing data for x,y, or mass
    alt_partition(data, mass_label, bins=7)
    norm = colors.Normalize(vmin=(data[mass_label].min()), vmax=(data[mass_label].max()))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.prism_r)
    for category in data['groups'].unique():
        subset = data[data['groups'] == category]
        BPTscatterplot(subset,clr=mapper.to_rgba((subset.iloc[0][mass_label])),label=category)


def observed_data_BPT_line(file_path, mass_label):
    data=pd.read_csv(file_path,index_col=0)
    sorted_data=partition(data, mass_label)
    norm = colors.Normalize(vmin=np.log(min(np.array(data[mass_label]))),
    vmax=np.log(max(np.array(data[mass_label]))))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    for part in sorted_data:
        uppermass=max(np.array(part[mass_label]))
        lowermass=min(np.array(part[mass_label]))
        BPTlineplot(part,clr=mapper.to_rgba(np.log(np.array(part[mass_label])
        [0])),label=str(uppermass)+'-'+str(lowermass))

def observed_data_BPT_scatter_mulitiple(file_paths, mass_labels):
    i=0
    arry=[]
    for file_path in file_paths:
        arry.append(pd.read_csv(file_path,index_col=0))
        arry[i]=partition(arry[i], mass_labels[i])
    print(len(arry))

def appended_param_list(linelistpath, modelparamspath):
    linedata=pd.read_csv(linelistpath, delimiter='\t')
    modelparams=pd.read_csv(modelparamspath)
    linedata['radius']=modelparams['RADIUS_1']
    linedata['column_density']=modelparams['COLUMN DENSITY_1']
    linedata=linedata.rename(columns={"H  1 4861.36A":"H_beta", "O  3 5007.00A":"OIII","H  1 6562.85A":"H_alpha", "N  2 6584.00A":"NII","NE 2 12.8101m":"NEII", "NE 5 14.3228m":"NEV"})
    return linedata
###read data file and get new, more workable dataframe with appended_param_list   

def get_radii(modelparamspath):
    modelparams=pd.read_csv(modelparamspath)
    return np.array((modelparams['RADIUS_1']).unique())
    
def hden_parser(appended_df, density):
    parsedlist=appended_df[appended_df['column_density']==density]
    return parsedlist

def radius_parser(appended_df, radius):
    parsedlist=appended_df[appended_df['radius']==radius]
    return parsedlist

def BPTlog10coordsNE(radius,temp,hden=20):
    path="/home/dillon/Dropbox/Dillon/Cloudy/"
    data=appended_param_list(path+str(temp)+".dat","/home/dillon/Dropbox/Dillon/Cloudy/model_parameters.csv")
    data=hden_parser(data,hden)
    data=radius_parser(data,radius)
    return np.log10(temp), np.log10(np.float(data['NEV']/data['NEII']))
    
def NE_ratio_plot(radius_list,temp_list):
    norm = colors.Normalize(vmin=np.log(min(radius_list)), vmax=np.log(max(radius_list)))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)  
    xpoints=[]
    ypoints=[]
    for radius in radius_list:
        for temp in temp_list:
            xpoints.append(BPTlog10coordsNE(radius,temp)[0])
            ypoints.append(BPTlog10coordsNE(radius,temp)[1])
        plt.scatter(xpoints, ypoints,color=mapper.to_rgba(np.log(radius)),label=str(radius))
        xpoints=[]  
        ypoints=[]  
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0,title="Inner Radius")    
    plt.xlabel("log(T)")
    plt.ylabel("log(NE V 14.3228m/NE II 12.8101m)")
    plt.tight_layout()
    
def BPT_show(title,legend_title):
    xkewl = np.arange(-10.0,0.25,0.0001)
    ykewl = (0.61/(xkewl-0.47)) + 1.19
    # sets up Kaufmann line
    xkauf = np.arange(-10.0,-0.15,0.0001)
    ykauf = (0.61/(xkauf - 0.05)) +1.3
    # plots Kewly and Kaufmann lines
    plot2 = plt.plot(xkewl,ykewl, 'b')
    plot3 = plt.plot(xkauf,ykauf, 'm', linestyle='--')
    plt.xlabel('log([NII]/[H$\\alpha$])')
    plt.ylabel('log([OIII]/[H$\\beta$])')
    plt.xlim(-2,1)
    plt.ylim(-1.5,1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0,title=legend_title)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    

temps =np.array([100000,300000,500000,700000,1000000,3000000,5000000])
radii=get_radii("/home/dillon/Dropbox/Dillon/Cloudy/model_parameters.csv")     
                    
alt_observed_data_BPT_scatter("/home/dillon/Dropbox/Dillon/Observations/GreeneAndHoSample.csv",'logMBH')


xkewl = np.arange(-10.0,0.25,0.0001)
ykewl = (0.61/(xkewl-0.47)) + 1.19
# sets up Kaufmann line
xkauf = np.arange(-10.0,-0.15,0.0001)
ykauf = (0.61/(xkauf - 0.05)) +1.3
# plots Kewly and Kaufmann lines
plot2 = plt.plot(xkewl,ykewl, 'b')
plot3 = plt.plot(xkauf,ykauf, 'm', linestyle='--')
plt.xlabel('log([NII]/[H$\\alpha$])')
plt.ylabel('log([OIII]/[H$\\beta$])')
plt.xlim(-2,1)
plt.ylim(-1.5,1.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show()

#%%

from scipy.constants import speed_of_light

c = speed_of_light*100

    
#transmission_curve=pd.read_csv("/home/dillon/Desktop/research/filter_transmission_curves/WISE_WISE.W1.dat",skiprows=7,delimiter=r"\s+",header=None)

#grid=read_in("/home/dillon/Desktop/research/Continuum/grid000000000_hybrid_50_percent_agn.con",skiprows=[5277])

def read_and_get_rows(grid_path):
    grid=pd.read_csv(grid_path,delimiter='\t', header=None, usecols=[0,6], engine='python', skipfooter=1)
    return grid

    
#grid=read_and_get_rows("/home/dillon/Desktop/research/CONTINUA/50_percent/grid000000000_hybrid_50_percent_agn.con")


def range_list_maker(grid_new, trans_x, trans_y):
    final_list = np.zeros(grid_new.size)
    range_checker=False
    for i in range(len(grid_new)):
        for j in range(trans_x.size-1):
            if trans_x[j] <= grid_new[i] <= trans_x[j+1]:
                final_list[i] = np.interp(grid_new[i],
(trans_x[j],trans_x[j+1]),(trans_y[j], trans_y[j+1]))
                range_checker=True
    return (range_checker, final_list)

range_list_maker = jit(range_list_maker)

def get_transmission_list(transmission_curve,grid):
    grid_new = np.array(grid[0])*10**(-4)  
    trans_x = np.array((transmission_curve[0]))*10**(-4)
    trans_y = np.array((transmission_curve[1]))
    range_checker, final_list = range_list_maker(grid_new, trans_x, trans_y)
    if range_checker==False:
        print("WARNING!!!!!! NO WAVELENGTHS ARE IN RANGE OF BANDPASS")
    else:
        print("At least one wavelength is in range of bandpass. Good to go.")       
    return final_list



#%%

filter_keys = ["W1","W2","W3","W4","60mu","100mu"]
deltas = np.array([-0.663468,-1.0425477,-5.5206831,-4.1102301,-30.50316,-33.2995647])

def get_trans_data(filter_keys):
    dict_of_datasets = {}
    for filter_key in filter_keys:
        if filter_key[0] == 'W':
            trans_path="/home/dillon/Desktop/research/filter_transmission_curves/WISE_WISE."+filter_key+".dat"
        else:
            trans_path="/home/dillon/Dropbox/Dillon/Filter Transmission Curves/IRAS_IRAS."+filter_key+".dat"
        dict_of_datasets[filter_key] = pd.read_csv(trans_path,delimiter=r"\s+",header=None)
    return dict_of_datasets

trans_curves = get_trans_data(filter_keys)

#%%

def alt_get_FW(filter_key, model_num, agn_percent,delta=0.663468):
    if model_num < 10:
        grid_path="/home/dillon/Desktop/research/CONTINUA/"+str(agn_percent)+"_percent/grid00000000"+str(model_num)+"_hybrid_"+str(agn_percent)+"_percent_agn.con"
    elif model_num >= 10 and model_num<100:
        grid_path="/home/dillon/Desktop/research/CONTINUA/"+str(agn_percent)+"_percent/grid0000000"+str(model_num)+"_hybrid_"+str(agn_percent)+"_percent_agn.con"
    else:
        grid_path="/home/dillon/Desktop/research/CONTINUA/"+str(agn_percent)+"_percent/grid000000"+str(model_num)+"_hybrid_"+str(agn_percent)+"_percent_agn.con"
    grid=read_and_get_rows(grid_path)  
    transmission_curve = trans_curves[filter_key]
    trans=get_transmission_list(transmission_curve,grid)
    wave=np.array(grid[0])
    lum=np.array(grid[6])/wave
    tot=np.zeros(len(wave))
    for k in range(len(wave)-1):
        tot[k]=(((wave[k+1]-wave[k])/2.0)*((lum[k]*trans[k])+(lum[k+1]*trans[k+1])))
    return np.sum(tot)/delta

alt_get_FW = jit(alt_get_FW)



#zero_percent_trans_list=np.zeros(len(filter_keys))
#trans_list=np.zeros(len(filter_keys))


ZW1= 8.1787*10**(-15)
ZW2=2.415*10**(-15)
ZW3=6.515*10**(-17)
#%%
#x,z_percent_trans_list=alt_get_FW("W1",548,0,W=True,delta=deltas[0],get_trans_list=True)
#y,trans_list=alt_get_FW("W1",0,10,W=True,delta=deltas[0],get_trans_list=True)
#%%


def stuff_i_want_to_do(num_rows=558):
    for percent in np.arange(0,100,10):
        print("!!!!", percent, "%%%")
        title = "alt_agn_"+str(percent)+"_percent.out"
    #    file_rows=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    #    if percent <= 10:
    #        get_trans=True
    #    print("get_trans is ", get_trans)
        FW_array = np.zeros(shape=(num_rows,len(filter_keys) + 7))
        for model_num in range(num_rows):
            print("MODEL_NUM IS", model_num,"★")
            FW_array[model_num,0] = percent
            FW_array[model_num,1] = model_num
            i=0
            for filter_key in filter_keys:
                print("filter key is: ",filter_key,".")
                #print(FW_array.shape)
                FW_array[model_num,i+2] = alt_get_FW(filter_key,model_num,percent,delta=deltas[i])
                i += 1
            FW_array[model_num,-5] = W1_W2 = 2.5 * np.log10((FW_array[model_num,3]/FW_array[model_num,2])*(ZW1/ZW2))
            FW_array[model_num,-4] = W2_W3 = 2.5 * np.log10((FW_array[model_num,4]/FW_array[model_num,2])*(ZW2/ZW3))
            print("W1-W2 is ", W1_W2)
            print("W2-W3 is ", W2_W3)
            FW_array[model_num,-3] = F60nu = (FW_array[model_num,6]*60**2/c)*10**(-4)
            FW_array[model_num,-2] = F100nu = (FW_array[model_num,7]*100**2/c)*10**(-4)
            FW_array[model_num,-1] = F60F100 = F60nu/F100nu
            print(FW_array[model_num])
            #format is %AGN model_num FW1 FW2 FW3 FW4 FW60 FW100 W1-W2 W2-W3 F60nu F100nu F60F100        
        np.savetxt(title,FW_array, delimiter=' ',fmt='%13.5e',header="%AGN model_num FW1 FW2 FW3 FW4 FW60 FW100 W1-W2 W2-W3 F60nu F100nu F60/F100")

      
#stuff_i_want_to_do = jit(stuff_i_want_to_do)

stuff_i_want_to_do(558)


#%%
x=y=np.arange(10)
np.savetxt('test.out', (x,y), delimiter='\t', fmt="%d")
#%%
tbb_data=pd.read_csv("/home/dillon/Desktop/TBBB_data.csv")
tbb_data=tbb_data.rename(columns={"H  1 4861.36A/Hbeta":"H_beta", "O  3 5007.00A/Hbeta":"OIII","H  1 6562.85A/Hbeta":"H_alpha", "N  2 6584.00A/Hbeta":"NII","NE 2 12.8101m/Hbeta":"NEII", "NE 5 14.3228m/Hbeta":"NEV"})
print(tbb_data)

def BPT_ratios_2_level_vary(dataframe,const_val=2.5,const_key='n',outer_vary_key='T',outer_ele_units=' K',inner_vary_key='U',color_log_scale=True):
    dataframe['NII/H_alpha']=dataframe['NII']/dataframe['H_alpha']
    parsed_df=dataframe[dataframe[const_key]==const_val]
    outer_vary_list=np.unique(dataframe[outer_vary_key])
    inner_vary_list=np.unique(dataframe[inner_vary_key])
    if color_log_scale==True:
        norm = colors.Normalize(vmin=np.log(min(outer_vary_list)), vmax=np.log(max(outer_vary_list)))
    else:
        norm = colors.Normalize(vmin=min(outer_vary_list), vmax=max(outer_vary_list))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    for outer_ele in outer_vary_list:
        switch=True
        for inner_ele in inner_vary_list:
            parsed_by_both=parsed_df[(parsed_df[outer_vary_key]==outer_ele) & (parsed_df[inner_vary_key]==inner_ele)]
            parsed_by_both['logNII']=np.log10(parsed_by_both['NII'])
            parsed_by_both['logOIII']=np.log10(parsed_by_both['OIII'])
            if color_log_scale==True:
                if switch==True:
                    BPTscatterplot(parsed_by_both,xaxis='logNII',yaxis='logOIII',clr=mapper.to_rgba(np.log(outer_ele)),label=str(outer_ele)+outer_ele_units)
                    switch=False
                else:
                    BPTscatterplot(parsed_by_both,xaxis='logNII',yaxis='logOIII',clr=mapper.to_rgba(np.log(outer_ele))) 
            else:
                if switch==True:
                    BPTscatterplot(parsed_by_both,xaxis='logNII',yaxis='logOIII',clr=mapper.to_rgba(outer_ele),label=str(outer_ele)+outer_ele_units)
                    switch=False
                else:
                    BPTscatterplot(parsed_by_both,xaxis='logNII',yaxis='logOIII',clr=mapper.to_rgba(outer_ele)) 
                    
    
def alt_NE_ratio_plot(ion_param_list,temp_list, dataframe,n=1.5):
    norm = colors.Normalize(vmin=min(ion_param_list), vmax=max(ion_param_list))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)  
    for ion_param in ion_param_list:
        xpoints=np.array(np.log10(temp_list))
        parseddf=dataframe[(dataframe['U']==ion_param) & (dataframe['n']==n)]
        ratio_points=parseddf['NEV']/parseddf['NEII']
        ypoints=np.array(ratio_points)
        plt.plot(xpoints, ypoints,color=mapper.to_rgba(ion_param))
        plt.scatter(xpoints, ypoints,color=mapper.to_rgba(ion_param),label=ion_param)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0,title="Ionization Parameter")    
    plt.xlabel("log(T)")
    plt.ylabel("log(NE V 14.3228m/NE II 12.8101m)")
    plt.tight_layout()
    plt.xlim([5,7])
    plt.ylim([0,20])
    plt.show()


def NE6_ratio_plot(ion_param_list,temp_list, dataframe,n=1.5):
    dataframe['NEVI']=10**(dataframe['Ne 6 flux']-dataframe['Hbeta flux'])
    norm = colors.Normalize(vmin=min(ion_param_list), vmax=max(ion_param_list))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)  
    for ion_param in ion_param_list:
        xpoints=np.array(np.log10(temp_list))
        parseddf=dataframe[(dataframe['U']==ion_param) & (dataframe['n']==n)]
        ratio_points=np.log10(parseddf['NEVI']/parseddf['NEV'])
        ypoints=np.array(ratio_points)
        plt.plot(xpoints, ypoints,color=mapper.to_rgba(ion_param))
        plt.scatter(xpoints, ypoints,color=mapper.to_rgba(ion_param),label=ion_param)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0,title="Ionization Parameter")    
    plt.xlabel("log(T)")
    plt.ylabel("log(NE VI/NE V)")
    plt.tight_layout()
    plt.xlim([5,7])
    plt.ylim([-4,.5])
    plt.show()

def NE6_to_x_ratio_plot(ion_param_list,temp_list,denom_key, dataframe,n=1.5):
    dataframe['NEVI']=10**(dataframe['Ne 6 flux']-dataframe['Hbeta flux'])
    dataframe['H_beta']=10**(dataframe['Hbeta flux'])
    norm = colors.Normalize(vmin=min(ion_param_list), vmax=max(ion_param_list))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)  
    for ion_param in ion_param_list:
        xpoints=np.array(np.log10(temp_list))
        parseddf=dataframe[(dataframe['U']==ion_param) & (dataframe['n']==n)]
        ratio_points=np.log10(parseddf['NEVI']/parseddf[denom_key])
        ypoints=np.array(ratio_points)
        plt.plot(xpoints, ypoints,color=mapper.to_rgba(ion_param))
        plt.scatter(xpoints, ypoints,color=mapper.to_rgba(ion_param),label=ion_param)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0,title="Ionization Parameter")    
    plt.xlabel("log(T)")
    plt.ylabel("log(NE VI/"+str("H beta")+")")
    plt.tight_layout()
    plt.show()


NE6_to_x_ratio_plot(np.unique(tbb_data['U']),np.unique(tbb_data['T']),"H_beta",tbb_data)

alt_NE_ratio_plot(np.unique(tbb_data['U']),np.unique(tbb_data['T']),tbb_data)

NE6_ratio_plot(np.unique(tbb_data['U']),np.unique(tbb_data['T']),tbb_data)

                
BPT_ratios_2_level_vary(tbb_data)
BPT_show("n=2.5","Temperature")

BPT_ratios_2_level_vary(tbb_data,const_val=1.5)
BPT_show("n=1.5","Temperature")

BPT_ratios_2_level_vary(tbb_data,const_val=100000,const_key='T',outer_vary_key='U',outer_ele_units='',inner_vary_key='n',color_log_scale=False)
BPT_show("T=100000","Ionization parameter") 

BPT_ratios_2_level_vary(tbb_data,const_val=5000000,const_key='T',outer_vary_key='U',outer_ele_units='',inner_vary_key='n',color_log_scale=False)
BPT_show("T=5000000","Ionization parameter") 

   