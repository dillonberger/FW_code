# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 17:42:15 2016

@author: dillonberger
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcdefaults
from matplotlib import colors
from matplotlib import cm

from numba import jit
from scipy.constants import speed_of_light

c = speed_of_light*100


rcdefaults()

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

filter_keys = ["W1","W2","W3","W4","60mu","100mu"]
deltas = np.array([-0.663468,-1.0425477,-5.5206831,-4.1102301,-30.50316,-33.2995647])

def read_and_get_rows(grid_path):
    grid=pd.read_csv(grid_path,delimiter='\t', header=None, usecols=[0,6], engine='python', skipfooter=1)
    return grid
    
def get_trans_data(filter_keys):
    dict_of_datasets = {}
    for filter_key in filter_keys:
        if filter_key[0] == 'W':
            trans_path="/Users/dillonberger/Documents/filter_transmission_curves/WISE_WISE."+filter_key+".dat"
        else:
            trans_path="/Users/dillonberger/Documents/filter_transmission_curves/IRAS_IRAS."+filter_key+".dat"
        dict_of_datasets[filter_key] = pd.read_csv(trans_path,delimiter=r"\s+",header=None)
    return dict_of_datasets


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

trans_curves = get_trans_data(filter_keys)

def alt_get_FW(filter_key, model_num, agn_percent,delta=0.663468):
    if model_num < 10:
        grid_path="/Users/dillonberger/GitHub/low_z_models/CONTINUA/"+str(agn_percent)+"_percent/grid00000000"+str(model_num)+"_hybrid_"+str(agn_percent)+"_percent_agn.con"
    elif model_num >= 10 and model_num<100:
        grid_path="/Users/dillonberger/GitHub/low_z_models/CONTINUA/"+str(agn_percent)+"_percent/grid0000000"+str(model_num)+"_hybrid_"+str(agn_percent)+"_percent_agn.con"
    else:
        grid_path="/Users/dillonberger/GitHub/low_z_models/CONTINUA/"+str(agn_percent)+"_percent/grid000000"+str(model_num)+"_hybrid_"+str(agn_percent)+"_percent_agn.con"
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

def main_write_out_files(num_rows=558):
    ZW1= 8.1787*10**(-15)
    ZW2=2.415*10**(-15)
    ZW3=6.515*10**(-17)
    for percent in range(0,110,10):
        print("!!!!", percent, "%%%")
        title = "alt_agn_"+str(percent)+"_percent.out"
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



main_write_out_files()