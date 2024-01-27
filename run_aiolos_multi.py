#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:24:14 2023

@author: wmisener
"""
import time
import os
from subprocess import run
from multiprocessing import Pool
import numpy as np
from itertools import repeat

def run_aiolos(directory, parstring, T_int, R_inner, solarfactor):
    '''
    Modifies input parfile within input directory to have the input T_int, 
    R_inner, and solar factor, and saves this as a new parfile. Then, runs this 
    parfile, saving the printed output to a constructed logfile
    '''
    gamma = solarfactor/3.3e-2 #bc CONSTOPA_PLANCK_FACTOR=3.3e-2
    #the new parfile with the input values
    parfile = parstring+"_Tint{:.0f}_R{:.0f}e8_gamma{:.0e}.par".format(T_int, R_inner/1e8, gamma)

    # makes a new parfile and modifies T_int
    os.system("sed 's/PARI_TINT.*/PARI_TINT   {:.0f}'".format(T_int)+"'/' "+directory+parstring+".par"+" > "+directory+parfile)
    # modifies the just-created parfile in place
    os.system("sed -i '' 's/PARI_DOMAIN_MIN.*/PARI_DOMAIN_MIN   {:.2e}'".format(R_inner)+"'/' "+directory+parfile)
    os.system("sed -i '' 's/CONSTOPA_SOLAR_FACTOR.*/CONSTOPA_SOLAR_FACTOR   {:.2e}'".format(solarfactor)+"'/' "+directory+parfile)
    # construct a logfile for the printed outputs
    logfilename = 'logfile_' + parstring + '_T{:.0f}_R{:.0f}e8_gamma{:.0e}.log'.format(T_int, R_inner/1e8, gamma)
    logfile = open(logfilename, 'w')
    
    # prints start to screen
    print('Running T_int={:.0f}, R_inner = {:.2e}, gamma = {:.0e}'.format(T_int, R_inner, gamma))
    # run aiolos with new parfile, sending printed outputs to logfile
    run(['./aiolos', '-dir', directory, '-par', parfile], stdout=logfile)
    print('Finished T_int={:.0f}, R_inner = {:.2e}, gamma = {:.0e}'.format(T_int, R_inner, gamma))
    #os.system('./aiolos -dir WM_runs_git/ -par CPML_test_noUV_ions_chemH2_gamma3e1_inithydro_06opacityscaling_r24e8_rho1e-5_Tint100_M5_pythontest.par > test_log.log')

def main_new():
    '''
    Uses multiprocessing.Pool to run simultaneous multiprocessing of aiolos runs
    Can (so far) vary T_int, R_inner, and constopa_solar_factor
    '''
    time_start = time.time()

    #folder = '3ME_T1000_aiolos_testcases'
    #os.mkdir(folder)
    
    T_input = 150#np.array([50, 100]) 
    #R_input = np.linspace(2.5e9, 5e9, num=6)
    
    R_input = np.array([2.6e9, 2.7e9, 2.8e9, 2.9e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9])
    #solar_factor = np.array([0.0033, 0.01, 0.033, 0.1, 0.33, 1])
    solar_factor = np.array([0.33, 1])
    #lenT = len(T_input)
    lens = len(solar_factor)
    lenR = len(R_input)
    #repeats one axis and tiles the other to make full set
    #T_arg = np.repeat(T_input, lenR)
    s_arg = np.repeat(solar_factor, lenR)
    R_arg = np.tile(R_input, lens)
    '''
    #to add a few select points
    add_R_arg = np.array([3.0e9, 3.5e9, 3.5e9])
    add_s_arg = np.array([1, 1, 0.33])
    
    R_arg = add_R_arg#np.concatenate((R_arg, add_R_arg))
    s_arg = add_s_arg#np.concatenate((s_arg, add_s_arg))
    '''
    directory = 'WM_runs_git/'
    parfile = 'CPML_test_noUV_noions_chemH2_inithydro_06opacityscaling_rho1e-5_M5_t4e9_pythonbatch' #the string of the base file, to be modified in run_aiolos
    ''' #old code to run one-by-one on one axis
    for i in range(len(T_input)):
        run_aiolos(directory, parfile, logfile)
        print('Finished {:.0f}, T_int={:.0f}'.format(i, T_int))
    '''
    #wraps arguments into one
    args_iter = zip(repeat(directory), repeat(parfile), repeat(T_input), R_arg, s_arg)
    
    pool = Pool()
    pool.starmap(run_aiolos, args_iter)
    

    total_time_s = time.time()-time_start
    print('Total time: {:.1f} s = {:.0f} h, {:.0f} m, {:.1f} s'.format(total_time_s, total_time_s//3600, total_time_s%3600//60, total_time_s%3600%60))

if __name__=='__main__':
    main_new()

