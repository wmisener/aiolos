#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:24:14 2023

@author: wmisener
"""
import time
import os
from multiprocessing import Pool

def run_aiolos():
    '''
    Runs aiolos
    '''
    os.system('./aiolos -dir WM_runs_git/ -par CPML_test_noUV_ions_chemH2_gamma3e0_inithydro_06opacityscaling_r24e8_rho1e-5_Tint100.par')

def main_new():
    '''
    Redone version of multi_time_evol which allows simultaneous multiprocessing.
    Uses multiprocessing.Pool to do so.
    '''
    time_start = time.time()
    #folder = '9_by_18_f07_Ts6500_noMdot_Freedmanopacities'
    folder = '3ME_T1000_aiolos_testcases'
    os.mkdir(folder)
    
    #f = 0.07 #initial mass
    f_input = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1]) 
    
    '''
    t_stop = 1e11 #yr
    l_M = 9
    l_T = 18
    M_c_input = np.linspace(2, 10, num=l_M)*M_Earth
    T_eq_input = np.linspace(300, 2000, num=l_T)
    
    #pairs up M_c and T_eq right
    M_c_arr = np.repeat(M_c_input, l_T)
    T_eq_arr = np.tile(T_eq_input, l_M)
    #f_arr = 0.05*(M_c_arr/M_Earth)**(0.5)#0.02*(M_c_arr/M_Earth)**(0.8)*(T_eq_arr/1000)**(-0.25) #Ginz16 Eq 18: t_disk = 1 Myr
    '''
    t_stop = 1e10 #yr
    T_eq = 1000
    M_c = 3*M_Earth
    
    '''
    #l_M = 16
    l_T = 18
    l_f = 10
    
    M_c_input = np.linspace(1, 16, num=l_M)*M_Earth
    #T_input = np.linspace(300, 2000, num=l_T)
    f_input = np.linspace(0.01, 0.1, num=l_f)
    ''''''
    l_T=7
    l_f=6
    M_c_input = np.linspace(1, 16, num=l_M)*M_Earth
    T_input = np.linspace(1400, 2000, num=l_T)
    f_input = np.linspace(0.05, 0.1, num=l_f)
    
    M_c_arr = np.repeat(M_c_input, l_f)
    T_arr = np.repeat(T_input, l_f)
    #f_arr = np.tile(f_input, l_T)
    ''''''
    #manual choice of ints
    #a_t is a list of 2-item lists: [M_c_element_#, T_eq_element_#]
    #a_t = np.array([[0,6],[0,7],[0,8],[0,9],[0,10],[1,10]])
    a_t = np.array([[1,4],[2,2],[2,3],[3,1],[3,2],[4,0],[4,1],[5,0],[6,0],[6,1],[6,2]])
    M_c_arr = M_c_input[a_t[:,0]] #takes first indices from M_c_input
    #T_eq_arr = T_eq_input[a_t[:,1]] #takes second indices from T_eq_input
    f_arr = f_input[a_t[:,1]] #takes second indices from T_eq_input
    '''
    
    #wraps arguments into one
    #args_iter = zip(repeat(f), M_c_arr, T_eq_arr, repeat(t_stop))
    args_iter = zip(f_input, repeat(M_c), repeat(T_eq), repeat(t_stop))
    #args_iter = zip(f_arr, M_c_arr, T_eq_arr, repeat(t_stop))

    kwargs_iter = repeat(dict(savedata=True, status=False, v=False, final_out=True, folder_name=folder))
    pool = Pool()
    data_arr = starmap_w_kwargs(pool, time_evolution_list, args_iter, kwargs_iter) #array of outputs from each time_evol_list run
    #data_massaged = np.insert(data_arr, 0, (M_c_arr, T_eq_arr), axis=1) #adds necessary M_c and T_eq to it 
    #data_massaged = np.insert(data_arr, 0, (M_c, T_eq), axis=1) #adds necessary M_c and T_eq to it 
    '''

    #args_iter = zip(f_arr, M_c_arr, repeat(T_eq), repeat(t_stop))
    args_iter = zip(f_arr, repeat(M_c), T_arr, repeat(t_stop))

    kwargs_iter = repeat(dict(savedata=True, status=False, v=False, final_out=True, folder_name=folder))
    pool = Pool()
    data_arr = starmap_w_kwargs(pool, time_evolution_list, args_iter, kwargs_iter) #array of outputs from each time_evol_list run
    data_massaged = np.insert(data_arr, 0, (T_arr, f_arr), axis=1) #adds necessary M_c and T_eq to it 
    '''
    #otherwise could just make the datafile post facto
    #np.savetxt('{}_datafile.csv'.format(folder), data_massaged, delimiter=',')
    total_time_s = time.time()-time_start
    print('Total time: {:.1f} s = {:.0f} h, {:.0f} m, {:.1f} s'.format(total_time_s, total_time_s//3600, total_time_s%3600//60, total_time_s%3600%60))
'''
if __name__=='__main__':
    main_new()
'''