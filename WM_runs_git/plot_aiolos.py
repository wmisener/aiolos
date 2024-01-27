#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:28:53 2022

@author: williammisener
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import sys
sys.path.append('/Users/wmisener/Documents/python files')
from basic_atmosphere_functions import radiative_detailed_profile, multiple_detailed_profiles, r_from_m

pi = 3.141592653589793238462643383 #value of pi
G = 6.674*10**(-8) #cm^3/g/s^2 #Newtonian Gravitational Constant
k_B = 1.381*10**(-16) #g cm^2/s^2/K #Boltzmann constant
M_Earth = 5.972*10**(27) #g #mass of Earth
R_Earth = 6.378*10**(8) #cm #radius of Earth
m_p = 1.673*10**(-24) #g #mass of a proton
sigma = 5.6704*10**(-5) #g/s^3/K^4 #Stefan-Boltzmann constant
c = 3e10 #cm/s #speed of light
weight_dict = {'H2': 2, 'H0': 1, 'p': 0.95, 'e': 5e-2}

cheat_sheet = {'r':0, 'rho':1, 'mom':2, 'e':3, 'P':10, 'v':11, 'T':12, 'cs':15, 'u_analytic':17, 'pot':19, 'M_tot':20, 'cool':21, 'heat':22, 'tau':23}
line_dict = {'H2':'solid', 'H0':'dashed', 'H+':'dashdot', 'p+':'dashdot', 'p':'dashdot', 'e':'dotted', 'e-':'dotted', 'eh2':'dotted'}

def Guillot_deep_T(T_eq):
    return T_eq/(2**(0.25))

def Guillot_skin_T(F, gamma):
    return (F*gamma/(16*sigma))**(0.25)
'''
def conv_dash(fl):
    if fl==b'-':
        return 0.0
    else:
        return(float(fl))
'''
def plot_aiolos_data_multipanel(t_list, filestring='output_default', t_inc=1e6, species_list=['H0', 'H+', 'e-'], no_hydro=False, plot_hse=False, savefig=False, gamma=30):
    '''
    Plots T(r), rho(r), mach(r), and Mdot(r) for a list of times in one run,
    for a list of species.

    Parameters
    ----------
    t_list : list of times to plot
    filestring : output file string to read in
    t_inc : Timestep increment. The default is 1e6.
    species_list : List of species. The default is ['H0', 'H+', 'e-'].
    no_hydro : If True, omits plotting Mdot and mach. The default is False.
    plot_hse : If True, plots an isothermal hydrostatic density gradient for
        comparison. The default is False.
    gamma : gamma value, solely for setting the Guillot skin temperature 
        horizontal line value. The default is 30.
    savefig : If not False, saves figure to the specified string. The default 
        is False.
    '''
    fig1 = plt.figure(figsize=(8,10))
    custom_lines = []
    custom_labels = []
    bigsize=12
    smallsize=10
    if no_hydro:
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212, sharex=ax1)
    else:
        ax1 = fig1.add_subplot(411)
        ax2 = fig1.add_subplot(412, sharex=ax1)
        ax3 = fig1.add_subplot(413, sharex=ax1)
        ax4 = fig1.add_subplot(414, sharex=ax1)
    
    for i in range(len(t_list)):
        t = t_list[i]
        color = 'C{:.0f}'.format(i)
        for species in species_list:
            filename = filestring + '_{}_t{:.0f}.dat'.format(species, t)
            print(filename)
            with open(filename) as f:
                n = len(list(f))
            ''' #For pulling out ghost cell, but I think this is fine to do in normal loadtxt w/ skiprows=1 instead of 2
            conv_dict = {7: conv_dash, 8: conv_dash, 9:conv_dash, 12: conv_dash, 13:conv_dash, 15: conv_dash, 16: conv_dash, 17: conv_dash, 18: conv_dash, 19: conv_dash, 20: conv_dash, 21: conv_dash, 22: conv_dash}
            data_ghost = np.loadtxt(filename, skiprows=1, max_rows=1, converters=conv_dict)
            print('Ghost:')
            print('T ', data_ghost[cheat_sheet['T']])
            print('rho ', data_ghost[cheat_sheet['rho']])
            ax1.scatter(data_ghost[cheat_sheet['r']], data_ghost[cheat_sheet['T']])
            ax2.scatter(data_ghost[cheat_sheet['r']], data_ghost[cheat_sheet['rho']])
            '''
            data = np.loadtxt(filename, skiprows=1, max_rows=(n-3))
    
            r = data[:,cheat_sheet['r']]
            T = data[:,cheat_sheet['T']]
            rho = data[:,cheat_sheet['rho']]
    
            ax1.scatter(r[0], T[0], color=color)
            ax2.scatter(r[0], rho[0], color=color)
            print('Bottom temp: T = {:.2f}'.format(T[0]))
            print('First five T', T[:5])
            print('Min T', min(T), 'R', r[np.argmin(T)])
            print('Bottom density: rho = {:.4e}'.format(rho[0]))
    
            #print('Bottom density: ', rho[0])
            #print(rho[:5])
            #print(T[:5])
            if t == -1:
                t_label = 'Last t'
            else:
                t_label = 't={:.1e}'.format(t*t_inc)
    
            aiolos_convergence(t, filestring)
            
            ax1.loglog(r, T, label=t_label, color=color, linestyle=line_dict[species])
            ax2.loglog(r, rho, color=color, linestyle=line_dict[species])

            if not no_hydro: #plots mach and Mdot in bottom panels
                mach = data[:,cheat_sheet['v']]/data[:,cheat_sheet['cs']]
                Mdot = 4*pi*data[:,cheat_sheet['mom']]*r*r
                
                for j in range(len(mach)):
                    if mach[j] >= 1:
                        print('Sonic point, r={:.2e} cm'.format(r[j]))
                        print('Mdot(R_s) {:.1e} g/s'.format(Mdot[j]))
                        print('rho(R_s) {:.1e} g/cc'.format(rho[j]))
                        print('Temp(R_s) {:.2f} K'.format(T[j]))
                        Total_M = data[-1, cheat_sheet['M_tot']]
                        if Mdot[j]*t_inc > 0.01*Total_M:
                            print('*** Losing a lot of mass ***')
                            print('Total M in domain {:.2e}'.format(Total_M))
                        break

                ax3.loglog(r, mach, color=color, linestyle=line_dict[species])
                ax4.loglog(r, Mdot, color=color, linestyle=line_dict[species])

            if i == 0: #makes species legend
                custom_lines.append(Line2D([0], [0], color='C0', linestyle=line_dict[species]))
                custom_labels.append(species)

            if plot_hse and species == 'H2': #plots hydrostatic equilibrium from the base
                M_c = 5*M_Earth
                mu = 2*m_p
                R_b = r[0]
                rho_b =rho[0]
                T_iso = 1000
                H_b = k_B*T_iso/mu*R_b**2/(G*M_c)
                H_r_iso = k_B*T_iso/mu*r**2/(G*M_c)                
                rho_hse = rho_b*np.exp(r/H_r_iso-R_b/H_b)
                ax2.loglog(r, rho_hse, color='k', linestyle='dotted')

    ax1.axhline(Guillot_deep_T(1000), color='k', linestyle='dashed')
    ax1.axhline(Guillot_skin_T(2.29e8, gamma), color='k', linestyle='dashed')
    '''
    if t==-1:
        ax3.text(0.7, (0.9-0.1*i), 't=-1', transform=ax4.transAxes, color=color)
    else:
        ax3.text(0.7, (0.3-0.1*i), 't={:.1e} s'.format(t*t_inc), transform=ax3.transAxes, color=color, fontsize=bigsize)
    '''
    ax1.legend(loc='upper right')
    ax2.legend(custom_lines, custom_labels, fontsize=smallsize)
    ax1.set_ylabel('Temperature $T$ (K)', fontsize=bigsize)
    ax2.set_ylabel('Density $\\rho$ (g/cm$^3$)', fontsize=bigsize)
    if no_hydro:
        ax2.set_xlabel('Radius $r$ (cm)', fontsize=bigsize)
    else:
        ax3.set_ylabel('Mach number', fontsize=bigsize)
        ax4.set_ylabel('Mass loss rate $\dot{M}$ (g/s)', fontsize=bigsize)
        ax3.axhline(1, color='k')
        ax4.set_xlabel('Radius $r$ (cm)', fontsize=bigsize)
        ax3.tick_params(axis='both', labelsize=smallsize)
        ax4.tick_params(axis='both', labelsize=smallsize)
    ''' #used to test convergence visually, superseded by aiolos_convergence
    diffs = np.zeros((len(t_list), 4))
    diff_i = 0
    for i in range(len(t_list)):
        
        #diffs[diff_i] = aiolos_convergence(t_list[diff_i], filestring)
        diff_i += 1
    
    fig2 = plt.figure()
    ax21 = fig2.add_subplot(111)
    #print(diffs)
    ax21.scatter(t_list, diffs[:,0], label='T')
    ax21.scatter(t_list, diffs[:,1], label='rho')
    ax21.scatter(t_list, diffs[:,2], label='mach')
    ax21.set_yscale('log')
    ax21.set_xlabel('Time ($10^6$ s)')
    ax21.set_ylabel('Percent difference between $t_i$ and $t_{i-1}$')
    ax21.legend()
    '''
    if savefig:
        fig1.savefig(savefig)

def comp_aiolos_data_multispecies_multipanel(t_i, filestring_list, species_list=['H0', 'H+', 'e-'], T_int=False, plot_tau=False, savefig=False, supply_ax=False, plot_hse=False):
    '''
    Compares multiple runs overplotted, in T(r), rho(r), and either Mdot(r) and 
    mach(r) or tau(r) and kappa_R(r), over a list of species

    A second figure displays these runs by the R_rcb and rho_rcb they 
    correspond to for a range of T_int
    
    Parameters
    ----------
    t_i : Simulation time to compare. This can either be an integer, in which 
        case the same time is used for each run, or a list of equal length as
        filestring_list, in which a different time is used for each run.
    filestring_list : List of output files to plot
    species_list : List of species to plot. The default is ['H0', 'H+', 'e-'].
    T_int : If not False, plots extrapolation of interior profile to RCB, using
        the specified internal temperature. The default is False.
    plot_tau : If True, plots tau(r) and kappa_R(r) in bottom two panels, 
        instead of default mach(r) and Mdot(r). The default is False.
    savefig : If not False, saves figure to the specified string. The default 
        is False.
    supply_ax : If not False, uses specified axis to plot rho_rcb vs r_rcb 
        rather than a new figure. The default is False.
    plot_hse : f True, plots an isothermal hydrostatic density gradient for
        comparison. The default is False.
    '''
    fig1 = plt.figure(figsize=(8,10))
    #fig1 = plt.figure(figsize=(8,2.5))
    
    ax1 = fig1.add_subplot(411)
    ax2 = fig1.add_subplot(412, sharex=ax1)
    ax3 = fig1.add_subplot(413, sharex=ax1)
    ax4 = fig1.add_subplot(414, sharex=ax1)

    if supply_ax:
        ax21 = supply_ax
    ''' #makes second figure showing rho_rcb and r_rcb
    else:
        fig2 = plt.figure()
        ax21 = fig2.add_subplot()
    '''
    #ax5 = fig1.add_subplot(515)
    custom_lines = []
    custom_labels = []
    
    #fig1, axs = plt.subplot_mosaic('AAAAB', layout='constrained', figsize=(8,2))
    bigsize=11
    smallsize=10
    #leg_list = ['$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=30$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=10$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=3$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=1$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=0.3$']
    #leg_list= ['Heavy electrons 5e-10', 'Heavy electrons 5e-19']
    #leg_list = ['$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=0.3$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=1$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=3$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=10$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=30$', '$\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR}=100$']
    leg_list = ['$r=2.0e9$', '$r=2.1e9$', '$r=2.2e9$', '$r=2.3e9$', '$r=2.4e9$', '$r=2.5e9$', '$r=3.0e9$', '$r=3.5e9$']
    #leg_list = ['$r=2.5e9$', '$r=3.0e9$', '$r=3.5e9$', '$r=4.0e9$', '$r=4.5e9$', '$r=5.0e9$']

    for i in range(len(filestring_list)):
        filestring = filestring_list[i]
        color = 'C{:.0f}'.format(i)
        if type(t_i) == list:
            t = t_i[i]
        else:
            t=t_i
        for species in species_list:
            filename = filestring + '_{}_t{:.0f}.dat'.format(species, t)
            print(filename)
            with open(filename) as f:
                n = len(list(f))
                #print(n)
            data = np.loadtxt(filename, skiprows=1, max_rows=(n-3)) #skips first and last line
    
            r = data[:,cheat_sheet['r']]
            T = data[:,cheat_sheet['T']]
            rho = data[:,cheat_sheet['rho']]
            print('base rho={:.2e}'.format(rho[0]))
            print('base T={:.2f}'.format(T[0]))
            #print('First five rho', rho[:5])
            Mdot = 4*pi*data[:,cheat_sheet['mom']]*r*r
            Mdot_Rs = 0
            mach = data[:,cheat_sheet['v']]#/data[:,cheat_sheet['cs']]
            for j in range(len(mach)):
                if mach[j] >= 1:
                    print('Sonic point, r={:.2e}'.format(r[j]))
                    Mdot_Rs = Mdot[j]
                    print('Mdot(R_s) {:.1e} g/s'.format(Mdot_Rs))
                    print('T(R_s) {:.0f}'.format(T[j]))
                    print('rho(R_s) {:.1e} g/cc'.format(rho[j]))
                    break
            
            aiolos_convergence(t, filestring, species=species)
            
            #axs['A'].semilogx(r, T, color=color, linestyle=line_dict[species], label=leg_list[i])
            ax1.semilogx(r, T, color=color, linestyle=line_dict[species])
            ax2.loglog(r, rho, color=color, linestyle=line_dict[species], label=species)
            if plot_tau and species=='H2': #plots tau and kappa in the bottom panels
                diagnostic_filename = 'diagnostic_' + filestring[7:] + '_t{:.0f}.dat'.format(t)
                diagnostic_data = np.loadtxt(diagnostic_filename, skiprows=0, max_rows=(n-3)) #skips first and last line
                r = diagnostic_data[:,0]
                #tau = diagnostic_data[:,11]
                kappa_S = diagnostic_data[:,19]
                kappa_R = diagnostic_data[:,27]
                tau_manual_S = np.zeros(len(r))
                tau_manual_R = np.zeros(len(r))
                for k in range(len(r)):
                    rr = r[k:]
                    rhorho = rho[k:]
                    kappakappa_S = kappa_S[k:]
                    kappakappa_R = kappa_R[k:]
                    tau_manual_S[k] = np.trapz(rhorho*kappakappa_S, x=rr)
                    tau_manual_R[k] = np.trapz(rhorho*kappakappa_R, x=rr)
                print('Bottom Rosseland tau: {:.2e}'.format(tau_manual_R[0]))    
                r_tau_S = r[np.argmin(np.abs(tau_manual_S-1))]
                r_tau_R = r[np.argmin(np.abs(tau_manual_R-1))]
                print('Rosseland tau=1: r={:.2e}'.format(r_tau_R))
                
                ax3.loglog(r, tau_manual_S, color=color, label='tau solar')
                ax3.loglog(r, tau_manual_R, color=color, linestyle='dotted', label='tau ross')
                ax3.axhline(1, color='k')
                
                ax3.legend()
                
                ax4.loglog(r, kappa_R)
                #ax4.loglog(r, 0.1*(rho/1e-3)**(0.6))
                
                #adds red lines at tau=1 surfaces #probably want to convert these to dots or something
                ax3.axvline(r_tau_S, color='red')
                ax3.axvline(r_tau_R, color='red', linestyle='dotted')
                ax2.axvline(r_tau_S, color='red')
                ax2.axvline(r_tau_R, color='red', linestyle='dotted')
                ax1.axvline(r_tau_S, color='red')
                ax1.axvline(r_tau_R, color='red', linestyle='dotted')

            else:
                ax3.loglog(r, mach, color=color, linestyle=line_dict[species])
                #axs['B'].scatter(1, Mdot_Rs, color=color, marker='D')
                ax4.loglog(r, Mdot, color=color, linestyle=line_dict[species], label=leg_list[i])
            if i == 0: #make legend
                ax2.legend(fontsize=smallsize)
                #custom_lines.append(Line2D([0], [0], color='C0', linestyle=line_dict[species]))
                #custom_labels.append(species)
            if species == 'H2':
                custom_lines.append(Line2D([0], [0], color=color, linestyle=line_dict[species]))
                custom_labels.append(leg_list[i])

            M_c = 5*M_Earth
            mu = 2*m_p

            if plot_hse and species == 'H2': #plots hydrostatic equilibrium from the base
                R_b = r[0]
                rho_b =rho[0]
                T_iso = 1000
                H_b = k_B*T_iso/mu*R_b**2/(G*M_c)
                H_r_iso = k_B*T_iso/mu*r**2/(G*M_c)                
                rho_hse = rho_b*np.exp(r/H_r_iso-R_b/H_b)
                ax2.loglog(r, rho_hse, color='k', linestyle='dotted')

            if T_int: #extends analytically to R_rcb
                P_trans = rho[0]/mu*k_B*T[0]
                kappa_trans = 0.1*(rho[0]/1e-3)**(0.6)
                L = 4*pi*r[0]**2*sigma*T_int**4
                dlogTdlogP_trans = 3/64/pi*P_trans*kappa_trans/G/M_c*L/sigma/T[0]**4
                print('T_int = {:.0f}, dlogT/dlogP = {:.2e}'.format(T_int, dlogTdlogP_trans))
    
                rr_in, TT_in, PP_in = radiative_detailed_profile(M_c, T[0], rho[0], T_int, r[0], return_arrs=True)
                rhorho_in = PP_in*mu/(k_B*TT_in)
                ax1.loglog(rr_in, TT_in, label='semi-analytic', color=color, linestyle='dotted')
                ax2.loglog(rr_in, rhorho_in, color=color, linestyle='dotted')
                if plot_tau and species=='H2':
                    tau_manual_S_in = np.zeros(len(rr_in))
                    tau_manual_R_in = np.zeros(len(rr_in))
                    rr_flip = np.flip(rr_in)
                    rhorho_flip = np.flip(rhorho_in) #doing this bc analytically calcs r going outside in, want inside out
                    for j in range(len(rr_in)):
                        rr = rr_flip[j:]
                        rhorho = rhorho_flip[j:]
                        kappakappa_S_in = kappa_S[0]
                        kappakappa_R_in = 0.1*(rhorho/1e-3)**(0.6)
                        tau_manual_S_in[j] = np.trapz(rhorho*kappakappa_S_in, x=rr) + tau_manual_S[0]
                        tau_manual_R_in[j] = np.trapz(rhorho*kappakappa_R_in, x=rr) + tau_manual_R[0]
                    
                    ax3.loglog(rr_flip, tau_manual_S_in, color=color)
                    ax3.loglog(rr_flip, tau_manual_R_in, color=color, linestyle='dotted')
            '''
            #ax21.loglog(r, rho/(m_p*weight_dict[species]), color=color, linestyle=line_dict[species])
            mom = data[:, cheat_sheet['mom']]
            ax21.loglog(r, mom, color=color, linestyle=line_dict[species])
            '''
        '''
        if t==-1:
            ax2.text(0.8, (0.9-0.1*i), 't=-1', transform=ax2.transAxes, color=color)
        else:
            ax2.text(0.8, (0.9-0.1*i), 't={:.1e}'.format(t*t_inc), transform=ax2.transAxes, color=color)
        '''
        '''
        if t==-1:
            ax3.text(0.7, (0.9-0.1*i), 't=-1', transform=ax4.transAxes, color=color)
        else:
            ax3.text(0.7, (0.3-0.1*i), 't={:.1e} s'.format(t*t_inc), transform=ax3.transAxes, color=color, fontsize=bigsize)
        '''

    ax4.set_xlabel('Radius $r$ (cm)', fontsize=bigsize)
    ax1.set_ylabel('Temperature $T$ (K)', fontsize=bigsize)
    ax2.set_ylabel('Density $\\rho$ (g/cc)', fontsize=bigsize)
    if plot_tau:
        ax3.set_ylabel('tau')
        ax4.set_ylabel('kappa_R')
    else:
        ax3.set_ylabel('Mach number', fontsize=bigsize)
        ax4.set_ylabel('Mass loss rate $\dot{M}$ (g/s)', fontsize=bigsize)
    ax3.axhline(1, color='k')
    ax3.tick_params(axis='both', labelsize=smallsize)
    ax4.tick_params(axis='both', labelsize=smallsize)
    ax1.axhline(1000, color='k')
    #ax1.set_xlim(2.3e9, 2.5e9)
    #ax1.set_xlim(1.5e9, 4e10)
    #ax4.axhline(5.9e12, color='k')
    #ax4.set_ylim(1e10, 5e13)
    #ax2.set_ylim(8e-6, 1.1e-5)
    #ax2.legend(fontsize=smallsize)
    #ax4.legend(fontsize=smallsize, loc='center right')
    #fig1.suptitle('Constant $R_\\mathrm{aiolos} = 2.2 \\times 10^9$ cm, $\\rho_\\mathrm{aiolos}=1.0 \\times 10^{-5}$ g/cc, $T_\\mathrm{int} = 150$ K')
    #fig1.suptitle('Constant $\\kappa_\mathrm{vis}/\\kappa_\mathrm{IR} = 3$, $\\rho_\\mathrm{aiolos}=1.0 \\times 10^{-5}$ g/cc, $T_\\mathrm{int} = 150$ K')
    ax4.legend(custom_lines, custom_labels, fontsize=smallsize)
    #fig1.savefig('aiolos_comp_electron_weight.pdf')
    if savefig:
        fig1.savefig(savefig)
    #plots which R_rcb and rho_rcb validly correspond to a run w/ low T_int
    if supply_ax:
        f_a, R_rcb_a, rho_rcb_a = multiple_detailed_profiles(M_c, T[0], rho[0], r[0])
        ax21.semilogx(rho_rcb_a, R_rcb_a)

    '''
    axs['A'].set_xlabel('Radius (cm)', fontsize=bigsize)
    axs['A'].set_ylabel('Temperature (K)', fontsize=bigsize)
    axs['B'].set_ylabel('Mass loss rate (g/s)', fontsize=bigsize)

    axs['A'].set_xlim(3.9e9, 8e10)
    axs['B'].set_yscale('log')
    axs['B'].set_ylim(1e12, 3e13)
    axs['B'].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    axs['A'].axhline(1000, color='k', linewidth=1, zorder=1)
    axs['B'].scatter(1, 5.9e12, color='k', marker='D', facecolors='none')

    axs['A'].legend(fontsize=smallsize, loc='upper right')
    fig1.savefig('proposal_aiolos.pdf')
    '''    

def comp_aiolos_data_multispecies(t_list, filestring_list, species_list=['H0', 'H+', 'e-'], choice=False, t_inc=1e6):
    '''
    Plots one parameter (choice) for multiple species, multiple times, and multiple filestrings

    Parameters
    ----------
    t_list : list of times
    filestring_list : list of output_... files to process
    species_list : list of species
    choice : parameter to plot (default is rho)
    t_inc : Timestep increment. The default is 1e6.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for filestring in filestring_list:
        for i in range(len(t_list)):
            t = t_list[i]
            color = 'C{:.0f}'.format(i)
            if t == -1:
                t_label = 'Last t'
            else:
                t_label = 't={:.1e}'.format(t*t_inc)
            if choice=='S_UV' or choice=='S_bol': #pull from diagnostic files
                diagnostic_filename = 'diagnostic_' + filestring[7:] + '_t{:.0f}.dat'.format(t) #removes 'output_' from filestring
                print(diagnostic_filename)
                with open(diagnostic_filename) as f:
                    n = len(list(f))
                diagnostic_data = np.loadtxt(diagnostic_filename, skiprows=1, max_rows=(n-3))
                r = diagnostic_data[:,0]
                if choice=='S_UV':
                    y = diagnostic_data[:,6]
                elif choice == 'S_bol':
                    y = diagnostic_data[:,7]
                ax.loglog(r, y, label=t_label, color=color)
            else: #pull from species specific output files
                for species in species_list:
                    filename = filestring + '_{}_t{:.0f}.dat'.format(species, t)
                    print(filename)
                    with open(filename) as f:
                        n = len(list(f))
    
                    data = np.loadtxt(filename, skiprows=2, max_rows=(n-3)) #skips first and last line
            
                    r = data[:,cheat_sheet['r']]
                    if choice:
                        if choice == 'Mdot':
                            y = 4*pi*data[:,cheat_sheet['mom']]*r*r
                        elif choice == 'pot':
                            y = -data[:,cheat_sheet[choice]]
                        elif choice == 'mach':
                            y = np.abs(data[:,cheat_sheet['v']]/data[:,cheat_sheet['cs']])
                            for i in range(len(y)):
                                if y[i] >= 1:
                                    print('Sonic point, r={:.2e}'.format(r[i]))
                                    break
                        elif choice == 'mfp':
                            kappa = 7.58
                            y = 1/(data[:,cheat_sheet['rho']]*kappa)
                        elif choice=='S_UV':
                            y = diagnostic_data[:,6]
                        elif choice == 'S_bol':
                            y = diagnostic_data[:,7]
                        else:
                            y = data[:,cheat_sheet[choice]]
                    else:
                        y = data[:,cheat_sheet['rho']]
                    #print(y[:10])
                    print('Max {:.3e}'.format(max(y)))
                    print('Min {:.3e}'.format(min(y)))
                    print('r(max) {:.2e}'.format(r[np.argmax(y)]))
                    print('Last {:.3e}'.format(y[-1]))
                    #lab = None
                    if i == 0:
                        species_lab = species
                    ax.loglog(r, y, label=species_lab, color=color, linestyle=line_dict[species])
        #ax.text(0.8, 0.5-0.1*i, 't={:.0f}'.format(t), transform=ax.transAxes, color=color)
    #ax.legend()
    ax.set_xlabel('r')
    if choice == 'T':
        plt.axhline(Guillot_deep_T(1000), color='k')
        #plt.ylim(100, 2000)
        #e = data[:,cheat_sheet['e']]
        #u = data[:,cheat_sheet['v']]
        #plt.scatter(r, y+ (e*u/sigma)**(0.25), s=2)
        plt.axhline(Guillot_skin_T(2.29e8, 30))
        plt.ylim(100, 2000)
    if choice:
        ax.set_ylabel(choice)
    if choice == 'S_UV':
        plt.ylim(1e-20, 1e0)
    else: 
        ax.set_ylabel('rho')
    #fig.savefig('photochem_M15_rho-9_heat.pdf')

def aiolos_convergence(t_i, filestring, species='H2'):
    '''
    Tests convergence, i.e., whether mass loss rate max variation between 
    timestep t_i and timestep t_i-1 is greater than 5%. Prints outcome (True or
    False).
    
    Code contains commented out lines calculating other differences, decided 
    5% and Mdot were best heuristics, but any mostly work.
    
    Parameters
    ----------
    t_i : int
        Timestep value.
    filestring : string
        Output filename.
    species : string, optional
        Species name. The default is 'H2'.

    Returns
    -------
    None.

    '''
    t_list = [t_i-1, t_i]
    #T_comp_l = []
    #rho_comp_l = []
    Mdot_comp_l = []
    #mach_comp_l = []
    
    for i in range(len(t_list)):
        #color = 'C{:.0f}'.format(i)
        t = t_list[i]
        filename = filestring + '_{}_t{:.0f}.dat'.format(species, t)
        #print(filename)
        with open(filename) as f:
            n = len(list(f))
            #print(n)
        data = np.loadtxt(filename, skiprows=1, max_rows=(n-3)) #skips first and last line

        r = data[:,cheat_sheet['r']]
        #T = data[:,cheat_sheet['T']]
        #rho = data[:,cheat_sheet['rho']]
        #print('base rho={:.2e}'.format(rho[0]))
        #print('base T={:.2f}'.format(T[0]))
        Mdot = 4*pi*data[:,cheat_sheet['mom']]*r*r
        #mach = data[:,cheat_sheet['v']]/data[:,cheat_sheet['cs']]

        #T_comp_l.append(T)
        #rho_comp_l.append(rho)
        Mdot_comp_l.append(Mdot)
        #mach_comp_l.append(mach)
    #T_comp_a = np.array(T_comp_l)
    #rho_comp_a = np.array(rho_comp_l)
    Mdot_comp_a = np.array(Mdot_comp_l)
    #mach_comp_a = np.array(mach_comp_l)
    division_len = -(len(t_list)-1)
    
    #T_diff = np.abs(np.diff(T_comp_a, axis=0)/T_comp_a[division_len:])#np.abs((T_comp_a[1]-T_comp_a[0])/T_comp_a[1])
    #rho_diff = np.abs(np.diff(rho_comp_a, axis=0)/rho_comp_a[division_len:])#np.abs((rho_comp_a[1] - rho_comp_a[0])/rho_comp_a[1])
    Mdot_diff = np.abs(np.diff(Mdot_comp_a, axis=0)/Mdot_comp_a[division_len:])#np.abs((Mdot_comp_a[1] - Mdot_comp_a[0])/Mdot_comp_a[1])
    #mach_diff = np.abs(np.diff(mach_comp_a, axis=0)/mach_comp_a[division_len:])#np.abs((mach_comp_a[1] - mach_comp_a[0])/mach_comp_a[1])
    print('**Convergence (delta Mdot < 5%): {} **'.format(np.max(Mdot_diff) < 0.05))
    #return np.max(T_diff), np.max(rho_diff), np.max(Mdot_diff), np.max(mach_diff) #for plotting, no longer used
    ''' #tried taking difference between differences (i.e. second derivative), but decided it didn't work any better
    max_T_diff = np.max(T_diff, axis=1)
    max_rho_diff = np.max(rho_diff, axis=1)
    max_mach_diff = np.max(mach_diff, axis=1)
    max_Mdot_diff = np.max(Mdot_diff, axis=1)
    return np.abs(np.diff(max_T_diff)/max_T_diff[-1]), np.abs(np.diff(max_rho_diff)/max_rho_diff[-1]), np.abs(np.diff(max_mach_diff)/max_mach_diff[-1]), np.abs(np.diff(max_Mdot_diff)/max_Mdot_diff[-1])
    print('T', np.abs(np.diff(max_T_diff)/max_T_diff[-1]))
    print('rho', np.abs(np.diff(max_rho_diff)/max_rho_diff[-1]))
    print('mach', np.abs(np.diff(max_mach_diff)/max_mach_diff[-1]))
    print('Mdot', np.abs(np.diff(max_Mdot_diff)/max_Mdot_diff[-1]))
    '''
    
    
def plot_aiolos_diagnostic(t_list, arg_list, filestring='diagnostic_default_rad_multi_t{:.0f}.dat'):
    for t in t_list:
        filename = filestring.format(t)
        with open(filename) as f:
            n = len(list(f))
            #print(n)
        data = np.loadtxt(filename, skiprows=2, max_rows=(n-3)) #skips first and last line
        r = data[:,0]
        for i in arg_list:
            plt.loglog(r, data[:,i], label=i)
            print(i, data[:5,i])
    plt.legend()

def plot_aiolos_fluxes(t, filestring='diagnostic_default_rad_multi_t{:.0f}.dat'):
    filename = filestring.format(t)
    with open(filename) as f:
        n = len(list(f))
        #print(n)
    data = np.loadtxt(filename, skiprows=2, max_rows=(n-3)) #skips first and last line
    r = data[:,0]
    J = data[:,3]
    S = data[:,2]
    T = data[:,29]
        
    plt.loglog(r, 4*pi*J, label='4 pi J')
    #plt.loglog(r, S/4, label='S/4')
    plt.loglog(r, 4*sigma*T**4, label='4pi sigma T4')
    plt.axhline(4*sigma*150**4, color='k', label='sigma*T_int^4')
    print(max(4*sigma*T**4))
    print(max(4*pi*J))
    plt.legend()
    #plt.ylim(1e5, 1e10)
    #plt.savefig('fluxes.pdf')

def plot_aiolos_energy(t, filestring='output_default', species_list=['H2']):
    for species in species_list:
        filename = filestring + '_{}_t{:.0f}.dat'.format(species, t)
        diagnostic_filename = 'diagnostic_' + filestring[7:] + '_t{:.0f}.dat'.format(t)
        print(filename)
        print(diagnostic_filename)
        with open(filename) as f:
            n = len(list(f))
            #print(n)
        data = np.loadtxt(filename, skiprows=2, max_rows=(n-3)) #skips first and last line

        r = data[:,cheat_sheet['r']]
        e = data[:,cheat_sheet['e']]
        rho = data[:,cheat_sheet['rho']]
        u = data[:,cheat_sheet['v']]
        Phi = data[:,cheat_sheet['pot']]
        P = data[:,cheat_sheet['P']]
        T = data[:,cheat_sheet['T']]
        plt.loglog(r, (e*u/sigma)**(0.25))
        #plt.loglog(r)
        '''
        gradu = np.gradient(u, r)
        divu = np.gradient(u*r**2, r)/r**2
        #plt.loglog(r, e/rho*gradu, label='e grad u')
        #plt.loglog(r, P/rho*divu, label='P/rho div u')
        #plt.loglog(r, e/rho*gradu - P/rho*divu, label='LHS')
        LHS = u*(e+P)
        dLHS = np.gradient(LHS*r**2, r)/r**2/rho
        RHS = u*np.gradient(Phi, r)
        plt.loglog(r, dLHS, label='u(E+P)')
        plt.loglog(r, RHS, label='grav')
        plt.loglog(r, dLHS-RHS, label='LHS')
        diagnostic_data = np.loadtxt(diagnostic_filename, skiprows=2, max_rows=(n-3)) #skips first and last line
        r = diagnostic_data[:,0]
        J = diagnostic_data[:,3]
        S = diagnostic_data[:,2]
        kappa = diagnostic_data[:,23]
        kappaS = diagnostic_data[:,19]
        T = diagnostic_data[:,29]

        plt.loglog(r, kappa*4*pi*J, label = 'kappa 4 pi J')
        plt.loglog(r, kappa*4*sigma*T**4, label = 'kappa sigma T4')
        plt.loglog(r, kappaS*S/4, label = 'kappa S/4')
        plt.loglog(r, kappa*4*pi*J + kappaS*S/4 - kappa*4*sigma*T**4, label='diff')
        plt.legend()
        '''
        '''
        TE = rho*3*k_B*T/(2*m_p)
        GE = -rho*Phi
        KE = 0.5*rho*u**2
        plt.loglog(r, e/rho, label='e')
        plt.loglog(r, KE/rho, label='KE')
        #plt.loglog(r, GE, label='Grav PE')
        plt.loglog(r, TE/rho, label='TE')
        plt.loglog(r, P/rho, label='P/rho')
        #plt.loglog(r, (KE+TE)/rho, label='KE+TE')
        plt.loglog(r, (e+P)/rho, label='h')
        #plt.loglog(r, np.abs((KE+TE-e)/e))
        #plt.loglog(r, 0.5*rho*u**2 - rho*Phi, label='Sum')
        #plt.loglog(r, P)
        #plt.loglog(r, abs(e - (0.5*rho*u**2 - rho*Phi)), label='Diff')
        '''
    plt.legend()
    plt.xlabel('Radius (cm)')
    #plt.ylabel('Specific Energy (erg/g)')
    plt.ylabel('Energy Flux (erg/cm^2/s)')
    #plt.ylabel('Specific Energy Flux (erg/g/s)')
    #plt.ylim(1e5, 1e11)
    #plt.savefig('energy_mass.pdf')

def plot_aiolos_temp(t, filestring='output_default', species_list=['H2'], T_int=200, T_eq=1000):
    T_irr = T_eq*4**0.25
    for species in species_list:
        filename = filestring + '_{}_t{:.0f}.dat'.format(species, t)
        diagnostic_filename = 'diagnostic_' + filestring[7:] + '_t{:.0f}.dat'.format(t)
        print(filename)
        print(diagnostic_filename)
        with open(filename) as f:
            n = len(list(f))
            #print(n)
        data = np.loadtxt(filename, skiprows=2, max_rows=(n-3)) #skips first and last line
        #r = data[:,cheat_sheet['r']]
        rho = data[:, cheat_sheet['rho']]
        T = data[:,cheat_sheet['T']]
        #plt.semilogx(r,T)
        diagnostic_data = np.loadtxt(diagnostic_filename, skiprows=1, max_rows=(n-3)) #skips first and last line
        r = diagnostic_data[:,0]
        tau = diagnostic_data[:,11]
        kappa_S = diagnostic_data[:,19]
        kappa_P = diagnostic_data[:,23]
        kappa_R = diagnostic_data[:,27]
        tau_manual_S = np.zeros(len(r))
        tau_manual_P = np.zeros(len(r))
        for i in range(len(r)):
            rr = r[i:]
            rhorho = rho[i:]
            kappakappa_S = kappa_S[i:]
            kappakappa_P = kappa_P[i:]
            tau_manual_S[i] = np.trapz(rhorho*kappakappa_S, x=rr)
            tau_manual_P[i] = np.trapz(rhorho*kappakappa_P, x=rr)
        r_tau_S = r[np.argmin(np.abs(tau_manual_S-1))]
        r_tau_P = r[np.argmin(np.abs(tau_manual_P-1))]
        plt.semilogy(r, tau, label='diagnostic 11')
        plt.semilogy(r, tau_manual_S, label='tau solar')
        plt.semilogy(r, tau_manual_P, label='tau planck')
        plt.axvline(r_tau_S, color='C1')
        plt.axvline(r_tau_P, color='C2')
        plt.axhline(1, color='k')
        '''
        T_tau = (T_int**4*(3*tau/4+1/2) + 0.93*0.5*T_irr**4)**(1/4)
        f_H = 1/2
        f =0.25
        mustar=1
        gamma=1
        T_tau_SB65 = (T_int**4*(3*tau/4+1/(4*f_H))+3*mustar/4*f*T_irr**4*(1/(3*f_H)+mustar/gamma+(gamma/(3*mustar)-mustar/gamma)*np.exp(-tau*gamma/mustar)))**0.25
        T_tau_CS18 = (T_eq**4+tau*T_int**4)**(0.25)
        plt.semilogx(r, T_tau)
        plt.semilogx(r, T_tau_SB65)
        plt.semilogx(r, T_tau_CS18)
        '''