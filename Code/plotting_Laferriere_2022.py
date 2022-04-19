#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:14:23 2022

@author: laferrierek
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
from matplotlib.lines import Line2D  
import matplotlib
from scipy import stats
from scipy.optimize import curve_fit
import math
from scipy import constants as const
import glob

# define known things
xsize = 64
wl = 512
wv = wl
loc = '/Users/laferrierek/Desktop/Moon_Papers/'
plt.rc("font", size=14, family="serif")
res = 350

# Functions
def readin(filename, sizez, sizey, sizex):
    # Reads in any fit file into a cube shape
    # Note: python flips the cubes when plotting, but saves the same way it's read in. 
    data = np.zeros((sizez, sizey, sizex))
    with fits.open(filename) as f_hdul:
        data[:, :, :] = f_hdul[0].data
    return data

def resmean(data, sigma_cut):
    """ built from the resistant mean method from IDL that Lori gave me"""
    data = data[~np.isnan(data)]
    data_median = np.median(data)
    absdev = np.abs(data - data_median)
    medabsdev = np.median(absdev) / 0.6745
    if medabsdev <= 1.0 * np.exp(-24):
        medabsdev = np.average(absdev) / 0.8
    cutoff = sigma_cut * medabsdev
    goodpts = data[absdev < cutoff]
    mean = np.average(goodpts)
    num_good = np.size(goodpts)
    sigma = np.sqrt(np.sum((goodpts - mean) ** 2) / num_good)
    sc = sigma_cut
    if sc < 1.75:
        sc = 1.75
    if sigma <= 3.4:
        sigma = sigma / (0.18553 + 0.505246 * sc - 0.0784189 * sc * sc)
    cutoff = sigma_cut * sigma
    goodpts = data[absdev <= cutoff]
    mean = np.average(goodpts)
    num_good = np.size(goodpts)
    sigma = np.sqrt(np.sum((goodpts - mean) ** 2) / num_good)
    sc = sigma_cut
    if sc < 1.75:
        sc = 1.75
    if sigma <= 3.4:
        sigma = sigma / (0.18553 + 0.505246 * sc - 0.0784189 * sc * sc)
    sigma = sigma / np.sqrt(np.size(data) - 1)
    return mean, sigma

# for model
temp_guess = 360
m_guess = 0.01
const_guess =  0.02
e_guess = 0.9

h = const.h
c = const.c
k = const.k

def scatter(wavelength, m, constant):
    # Scattered model is just a linear line. 
    return m * wavelength + constant

def planck(wavelength, temperature, emiss):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-emiss)*top 
    return them

def planck_14(wavelength, temperature):
    # Thermal part with emissivity held. 
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_14(wave, im, modtran, incidence, y, x, ysize, plots):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_14, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_14(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])

    # Refit thermal, with scatter removed, emissisivity as 1-scat
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi

    # Define an emissivty which is linear, but not a constant.
    emiss_a = scat_part_2 #1- scat_part_2
    emiss_fit =  scat_part_2[min_c:max_c][index_therm_2] #1 -
    
    # Fit thermal with this emissivity to get new temperature
    def planck_fit_14(wavelength, temperature):
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top
        return them
    
    opt, cov = curve_fit(planck_fit_14, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_fit = emiss_a
    therm_part_2 = planck_fit_14(wave[:, y, x], temp_2)
    temp_a = temp_2
    
    # Now, allow emissivity to vary with a slope and intercept
    def planck_fit(wavelength, slope_check, interc):
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temp_a)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        emissivity = scatter(wavelength*10**(6), slope_check, interc)
        them = (1-emissivity)*top
        return them
    
    # Fit the values
    index_therm_3 = ~np.isnan(im_adj[min_c:max_c])
    opt, cov = curve_fit(planck_fit, wave[min_c:max_c, y, x][index_therm_3], im_adj[min_c:max_c][index_therm_3], bounds=[[-0.1, 0.9],[0.1, 1.1]])
    emiss_2 = scatter(wave[:, y, x], opt[0], opt[1])   
    emiss_s1 = emiss_2
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2)
   
    # Refit scatter with this newly fit emissivity removed. 
    im_adj_2 = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    opt, cov = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2])
    new_scat = scatter(wave[:, y, x], opt[0], opt[1])

    # Outputs for the scattered light
    m_3 = opt[0]
    con_3 = opt[1]
    unc_m = np.sqrt(cov[0, 0])
    unc_con = np.sqrt(cov[1, 1])

    # Now, we refit for temperature again, with this final scattered light removed.  
    im_adj_3 = im[:, y, x] - new_scat*modtran[:, y, x]*cosi/np.pi
    emiss_fit = emiss_2[min_c:max_c][index_therm_2]
    opt, covt = curve_fit(planck_fit_14, wave_therm_2[index_therm_2], im_adj_3[min_c:max_c][index_therm_2], p0=[temp_2])

    thermal_part_3 = planck(wave[:, y, x], opt[0], emiss_2) 
    emiss_1a = emiss_2
    
    # Finally, we refit the thermal region with our final temperature. 
    index_therm_3 = ~np.isnan(im_adj_3[min_c:max_c])
    temp_a = opt[0]
    opt, cov = curve_fit(planck_fit, wave[min_c:max_c, y, x][index_therm_3], im_adj_3[min_c:max_c][index_therm_3], bounds=[[-0.1, 0.9],[0.1, 1.1]])#, p0=[0, 0.72], bounds=[[-0.1, 0.5],[0.1, 0.95]])

    # Final thermal products
    emiss_2 = scatter(wave[:, y, x], opt[0], opt[1])
    thermal_part_4 = planck(wave[:, y, x], temp_a, emiss_2)
    temp_4 = temp_a
      
    #Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_4)*np.pi/(modtran[:, y, x]*cosi)

    params = np.array([temp_4, m_3, con_3, np.nanmean(emiss_2)] )
    return app_ref, thermal_part_4, new_scat, modtran, cosi

#%% Data

# hibd, d339, 346
ysize =100
d33900_hibd = readin(loc+'IBD_33900.fit', 2, ysize, xsize)
d33901_hibd = readin(loc+'IBD_33901.fit', 2, ysize, xsize)
d33902_hibd = readin(loc+'IBD_33902.fit', 2, ysize, xsize)
d34600_hibd = readin(loc+'IBD_34600.fit', 2, ysize, xsize)
d34601_hibd = readin(loc+'IBD_34601.fit', 2, ysize, xsize)
d34602_hibd = readin(loc+'IBD_34602.fit', 2, ysize, xsize)

# bd
d33900_bd = readin(loc+'BD_33900.fit', 512, ysize, xsize)
d33901_bd = readin(loc+'BD_33901.fit', 512, ysize, xsize)
d33902_bd = readin(loc+'BD_33902.fit', 512, ysize, xsize)
d34600_bd = readin(loc+'BD_34600.fit', 512, ysize, xsize)
d34601_bd = readin(loc+'BD_34601.fit', 512, ysize, xsize)
d34602_bd = readin(loc+'BD_34602.fit', 512, ysize, xsize)

# res mean bd
d33900_bd_reseman = readin(loc+'d33900_bd_resmean.fit', 512, ysize, xsize)
d33901_bd_reseman = readin(loc+'d33901_bd_resmean.fit', 512, ysize, xsize)
d33902_bd_reseman = readin(loc+'d33902_bd_resmean.fit', 512, ysize, xsize)

# parameters
d33900_parameters = readin(loc+'Parameters_33900.fit', 8, ysize, xsize)
d33901_parameters = readin(loc+'Parameters_33901.fit', 8, ysize, xsize)
d33902_parameters = readin(loc+'Parameters_33902.fit', 8, ysize, xsize)
d34600_parameters = readin(loc+'Parameters_34600.fit', 8, ysize, xsize)
d34601_parameters = readin(loc+'Parameters_34601.fit', 8, ysize, xsize)
d34602_parameters = readin(loc+'Parameters_34602.fit', 8, ysize, xsize)

# pyroxene ibd
d33900_pibd= readin(loc+'d33900_pibd.fit', 2, ysize, xsize)
d33901_pibd= readin(loc+'d33901_pibd.fit', 2, ysize, xsize)
d33902_pibd= readin(loc+'d33902_pibd.fit', 2, ysize, xsize)
d34600_pibd= readin(loc+'d34600_pibd.fit', 2, ysize, xsize)
d34601_pibd= readin(loc+'d34601_pibd.fit', 2, ysize, xsize)
d34602_pibd= readin(loc+'d34602_pibd.fit', 2, ysize, xsize)

# d352
ysize = 135
d35200_hibd = readin(loc+'IBD_35200.fit', 2, ysize, xsize)
d35201_hibd = readin(loc+'IBD_35201.fit', 2, ysize, xsize)
d35202_hibd = readin(loc+'IBD_35202.fit', 2, ysize, xsize)

d35200_bd = readin(loc+'BD_35200.fit', 512, ysize, xsize)
d35201_bd = readin(loc+'BD_35201.fit', 512, ysize, xsize)
d35202_bd = readin(loc+'BD_35202.fit', 512, ysize, xsize)

d35200_parameters = readin(loc+'Parameters_35200.fit', 8, ysize, xsize)
d35201_parameters = readin(loc+'Parameters_35201.fit', 8, ysize, xsize)
d35202_parameters = readin(loc+'Parameters_35202.fit', 8, ysize, xsize)

d35200_pibd= readin(loc+'d35200_pibd.fit', 2, ysize, xsize)
d35201_pibd= readin(loc+'d35201_pibd.fit', 2, ysize, xsize)
d35202_pibd= readin(loc+'d35202_pibd.fit', 2, ysize, xsize)

# 
d33900_im = readin(loc+'Calibrated_cube_d33900.fit', wl, 100, xsize)
d33900_sat = readin(loc+'Satmask_cube_d33900_fixed.fit', 1, 100, xsize)
mask_sat = d33900_sat == 0
d33900_sat[~mask_sat] = np.nan
d33900_sat[mask_sat] = 1

d33901_im = readin(loc+'Transfer/Calibrated_cube_d33901.fit', wl, 100, xsize)
d33901_sat = readin(loc+'Transfer/Satmask_cube_d33901_fixed.fit', 1, 100, xsize)
mask_sat = d33901_sat == 0
d33901_sat[~mask_sat] = np.nan
d33901_sat[mask_sat] = 1

d33902_im = readin(loc+'Calibrated_cube_d33902.fit', wl, 100, xsize)
d33902_sat = readin(loc+'Satmask_cube_d33902_fixed.fit', 1, 100, xsize)
mask_sat = d33902_sat == 0
d33902_sat[~mask_sat] = np.nan
d33902_sat[mask_sat] = 1

# 346
d34601_im = readin(loc+'Dec1201/Calibrated_cube_d34601.fit', wl, 100, xsize)
d34601_sat = readin(loc+'Satmask_cube_d34601_fixed.fit', 1, 100, xsize)
mask_sat = d34601_sat == 0
d34601_sat[~mask_sat] = np.nan
d34601_sat[mask_sat] = 1

d34600_im = readin(loc+'Calibrated_cube_d34600.fit', wl, 100, xsize)
d34600_sat = readin(loc+'Satmask_cube_d34600_fixed.fit', 1, 100, xsize)
mask_sat = d34600_sat == 0
d34600_sat[~mask_sat] = np.nan
d34600_sat[mask_sat] = 1

d34602_im = readin(loc+'Calibrated_cube_d34602.fit', wl, 100, xsize)
d34602_sat = readin(loc+'Satmask_cube_d34602_fixed.fit', 1, 100, xsize)
mask_sat = d34602_sat == 0
d34602_sat[~mask_sat] = np.nan
d34602_sat[mask_sat] = 1

# 352
d35201_im = readin(loc+'Dec1801/Calibrated_cube_d35201.fit', wl, 135, xsize)
d35201_sat = readin(loc+'Satmask_cube_d35201_fixed.fit', 1, 135, xsize)
mask_sat = d35201_sat == 0
d35201_sat[~mask_sat] = np.nan
d35201_sat[mask_sat] = 1

d35200_im = readin(loc+'Calibrated_cube_d35200.fit', wl, 135, xsize)
d35200_sat = readin(loc+'Satmask_cube_d35200_fixed.fit', 1, 135, xsize)
mask_sat = d35200_sat == 0
d35200_sat[~mask_sat] = np.nan
d35200_sat[mask_sat] = 1

d35202_im = readin(loc+'Calibrated_cube_d35202.fit', wl, 135, xsize)
d35202_sat = readin(loc+'Satmask_cube_d35202_fixed.fit', 1, 135, xsize)
mask_sat = d35202_sat == 0
d35202_sat[~mask_sat] = np.nan
d35202_sat[mask_sat] = 1

# Repeat data
repeats = np.loadtxt(loc+'d1d2d3_repeats_01_full.txt', delimiter=' ', unpack=True)
repeats = (repeats).astype('int')
repeats = repeats[:, 0:80]

d33901_repeat = repeats[0:2, :] 
d34601_repeat = repeats[2:4, :]
d35201_repeat =  repeats[4:6, :]

# wave
d33901_wave = readin(loc+'Transfer/Wave_cube_d33901.fit', wl, 100, xsize)
d33901_wave = d33900_wave = d33902_wave = readin(loc+'/Transfer/Wave_cube_d33901.fit', 512, 100, 64)
d34600_wave = d34601_wave = d34602_wave = readin(loc+'Dec1201/Wave_cube_d34601.fit', 512, 100, 64)
d35200_wave = d35201_wave = readin(loc+'/Dec1801/Wave_cube_d35201.fit', 512, 135, 64)

# Incidence angle
sec = 'Transfer/'
ysize = 100
d33900_inc = readin(loc+sec+'incidence_ang_Dec05_0.fit', 2, ysize, xsize)
d33901_inc = readin(loc+sec+'incidence_ang_Dec05_1.fit', 1, ysize, xsize)
d33902_inc = readin(loc+sec+'incidence_ang_Dec05_2.fit', 2, ysize, xsize)
sec = 'Dec1201/'
d34600_inc = readin(loc+sec+'incidence_ang_Dec12_0.fit', 1, ysize, xsize)
d34601_inc = readin(loc+sec+'incidence_ang_Dec12_1.fit', 1, ysize, xsize)
d34602_inc = readin(loc+sec+'incidence_ang_Dec12_2.fit', 1, ysize, xsize)
sec = 'Dec1801/'
ysize = 135
d35200_inc = readin(loc+sec+'incidence_ang_Dec18_0.fit', 1, ysize, xsize)
d35201_inc = readin(loc+sec+'incidence_ang_Dec18_1.fit', 1, ysize, xsize)
d35202_inc = readin(loc+sec+'incidence_ang_Dec18_2.fit', 1, ysize, xsize)

# New bd
d33900_bd_new_resmean = readin(loc+'d33900_bd_new_resmean.fit', 512, 100, 64)
d33901_bd_new_resmean = readin(loc+'d33901_bd_new_resmean.fit', 512, 100, 64)
d33902_bd_new_resmean = readin(loc+'d33902_bd_new_resmean.fit', 512, 100, 64)

d34600_bd_new_resmean = readin(loc+'d34600_bd_new_resmean.fit', 512, 100, 64)
d34601_bd_new_resmean = readin(loc+'d34601_bd_new_resmean.fit', 512, 100, 64)
d34602_bd_new_resmean = readin(loc+'d34602_bd_new_resmean.fit', 512, 100, 64)

d35200_bd_new_resmean = readin(loc+'d35200_bd_new_resmean.fit', 512, 135, 64)
d35201_bd_new_resmean = readin(loc+'d35201_bd_new_resmean.fit', 512, 135, 64)
d35202_bd_new_resmean = readin(loc+'d35202_bd_new_resmean.fit', 512, 135, 64)

# tod
d33901_tod = readin(loc+'d33901_tod_xymap.fit', 1, 100, 64)
d34601_tod = readin(loc+'d34601_tod_xymap.fit', 1, 100, 64)
d35201_tod = readin(loc+'d35201_tod_xymap.fit', 1, 135, 64)

d33901_app_ref = readin(loc+'App_refl_33901.fit', 512, 100, 64)
d34601_app_ref = readin(loc+'App_refl_34601.fit', 512, 100, 64)
d35201_app_ref = readin(loc+'App_refl_35201.fit', 512, 135, 64)

modtran = readin(loc+'Modtran_d33901.fit', 512, 100, 64)

#%% Or, data
def data(st, ysize):
    parameters = readin(loc+'Parameters_%5.0f.fit'%st, 8, ysize, xsize)
    ibd = readin(loc+'IBD_%5.0f.fit'%st, 2, ysize, xsize)
    app = readin(loc+'App_refl_%5.0f.fit'%st, wv, ysize, xsize)
    bd = readin(loc+'BD_%5.0f.fit'%st, wv, ysize, xsize)
    st = str(st)
    date = st[0:3]
    scan = st[3:5]
    datescan = 'day'+ date +'/' +scan+'/'
    mypath = loc+'wave_data_day'+ date + '**.fit'
    for fpath in glob.glob(mypath):
        wave = readin(fpath, wv, ysize, xsize)
    latlong = readin(loc +'pix_to_longlat_day'+date+scan+'.fit', 2, ysize, xsize)
    return parameters, ibd, app, bd, wave, latlong

d33900_parameters, d33900_hibd, d33900_app, d33900_3bd, d33900_wave,  d33900_latlong = data(33900, 100)
d33901_parameters, d33901_hibd, d33901_app, d33901_3bd, d33901_wave,  d33901_latlong = data(33901, 100)
d33902_parameters, d33902_hibd, d33902_app, d33902_3bd, d33902_wave,  d33902_latlong = data(33902, 100)
d34600_parameters, d34600_hibd, d34600_app, d34600_3bd, d34600_wave,  d34600_latlong = data(34600, 100)
d34601_parameters, d34601_hibd, d34601_app, d34601_3bd, d34601_wave,  d34601_latlong = data(34601, 100)
d34602_parameters, d34602_hibd, d34602_app, d34602_3bd, d34602_wave,  d34602_latlong = data(34602, 100)
d35200_parameters, d35200_hibd, d35200_app, d35200_3bd, d35200_wave,  d35200_latlong = data(35200, 135)
d35201_parameters, d35201_hibd, d35201_app, d35201_3bd, d35201_wave,  d35201_latlong = data(35201, 135)
d35202_parameters, d35202_hibd, d35202_app, d35202_3bd, d35202_wave,  d35202_latlong = data(35202, 135)

#%% Figure 1 - not finalize (powerpoint)

wvl = 19
fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d33900_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d33900_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d33901_repeat[1, :]-23, d33901_repeat[0, :], c='b', s=7)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d33901_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d33901_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d33901_repeat[1, :], d33901_repeat[0, :], c='b', s=7)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d33902_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d33902_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d33901_repeat[1, :]+23, d33901_repeat[0, :], c='b', s=7)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

#
fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d34600_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d34600_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d34601_repeat[1, :]-23, d34601_repeat[0, :], c='b', s=15)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d34601_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d34601_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d34601_repeat[1, :], d34601_repeat[0, :], c='b', s=15)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d34602_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d34602_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d34601_repeat[1, :]+23, d34601_repeat[0, :], c='b', s=15)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

#
fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d35200_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d35200_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d35201_repeat[1, :]-23, d35201_repeat[0, :], c='b', s=15)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d35201_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d35201_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d35201_repeat[1, :], d35201_repeat[0, :], c='b', s=15)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

fig = plt.figure(figsize=(8, 10), dpi=res)
plt.imshow(d35202_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
plt.imshow(d35202_sat[0, :, :], cmap='copper_r', alpha=0.6)
plt.scatter(d35201_repeat[1, :]+23, d35201_repeat[0, :], c='b', s=15)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()
#%% Figure 2
# Mare oriental - [-19.5, 265]
# mare Humorum - [-23.75, 321.5]
# Mare Nubium - [-19.5, 343]
# Mare Austrile - [-47, 92]
# Aitken Basin - [-16, 172.9]
# Schrodinger - [-74, 133]
# South Pole - [-90, 0]
# Tsiolkovskiv - [-20, 129]
# Apollo - [-35, 208]
# Pyrgius A - [-24, 296]
# Schickard - [-44, 305]
# Zeemna - [-75, 225]

latlong = d33901_latlong
im = d33901_im
th = 2

plt.figure(figsize=(5,5 ), dpi=300)
# Mare
lat = np.array([-19.5, -23.5, -21.3]) #, -47])
long = np.array([265, 322.5, 343.4]) #, 92])
name = np.array(['Oriental', 'Humorum', 'Nubium']) #, 'Australe'])
color = ['dodgerblue', 'orange', 'green', 'fuschia']
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
for i in range(0, np.size(lat)):
    plt.scatter(y[i], x[i], c=color[i], marker='^')
    plt.annotate(str(name[i]), (y[i]+6, x[i]-2), c=color[i], size=10)
    plt.annotate('Mare', (y[i]+6, x[i]+2), c=color[i], size=10)

# Craters
lat = np.array([  -44])
long = np.array([ 305])
name = np.array([ 'Schickard'])
color = ['darkturquoise', ]
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
for i in range(0, np.size(lat)):
    plt.scatter(y[i], x[i], c=color[i], marker='.')
    plt.annotate(str(name[i]), (y[i]+11, x[i]+5), c=color[i], size=10)
    plt.annotate('Crater', (y[i]+11, x[i]), c=color[i], size=10)

# SP
lat = np.array([ -89])
long = np.array([  341]) #,  2,  4])
name = np.array([ 'South Pole'])
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
plt.scatter(y, x, c='limegreen', marker='.', vmin=0, vmax=10)
plt.annotate('South', (y[i]-10, x[i]+2), c='limegreen', size=10)
plt.annotate('Pole', (y[i]-10, x[i]-2), c='limegreen', size=10)
plt.imshow(im[wvl, :, :], vmin=-1 , vmax =25, cmap='Greys_r')
plt.gca().invert_yaxis()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

latlong = d34601_latlong
im = d34601_im
th = 2
plt.figure(figsize=(5,5 ), dpi=300)
# Mare
lat = np.array([-19.5, -23.5]) #, -47])
long = np.array([265, 322.5]) #, 92])
name = np.array(['Oriental', 'Humorum']) #, 'Australe'])
color = ['dodgerblue', 'orange']
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
i = 0
plt.scatter(y[i], x[i], c=color[i], marker='^')
plt.annotate(str(name[i]), (y[i]-8, x[i]+4), c=color[i], size=10)
plt.annotate('Mare', (y[i]-8, x[i]+7), c=color[i], size=10)
i =1
plt.scatter(y[i], x[i], c=color[i], marker='^')
plt.annotate(str(name[i]), (y[i]-10, x[i]-11), c=color[i], size=10)
plt.annotate('Mare', (y[i]-10, x[i]-8), c=color[i], size=10)

# Craters
lat = np.array([-35,  -44])
long = np.array([ 208, 305])
name = np.array([ 'Apollo', 'Schickard'])
color = [ 'darkorchid', 'darkturquoise', ]
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
i=0
plt.scatter(y[i], x[i], c=color[i], marker='.', s=50)
plt.annotate(str(name[i]), (y[i]-3, x[i]-5), c=color[i], size=10)
plt.annotate('Crater', (y[i]-3, x[i]-8), c=color[i], size=10)
i=1
plt.scatter(y[i], x[i], c=color[i], marker='.', s=50)
plt.annotate(str(name[i]), (y[i]+5, x[i]-16), c=color[i], size=10)
plt.annotate('Crater', (y[i]+5, x[i]-20), c=color[i], size=10)

# SP
lat = np.array([ -89])
long = np.array([  2]) 
name = np.array([ 'South Pole'])
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
plt.scatter(y, x, c='limegreen', marker='.', vmin=0, vmax=10)
plt.annotate('South', (y[i]-10, x[i]+2), c='limegreen', size=10)
plt.annotate('Pole', (y[i]-10, x[i]-2), c='limegreen', size=10)
plt.imshow(im[wvl, :, :], vmin=-1 , vmax =25, cmap='Greys_r')
plt.gca().invert_yaxis()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

latlong = d35201_latlong
im = d35201_im
th = 2
plt.figure(figsize=(5,5 ), dpi=300)
# Mare
lat = np.array([ -23.5, -21.3, -47])
long = np.array([322.5, 343.4, 92])
name = np.array([ 'Humorum', 'Nubium', 'Australe'])
color = [ 'orange', 'green', 'magenta']
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
for i in range(0, np.size(lat)):
    plt.scatter(y[i], x[i], c=color[i], marker='^')
    plt.annotate(str(name[i]), (y[i]-15, x[i]+3), c=color[i], size=10)
    plt.annotate('Mare', (y[i]-15, x[i]+7), c=color[i], size=10)

# Craters
lat = np.array([-35,  -44])
long = np.array([ 208, 305])
name = np.array([ 'Apollo', 'Schickard'])
color = ['darkorchid', 'darkturquoise', ]
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
i=0
plt.scatter(y[i], x[i], c=color[i], marker='.', s=50)
plt.annotate(str(name[i]), (y[i]-5, x[i]-6), c=color[i], size=10)
plt.annotate('Crater', (y[i]-5, x[i]-10), c=color[i], size=10)
i=1
plt.scatter(y[i], x[i], c=color[i], marker='.', s=50)
plt.annotate(str(name[i]), (y[i]+5, x[i]-16), c=color[i], size=10)
plt.annotate('Crater', (y[i]+5, x[i]-20), c=color[i], size=10)

# SP
lat = np.array([ -89])
long = np.array([  6]) 
name = np.array([ 'South Pole'])
locxy = np.zeros((np.size(lat), 2))
for i in range(0, np.size(lat)):
    locxy[i, :] = np.nanmean(np.argwhere((np.isclose(latlong[0, :, :],long[i], atol=th) & np.isclose(latlong[1, :, :],lat[i], atol=th))), 0)
x = locxy[:, 0]
y = locxy[:, 1]
plt.scatter(y, x, c='limegreen', marker='.', vmin=0, vmax=10)
plt.annotate('South', (y[i], x[i]+7), c='limegreen', size=10)
plt.annotate('Pole', (y[i]+13, x[i]+7), c='limegreen', size=10)
plt.imshow(im[wvl, :, :], vmin=-1 , vmax =25, cmap='Greys_r')
plt.gca().invert_yaxis()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

#%% Figure 3; The original figure. 
im = d33901_im
wave = d33901_wave
cosi = np.cos(np.radians(d33901_inc[0]))
app_ref = d33901_app_ref
hibd = d33901_hibd

y = 65
x = 25
app_reflectance, thermal_part_4, new_scat, modtran, cosi = model_fitting_14(wave, im, modtran, cosi, y, x, ysize, True)

fit_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.59, wave[:, y, x] <= 2.61))))
fit_b = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
fit_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.59, wave[:, y, x] <= 3.61))))
fit_d = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.69, wave[:, y, x] <= 3.71))))
wave_app = np.concatenate([wave[fit_a:fit_b, y, x], wave[fit_c:fit_d, y, x]])
spec_app = np.concatenate([app_ref[fit_a:fit_b, y, x], app_ref[fit_c:fit_d, y, x]])
index_app = ~(np.isnan(wave_app) | np.isnan(spec_app))

min_bd = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.59, wave[:, y, x] <= 2.61))))
max_bd = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 3.64, wave[:, y, x] <= 3.66))))

min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

# Least squares fit for linear relationship of apparent reflectance
opt, cov = curve_fit(scatter, wave_app[index_app], spec_app[index_app])
app_fit = scatter(wave[:, y, x], opt[0], opt[1]) 

res_app_ref = np.zeros(512)*np.nan
for i in range(512):
    res_app_ref[i] = resmean(app_ref[i-2:i+3, y-2:y+2, x-2:x+2], 2)[0]

fig = plt.figure(constrained_layout=True, figsize=(10, 8), dpi=500)
ax_dict = fig.subplot_mosaic(
    """
    AA
    BC
    """
    )
ax_dict['A'].plot(wave[:, y, x], im[:, y, x], c='blue', label='Calibrated')
ax_dict['A'].plot(wave[:, y, x], thermal_part_4 +new_scat*modtran[:, y, x]*cosi/np.pi, c='red', label="Model")
ax_dict['A'].plot(wave[:, y, x], thermal_part_4,  c='orange', linestyle='dashed', label='Thermal Model')
ax_dict['A'].plot(wave[:, y, x], new_scat*modtran[:, y, x]*cosi/np.pi, c='green', linestyle='dashed', label='Scattered model')
ax_dict['A'].fill_between([wave[min_a, y, x], wave[max_a, y, x]], 0, 12, color='grey', alpha=0.3, label='Fit Regions')
ax_dict['A'].fill_between([wave[min_b, y, x], wave[max_b, y, x]], 0, 12, color='grey', alpha=0.3)
ax_dict['A'].fill_between([wave[min_c, y, x], wave[max_c, y, x]], 0, 12, color='grey', alpha=0.3)
ax_dict['A'].set_ylim((0, 10))
ax_dict['A'].set_ylabel(r'Radiance (W/$m^2/ \mu$m)') # these are whatever uit is output into the calibrated data. 
ax_dict['A'].set_xlim((1.3, 4.5))
ax_dict['A'].set_xlabel(r'Wavelength ($\mu$m)')
ax_dict['A'].legend(loc='upper center')
ax_dict['A'].annotate('a', (1.35, 9.3), fontweight='bold')

ax_dict['B'].plot(wave[:, y, x], app_fit, c = 'blue', label='Best fit')
ax_dict['B'].plot(wave[:, y, x], res_app_ref[:], c='k', label='Data')
ax_dict['B'].fill_between(wave[ fit_a:fit_b , y, x], 0, 1, color='red', alpha=0.5, label='Fit Regions')
ax_dict['B'].fill_between(wave[ fit_c:fit_d , y, x], 0, 1, color='red', alpha=0.5)
ax_dict['B'].set_ylim((0, 0.5))
ax_dict['B'].set_xlim((2.5, 3.8))
ax_dict['B'].set_ylabel(r'Apparent Reflectance')
ax_dict['B'].set_xlabel(r'Wavelength $(\mu m)$')
ax_dict['B'].legend(loc='lower center')
ax_dict['B'].annotate('b', (2.53, 0.47), fontweight='bold')

ax_dict['C'].plot(wave[:, y, x], res_app_ref/app_fit, c='k', label='Data')
ax_dict['C'].fill_between(wave[:, y, x], res_app_ref/app_fit, 1, color='red', alpha=0.2, label='Integrated Region')
ax_dict['C'].set_ylabel('Continuum Removed Reflectance')
ax_dict['C'].set_xlabel(r'Wavelength($\mu m$)')
ax_dict['C'].set_ylim((0.85, 1))
ax_dict['C'].set_xlim((2.6, 3.65))
ax_dict['C'].set_xlim((wave[min_bd, y, x], wave[max_bd, y, x]))
ax_dict['C'].legend()
ax_dict['C'].annotate('c', (2.61, 0.99), fontweight='bold')
plt.show()


#%% Figure 5 - reorder. 
# radiance, then apparent reflectance. morning evening noon different colors on plot. 
wvl = 19
combinedim = np.vstack(( d35201_im[wvl, :, :], d34601_im[wvl, :, :],d33901_im[wvl, :, :]))
kk = 44
y339 = repeats[0, kk]
x339 = repeats[1, kk]
y346 = repeats[2, kk]
x346 = repeats[3, kk]
y352 = repeats[4, kk]
x352 = repeats[5, kk]
cmap = matplotlib.cm.Greys_r
cmap.set_bad('k')
fig = plt.figure(constrained_layout=True, figsize=(8, 7), dpi=res)
ax_dict = fig.subplot_mosaic(
    """
    BBC
    AAC
    """
    )
    #[ ['rad'],['ref', 'rem']])
ax_dict['A'].plot(d33901_wave[:, y339, x339], d33901_app_ref[:, y339, x339], c='blue', label='%2.1f hr'%d33901_tod[0, y339, x339])
ax_dict['A'].plot(d34601_wave[:, y346, x346], d34601_app_ref[:, y346, x346], c='red', label='%2.1f hr'%d34601_tod[0, y346, x346])
ax_dict['A'].plot(d35201_wave[:, y352, x352], d35201_app_ref[:, y352, x352],  c='orange', label='%2.1f hr'%d35201_tod[0, y352, x352])
ax_dict['A'].set_ylim((0, 0.5))
ax_dict['A'].set_ylabel(r'Apparent Reflectance') # these are whatever uit is output into the calibrated data. 
ax_dict['A'].set_xlim((1.3, 4.5))
ax_dict['A'].legend(loc='lower center')
ax_dict['A'].set_xlabel(r'Wavelength ($\mu$m)')
ax_dict['A'].annotate('b', (1.35, 0.45), fontweight='bold')
ax_dict['A'].minorticks_on()
ax_dict['A'].annotate(('Latitude: %2.1f, Longitude: %2.1f'%(d33901_latlong[1,y339,x339], d33901_latlong[0, y339, x339])), (1.7, 0.45))

ax_dict['B'].plot(d33901_wave[:, y339, x339], d33901_im[:, y339, x339], c='blue',label='%3.0f'%d33901_parameters[0, y339, x339])
ax_dict['B'].plot(d34601_wave[:, y346, x346], d34601_im[:, y346, x346], c='red', label='%3.0f'%d34601_parameters[0, y346, x346])
ax_dict['B'].plot(d35201_wave[:, y352, x352], d35201_im[:, y352, x352],  c='orange', label='%3.0f'%d35201_parameters[0, y352, x352])
ax_dict['B'].set_ylim((0., 10))
ax_dict['B'].set_xlim((1.3, 4.5))
ax_dict['B'].set_ylabel(r'Radiance (W/$m^2/ \mu m$)')
ax_dict['B'].set_xlabel(r'Wavelength $(\mu m)$')
ax_dict['B'].legend(loc='upper right', title='Temp (K)')
ax_dict['B'].annotate('a', (1.4, 9.2), fontweight='bold')
ax_dict['B'].minorticks_on()

ax_dict['C'].imshow(combinedim, cmap='Greys_r', vmin=0, vmax=22)
ax_dict['C'].scatter(x339, 235+y339, c='cornflowerblue', s=5)
ax_dict['C'].scatter(x346, 135+y346, c='red', s=5)
ax_dict['C'].scatter(x352, y352+0, c='orange', s=5)
ax_dict['C'].invert_yaxis()
ax_dict['C'].annotate('Evening', (x352, y352-32), c='orange')
ax_dict['C'].annotate('Noon', (x346-30, y346+135+35), c='r')
ax_dict['C'].annotate('Morning', (x339-10, y339+14+235), c='cornflowerblue')
ax_dict['C'].axis('off')
ax_dict['C'].annotate('c', (3, 325), fontweight='bold', c='w')
plt.tight_layout()

#%% Figure 6
#01
valida = np.logical_and(~np.isnan(d33901_parameters[0, :, :]), d33901_pibd[0, :, :] >=0.025)
validb = np.logical_and(~np.isnan(d34601_parameters[0, :, :]), d34601_pibd[0, :, :] >=0.025)
validc = np.logical_and(~np.isnan(d35201_parameters[0, :, :]), d35201_pibd[0, :, :] >=0.025)

all_ibd_m01 = 100*np.concatenate((d33901_hibd[0, :, :][valida], d34601_hibd[0, :, :][validb], d35201_hibd[0, :, :][validc]))
all_temp_m01 = np.concatenate((d33901_parameters[0, :, :][valida], d34601_parameters[0, :, :][validb], d35201_parameters[0, :,: ][validc]))

valida = np.logical_and(~np.isnan(d33901_parameters[0, :, :]), d33901_pibd[0, :, :] <0.02)
validb = np.logical_and(~np.isnan(d34601_parameters[0, :, :]), d34601_pibd[0, :, :] <0.02)
validc = np.logical_and(~np.isnan(d35201_parameters[0, :, :]), d35201_pibd[0, :, :] <0.02)

all_ibd_h01 = 100*np.concatenate((d33901_hibd[0, :, :][valida], d34601_hibd[0, :, :][validb], d35201_hibd[0, :, :][validc]))
all_temp_h01 = np.concatenate((d33901_parameters[0, :, :][valida], d34601_parameters[0, :, :][validb], d35201_parameters[0, :,: ][validc]))

#00
valida = np.logical_and(~np.isnan(d33900_parameters[0, :, :]), d33900_pibd[0, :, :] >=0.025)
validb = np.logical_and(~np.isnan(d34600_parameters[0, :, :]), d34600_pibd[0, :, :] >=0.025)
validc = np.logical_and(~np.isnan(d35200_parameters[0, :, :]), d35200_pibd[0, :, :] >=0.025)

all_ibd_m00 = 100*np.concatenate((d33900_hibd[0, :, :][valida], d34600_hibd[0, :, :][validb], d35200_hibd[0, :, :][validc]))
all_temp_m00 = np.concatenate((d33900_parameters[0, :, :][valida], d34600_parameters[0, :, :][validb], d35200_parameters[0, :,: ][validc]))

valida = np.logical_and(~np.isnan(d33900_parameters[0, :, :]), d33900_pibd[0, :, :] <0.02)
validb = np.logical_and(~np.isnan(d34600_parameters[0, :, :]), d34600_pibd[0, :, :] <0.02)
validc = np.logical_and(~np.isnan(d35200_parameters[0, :, :]), d35200_pibd[0, :, :] <0.02)

all_ibd_h00 = 100*np.concatenate((d33900_hibd[0, :, :][valida], d34600_hibd[0, :, :][validb], d35200_hibd[0, :, :][validc]))
all_temp_h00 = np.concatenate((d33900_parameters[0, :, :][valida], d34600_parameters[0, :, :][validb], d35200_parameters[0, :,: ][validc]))

#02
valida = np.logical_and(~np.isnan(d33902_parameters[0, :, :]), d33902_pibd[0, :, :] >=0.025)
validb = np.logical_and(~np.isnan(d34602_parameters[0, :, :]), d34602_pibd[0, :, :] >=0.025)
#validc = np.logical_and(~np.isnan(d35202_parameters[0, :, :]), d35202_pibd[0, :, :] >=0.025)

all_ibd_m02 = 100*np.concatenate((d33902_hibd[0, :, :][valida], d34602_hibd[0, :, :][validb]))#, d35202_hibd[0, :, :][validc]))
all_temp_m02 = np.concatenate((d33902_parameters[0, :, :][valida], d34602_parameters[0, :, :][validb]))#, d35202_parameters[0, :,: ][validc]))

valida = np.logical_and(~np.isnan(d33902_parameters[0, :, :]), d33902_pibd[0, :, :] <0.02)
validb = np.logical_and(~np.isnan(d34602_parameters[0, :, :]), d34602_pibd[0, :, :] <0.02)
validc = np.logical_and(~np.isnan(d35202_parameters[0, :, :]), d35202_pibd[0, :, :] <0.02)

all_ibd_h02 = 100*np.concatenate((d33902_hibd[0, :, :][valida], d34602_hibd[0, :, :][validb]))#, d35202_hibd[0, :, :][validc]))
all_temp_h02 = np.concatenate((d33902_parameters[0, :, :][valida], d34602_parameters[0, :, :][validb]))#, d35202_parameters[0, :,: ][validc]))

all_ibd_m = np.concatenate((all_ibd_m00, all_ibd_m01, all_ibd_m02))
all_temp_m = np.concatenate((all_temp_m00, all_temp_m01, all_temp_m02))

all_ibd_h = np.concatenate((all_ibd_h00, all_ibd_h01, all_ibd_h02))
all_temp_h = np.concatenate((all_temp_h00, all_temp_h01, all_temp_h02))

st = 240
st = np.nanmin(all_temp_m)
end = 390
end = np.nanmax(all_temp_h)
binsize = 5
binnum = np.int((end-st)/binsize)

# Side by side temperature plots
fig = plt.figure(constrained_layout=True, figsize=(16, 6), dpi=res)
ax_dict = fig.subplot_mosaic(
    """
    AB
    """
    )
ax_dict['A'].plot(all_temp_m, all_ibd_m, '.', c='blue', alpha=0.5, label='Maria')
ax_dict['A'].plot(all_temp_h, all_ibd_h, '.', c='red', alpha=0.5, label='Highlands')
ax_dict['A'].legend()
ax_dict['A'].set_ylim((0, 8))
ax_dict['A'].set_ylabel('IBD (%)')
ax_dict['A'].set_xlim((240, 400))
ax_dict['A'].set_xlabel('Temperature (K)')
ax_dict['A'].annotate('a', (242, 7.5), size=25, weight='bold')
bin_means, bin_edges, binnumber = stats.binned_statistic(all_temp_m,
                all_ibd_m, statistic='mean', bins=binnum)
bin_meansst, bin_edgesst, binnumberst = stats.binned_statistic(all_temp_m,
                all_ibd_m, statistic='std', bins=binnum)
ax_dict['B'].errorbar((bin_edges[:-1]+bin_edges[1:])/2, bin_means, yerr = bin_meansst, fmt=" ", ecolor='darkblue', capsize=5)
ax_dict['B'].hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='blue', lw=5,
           label='Maria')
bin_means, bin_edges, binnumber = stats.binned_statistic(all_temp_h,
                all_ibd_h, statistic='mean', bins=binnum)
ax_dict['B'].hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='red', lw=5, alpha=0.8,
           label='Highlands')
bin_meansst, bin_edgesst, binnumberst = stats.binned_statistic(all_temp_h,
                all_ibd_h, statistic='std', bins=binnum)
ax_dict['B'].errorbar((bin_edges[:-1]+bin_edges[1:])/2, bin_means, yerr = bin_meansst, fmt=' ', ecolor='darkred', capsize=5)
ax_dict['B'].set_ylim((0, 8))
ax_dict['B'].set_ylabel('IBD (%)')
ax_dict['B'].set_xlim((240, 400))
ax_dict['B'].set_xlabel('Temperature (K)')
ax_dict['B'].legend(title='Bins of 5 K')
ax_dict['B'].annotate('b', (242, 7.5), size=25, weight='bold')

plt.show()

#%% Figure 7
#01
valida = np.logical_and(~np.isnan(d33901_inc[0, :, :]), d33901_pibd[0, :, :] >=0.025)
validb = np.logical_and(~np.isnan(d34601_inc[0, :, :]), d34601_pibd[0, :, :] >=0.025)
validc = np.logical_and(~np.isnan(d35201_inc[0, :, :]), d35201_pibd[0, :, :] >=0.025)

all_ibd_m01 = 100*np.concatenate((d33901_hibd[0, :, :][valida], d34601_hibd[0, :, :][validb], d35201_hibd[0, :, :][validc]))
all_inc_m01 = np.concatenate((d33901_inc[0, :, :][valida], d34601_inc[0, :, :][validb], d35201_inc[0, :,: ][validc]))

valida = np.logical_and(~np.isnan(d33901_inc[0, :, :]), d33901_pibd[0, :, :] <0.02)
validb = np.logical_and(~np.isnan(d34601_inc[0, :, :]), d34601_pibd[0, :, :] <0.02)
validc = np.logical_and(~np.isnan(d35201_inc[0, :, :]), d35201_pibd[0, :, :] <0.02)

all_ibd_h01 = 100*np.concatenate((d33901_hibd[0, :, :][valida], d34601_hibd[0, :, :][validb], d35201_hibd[0, :, :][validc]))
all_inc_h01 = np.concatenate((d33901_inc[0, :, :][valida], d34601_inc[0, :, :][validb], d35201_inc[0, :,: ][validc]))

#00
valida = np.logical_and(~np.isnan(d33900_inc[0, :, :]), d33900_pibd[0, :, :] >=0.025)
validb = np.logical_and(~np.isnan(d34600_inc[0, :, :]), d34600_pibd[0, :, :] >=0.025)
validc = np.logical_and(~np.isnan(d35200_inc[0, :, :]), d35200_pibd[0, :, :] >=0.025)

all_ibd_m00 = 100*np.concatenate((d33900_hibd[0, :, :][valida], d34600_hibd[0, :, :][validb], d35200_hibd[0, :, :][validc]))
all_inc_m00 = np.concatenate((d33900_inc[0, :, :][valida], d34600_inc[0, :, :][validb], d35200_inc[0, :,: ][validc]))

valida = np.logical_and(~np.isnan(d33900_inc[0, :, :]), d33900_pibd[0, :, :] <0.02)
validb = np.logical_and(~np.isnan(d34600_inc[0, :, :]), d34600_pibd[0, :, :] <0.02)
validc = np.logical_and(~np.isnan(d35200_inc[0, :, :]), d35200_pibd[0, :, :] <0.02)

all_ibd_h00 = 100*np.concatenate((d33900_hibd[0, :, :][valida], d34600_hibd[0, :, :][validb], d35200_hibd[0, :, :][validc]))
all_inc_h00 = np.concatenate((d33900_inc[0, :, :][valida], d34600_inc[0, :, :][validb], d35200_inc[0, :,: ][validc]))

#02
valida = np.logical_and(~np.isnan(d33902_inc[0, :, :]), d33902_pibd[0, :, :] >=0.025)
validb = np.logical_and(~np.isnan(d34602_inc[0, :, :]), d34602_pibd[0, :, :] >=0.025)
#validc = np.logical_and(~np.isnan(d35202_inc[0, :, :]), d35202_pibd[0, :, :] >=0.025)
all_ibd_m02 = 100*np.concatenate((d33902_hibd[0, :, :][valida], d34602_hibd[0, :, :][validb]))#, d35202_hibd[0, :, :][validc]))
all_inc_m02 = np.concatenate((d33902_inc[0, :, :][valida], d34602_inc[0, :, :][validb]))#, d35202_inc[0, :,: ][validc]))

valida = np.logical_and(~np.isnan(d33902_inc[0, :, :]), d33902_pibd[0, :, :] <0.02)
validb = np.logical_and(~np.isnan(d34602_inc[0, :, :]), d34602_pibd[0, :, :] <0.02)
validc = np.logical_and(~np.isnan(d35202_inc[0, :, :]), d35202_pibd[0, :, :] <0.02)

all_ibd_h02 = 100*np.concatenate((d33902_hibd[0, :, :][valida], d34602_hibd[0, :, :][validb]))#, d35202_hibd[0, :, :][validc]))
all_inc_h02 = np.concatenate((d33902_inc[0, :, :][valida], d34602_inc[0, :, :][validb]))#, d35202_inc[0, :,: ][validc]))
all_ibd_m = np.concatenate((all_ibd_m00, all_ibd_m01, all_ibd_m02))
all_inc_m = np.concatenate((all_inc_m00, all_inc_m01, all_inc_m02))
all_ibd_h = np.concatenate((all_ibd_h00, all_ibd_h01, all_ibd_h02))
all_inc_h = np.concatenate((all_inc_h00, all_inc_h01, all_inc_h02))
st = np.nanmin(all_inc_m)
end = np.nanmax(all_inc_h)
binsize = 5
binnum = np.int((end-st)/binsize)

# Side by side Incidence plots
fig = plt.figure(constrained_layout=True, figsize=(16, 6), dpi=res)
ax_dict = fig.subplot_mosaic(
"""
AB
"""
)
ax_dict['A'].plot(all_inc_m, all_ibd_m, '.', c='blue', alpha=0.5, label='Maria')
ax_dict['A'].plot(all_inc_h, all_ibd_h, '.', c='red', alpha=0.5, label='Highlands')
ax_dict['A'].legend()
ax_dict['A'].set_ylim((0, 8))
ax_dict['A'].set_ylabel('IBD (%)')
ax_dict['A'].set_xlim((35, 90))
ax_dict['A'].set_xlabel(r'Incidence angle ($^\circ$)')
ax_dict['A'].annotate('a', (85, 7.5), size=25, weight='bold')
bin_means, bin_edges, binnumber = stats.binned_statistic(all_inc_m,
all_ibd_m, statistic='mean', bins=binnum)
bin_meansst, bin_edgesst, binnumberst = stats.binned_statistic(all_inc_m,
all_ibd_m, statistic='std', bins=binnum)
ax_dict['B'].errorbar((bin_edges[:-1]+bin_edges[1:])/2, bin_means, yerr = bin_meansst, fmt=" ", ecolor='darkblue', capsize=5)
ax_dict['B'].hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='blue', lw=5,
label='Maria')
bin_means, bin_edges, binnumber = stats.binned_statistic(all_inc_h,
all_ibd_h, statistic='mean', bins=binnum)
ax_dict['B'].hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='red', lw=5, alpha=0.8,
label='Highlands')
bin_meansst, bin_edgesst, binnumberst = stats.binned_statistic(all_inc_h,
all_ibd_h, statistic='std', bins=binnum)
ax_dict['B'].errorbar((bin_edges[:-1]+bin_edges[1:])/2, bin_means, yerr = bin_meansst, fmt=' ', ecolor='darkred', capsize=5)
ax_dict['B'].set_ylim((0, 8))
ax_dict['B'].set_ylabel('IBD (%)')
ax_dict['B'].set_xlim((35, 90))
ax_dict['B'].set_xlabel(r'Incidence angle ($^\circ$)')
ax_dict['B'].legend(title='Bins of 5$^\circ$')
ax_dict['B'].annotate('b', (85, 7.5), size=25, weight='bold')
plt.show()

#%% Figure 8
def breakdown3d( ysize, xsize,amin, amax):
    i = 0
    night = np.zeros((512, 5000))*np.nan
    wnight = np.zeros((512, 5000))*np.nan
    for y in range(ysize):
        for x in range(xsize):
            if ~np.isnan(d33901_parameters[0, y, x]):
                if d33901_parameters[0, y, x] > amin and d33901_parameters[0, y, x] < amax and d33901_pibd[0, y, x] >=0.025: # and ~np.isnan(d33901_parameters[0, y,x+2]):
                    night[:,i] = 1-d33901_bd_new_resmean[:, y, x]
                    wnight[:, i] = d33901_wave[:,y , x]
                    i +=1
            if ~np.isnan(d34601_parameters[0, y, x]):
                if d34601_parameters[0, y, x] > amin and d34601_parameters[0, y, x] < amax and d34601_pibd[0, y, x] >=0.025: # and ~np.isnan(d34601_parameters[0, y,x+2]):
                    night[:,i] = 1-d34601_bd_new_resmean[:, y, x]
                    wnight[:, i] = d34601_wave[:,y , x]
                    i +=1
    for y in range(135):
        for x in range(xsize):
            if ~np.isnan(d35201_parameters[0, y, x]):
                if d35201_parameters[0, y, x] > amin and d35201_parameters[0, y, x] < amax and d35201_pibd[0, y, x] >=0.025: # and ~np.isnan(d35201_parameters[0, y,x+2]):
                    night[:,i] = 1-d35201_bd_new_resmean[:, y, x]
                    wnight[:, i] = d35201_wave[:,y , x]
                    i +=1       
    for y in range(ysize):
        for x in range(xsize):
            if ~np.isnan(d33900_parameters[0, y, x]):
                if d33900_parameters[0, y, x] > amin and d33900_parameters[0, y, x] < amax and d33900_pibd[0, y, x] >=0.025: # and ~np.isnan(d33900_parameters[0, y,x+2]):
                    night[:,i] = 1-d33900_bd_new_resmean[:, y, x]
                    wnight[:, i] = d33900_wave[:,y , x]
                    i +=1
            if ~np.isnan(d34600_parameters[0, y, x]):
                if d34600_parameters[0, y, x] > amin and d34600_parameters[0, y, x] < amax and d34600_pibd[0, y, x] >=0.025: # and ~np.isnan(d34600_parameters[0, y,x+2]):
                    night[:,i] = 1-d34600_bd_new_resmean[:, y, x]
                    wnight[:, i] = d34600_wave[:,y , x]
                    i +=1
    for y in range(135):
        for x in range(xsize):
            if ~np.isnan(d35200_parameters[0, y, x]):
                if d35200_parameters[0, y, x] > amin and d35200_parameters[0, y, x] < amax and d35200_pibd[0, y, x] >=0.025: # and ~np.isnan(d35200_parameters[0, y,x+2]):
                    night[:,i] = 1-d35200_bd_new_resmean[:, y, x]
                    wnight[:, i] = d35200_wave[:,y , x]
                    i +=1      
    for y in range(ysize):
        for x in range(xsize):
            if ~np.isnan(d33902_parameters[0, y, x]):
                if d33902_parameters[0, y, x] > amin and d33902_parameters[0, y, x] < amax and d33902_pibd[0, y, x] >=0.025: # and ~np.isnan(d33901_parameters[0, y,x+2]):
                    night[:,i] = 1-d33902_bd_new_resmean[:, y, x]
                    wnight[:, i] = d33902_wave[:,y , x]
                    i +=1
            if ~np.isnan(d34602_parameters[0, y, x]):
                if d34602_parameters[0, y, x] > amin and d34602_parameters[0, y, x] < amax and d34602_pibd[0, y, x] >=0.025: # and ~np.isnan(d34601_parameters[0, y,x+2]):
                    #night[:,i] = 1-d34602_bd_new_resmean[:, y, x]
                    #wnight[:, i] = d34602_wave[:,y , x]
                    i +=1
    for y in range(135):
        for x in range(xsize):
            if ~np.isnan(d35202_parameters[0, y, x]):
                if d35202_parameters[0, y, x] > amin and d35202_parameters[0, y, x] < amax and d35202_pibd[0, y, x] >=0.025: # and ~np.isnan(d35201_parameters[0, y,x+2]):
                    #night[:,i] = 1-d35202_bd_new_resmean[:, y, x]
                    #wnight[:, i] = d35202_wave[:,y , x]
                    i +=1       
    out = np.zeros((512))*np.nan
    owave = np.zeros((512))*np.nan
    for i in range(512):
        out[i] = resmean(night[i, :], 2)[0] 
        owave[i]= resmean(wnight[i, :], 2)[0]
    return owave, out

def breakdown3dh( ysize, xsize,amin, amax):
    i = 0
    night = np.zeros((512, 50000))*np.nan
    wnight = np.zeros((512, 50000))*np.nan
    for y in range(ysize):
        for x in range(xsize):
            if ~np.isnan(d33901_parameters[0, y, x]):
                if d33901_parameters[0, y, x] > amin and d33901_parameters[0, y, x] < amax and d33901_pibd[0, y, x] <0.025: # and ~np.isnan(d33901_parameters[0, y,x+2]):
                    night[:,i] = 1-d33901_bd_new_resmean[:, y, x]
                    wnight[:, i] = d33901_wave[:,y , x]
                    i +=1
            if ~np.isnan(d34601_parameters[0, y, x]):
                if d34601_parameters[0, y, x] > amin and d34601_parameters[0, y, x] < amax and d34601_pibd[0, y, x] <0.025: # and ~np.isnan(d34601_parameters[0, y,x+2]):
                    night[:,i] = 1-d34601_bd_new_resmean[:, y, x]
                    wnight[:, i] = d34601_wave[:,y , x]
                    i +=1
    for y in range(135):
        for x in range(xsize):
            if ~np.isnan(d35201_parameters[0, y, x]):
                if d35201_parameters[0, y, x] > amin and d35201_parameters[0, y, x] < amax and d35201_pibd[0, y, x] <0.025: # and ~np.isnan(d35201_parameters[0, y,x+2]):
                    night[:,i] = 1-d35201_bd_new_resmean[:, y, x]
                    wnight[:, i] = d35201_wave[:,y , x]
                    i +=1       
    for y in range(ysize):
        for x in range(xsize):
            if ~np.isnan(d33900_parameters[0, y, x]):
                if d33900_parameters[0, y, x] > amin and d33900_parameters[0, y, x] < amax and d33900_pibd[0, y, x] <0.025: # and ~np.isnan(d33900_parameters[0, y,x+2]):
                    night[:,i] = 1-d33900_bd_new_resmean[:, y, x]
                    wnight[:, i] = d33900_wave[:,y , x]
                    i +=1
            if ~np.isnan(d34600_parameters[0, y, x]):
                if d34600_parameters[0, y, x] > amin and d34600_parameters[0, y, x] < amax and d34600_pibd[0, y, x] <0.025: # and ~np.isnan(d34600_parameters[0, y,x+2]):
                    night[:,i] = 1-d34600_bd_new_resmean[:, y, x]
                    wnight[:, i] = d34600_wave[:,y , x]
                    i +=1
    for y in range(135):
        for x in range(xsize):
            if ~np.isnan(d35200_parameters[0, y, x]):
                if d35200_parameters[0, y, x] > amin and d35200_parameters[0, y, x] < amax and d35200_pibd[0, y, x] <0.025: # and ~np.isnan(d35200_parameters[0, y,x+2]):
                    night[:,i] = 1-d35200_bd_new_resmean[:, y, x]
                    wnight[:, i] = d35200_wave[:,y , x]
                    i +=1      
    for y in range(ysize):
        for x in range(xsize):
            if ~np.isnan(d33902_parameters[0, y, x]):
                if d33902_parameters[0, y, x] > amin and d33902_parameters[0, y, x] < amax and d33902_pibd[0, y, x] <0.025: # and ~np.isnan(d33900_parameters[0, y,x+2]):
                    night[:,i] = 1-d33902_bd_new_resmean[:, y, x]
                    wnight[:, i] = d33902_wave[:,y , x]
                    i +=1
            if ~np.isnan(d34602_parameters[0, y, x]):
                if d34602_parameters[0, y, x] > amin and d34602_parameters[0, y, x] < amax and d34602_pibd[0, y, x] <0.025: # and ~np.isnan(d34601_parameters[0, y,x+2]):
                    night[:,i] = 1-d34602_bd_new_resmean[:, y, x]
                    wnight[:, i] = d34602_wave[:,y , x]
                    i +=1 
    for y in range(135):
        for x in range(xsize):
            if ~np.isnan(d35202_parameters[0, y, x]):
                if d35202_parameters[0, y, x] > amin and d35202_parameters[0, y, x] < amax and d35202_pibd[0, y, x] <0.025: # and ~np.isnan(d35200_parameters[0, y,x+2]):
                    night[:,i] = 1-d35202_bd_new_resmean[:, y, x]
                    wnight[:, i] = d35201_wave[:,y , x]
                    i +=1 
    out = np.nanmean(night, axis=1)
    owave = np.nanmean(wnight, axis=1)
    print(amin, amax, i)
    out = np.zeros((512))*np.nan
    owave = np.zeros((512))*np.nan
    for i in range(512):
        out[i] = resmean(night[i, :], 2)[0] 
        owave[i]= resmean(wnight[i, :], 2)[0]
    return owave, out

#Plot all the temperature slices
step = 10
labels = ['%2.0f - %2.0f' %(i, i+step) for i in range(250, 400, step)]
mare = np.zeros((512, np.size(labels)))*np.nan
mare_wave = np.zeros((512, np.size(labels)))*np.nan
high = np.zeros((512, np.size(labels)))*np.nan
high_wave = np.zeros((512, np.size(labels)))*np.nan
for i,j in zip(range(0, np.size(labels)), range(250, 400, step)):
    mare_wave[:, i],  mare[:, i] = breakdown3d(100, 64, j, j+step)
    high_wave[:, i], high[:, i] = breakdown3dh(100, 64, j, j+step)

# plot all
colors = mpl.cm.jet(np.linspace(0, 1, np.size(labels)))
plt.rc("font", size=17, family="serif")
fig, ax = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
ax[1].vlines(2.95, 0, 1.0, colors='k', linestyle='dashed')
ax[1].vlines(3.14, 0, 1.0, colors='k', linestyle='dashed')
ax[0].vlines(2.95, 0, 1.0, colors='k', linestyle='dashed')
ax[0].vlines(3.14, 0, 1.0, colors='k', linestyle='dashed')
for i in range(0, np.size(labels)):
    ax[0].plot(mare_wave[:, i], mare[:, i], color=colors[i],  label=labels[i])
    ax[1].plot(high_wave[:, i], high[:, i], color=colors[i], label=labels[i])
ax[0].set_ylim((0.86, 1.00))
ax[0].set_ylabel('Continuum removed reflectance')
ax[0].set_xlabel(r'Wavelength ($\mu$m)')
ax[0].set_xlim((2.65, 3.65))
ax[0].minorticks_on()
ax[0].set_title('Maria')

ax[1].legend(title='Temperature (K)', bbox_to_anchor=(1.0, 1), loc='upper left')
ax[1].set_ylim((0.86, 1.00))
ax[1].set_ylabel('Continuum removed reflectance')
ax[1].set_xlabel(r'Wavelength ($\mu$m)')
ax[1].set_xlim((2.65, 3.65))
ax[1].minorticks_on()
ax[1].set_title('Highlands')
plt.tight_layout()
plt.show()

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Highlands', markerfacecolor='red'),
                   Line2D([0], [0], marker='o', color='w', label='Maria',markerfacecolor='blue')]
tt = 11
ee = 1
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
plt.plot(mare_wave[:, tt], mare[:, tt], c='blue')
plt.plot(high_wave[:, tt], high[:, tt], c='red')
plt.plot(mare_wave[:, ee], mare[:, ee], c='blue', linestyle='dashed')
plt.plot(high_wave[:, ee], high[:, ee], c='red', linestyle='dashed')
plt.plot(mare_wave[:, tt], mare[:, tt], c='k', label=labels[tt], zorder=1)
plt.plot(high_wave[:, ee], high[:, ee], c='k', linestyle='dashed', label=labels[ee], zorder=1)
plt.vlines(2.95, 0, 1, color='k', linestyle='dashed')
plt.vlines(3.14, 0, 1, color='k', linestyle='dashed')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Continuum removed reflectance')
plt.ylim((0.85, 1.0))
plt.xlim((2.65, 3.65))
leg2 = ax.legend(title='Temp (K)', loc='lower left')
leg1 = ax.legend(handles=legend_elements, loc='lower right')
ax.add_artist(leg2)
plt.minorticks_on()
plt.show()

#%% Figure 9 Three plots, hot, medium, cold
hmin = 370
hmax = 380
mmin = 320
mmax = 330
cmin = 270
cmax = 280

marehotw, marehot = breakdown3d(100, 64, hmin, hmax)
highhotw, highhot = breakdown3dh(100, 64, hmin, hmax)

maremedw, maremed = breakdown3d(100, 64, mmin, mmax)
highmedw, highmed = breakdown3dh(100, 64, mmin, mmax)

marecoldw, marecold = breakdown3d(100, 64, cmin, cmax)
highcoldw, highcold = breakdown3dh(100, 64, cmin, cmax)

fig, ax = plt.subplots(1, 2, figsize=(18, 8), dpi=500)
ax[0].plot(highhotw, highhot, c='red', label='%3.0f < T < %3.0f K' %(hmin, hmax))
ax[0].plot(highmedw, highmed, c='green', label='%3.0f < T < %3.0f K' %(mmin, mmax))
ax[0].plot(highcoldw, highcold, c='blue', label='%3.0f < T < %3.0f K' %(cmin, cmax))
ax[0].set_title('Highland')
ax[0].set_ylim((0.85, 1.0))
ax[0].set_xlim((2.65, 3.65))
ax[0].set_xlabel('Wavelength ($\mu$m)')
ax[0].set_ylabel('Continuum Removed Reflectance')
ax[0].vlines(2.95, 0, 1.0, colors='k', linestyle='dashed')
ax[0].vlines(3.14, 0, 1.0, colors='k', linestyle='dashed')
ax[0].legend(loc ='lower right')

ax[1].plot(marehotw, marehot, c='red', label='%3.0f < T < %3.0f K' %(hmin, hmax))
ax[1].plot(maremedw, maremed, c='green', label='%3.0f < T < %3.0f K' %(mmin, mmax))
ax[1].plot(marecoldw, marecold, c='blue', label='%3.0f < T < %3.0f K' %(cmin, cmax))
ax[1].set_title('Mare')
ax[1].set_ylim((0.85, 1.0))
ax[1].set_xlim((2.65, 3.65))
ax[1].set_xlabel('Wavelength ($\mu$m)')
ax[1].set_ylabel('Continuum Removed Reflectance')
ax[1].vlines(2.95, 0, 1.0, colors='k', linestyle='dashed')
ax[1].vlines(3.14, 0, 1.0, colors='k', linestyle='dashed')
ax[1].legend(loc='lower right')
plt.show()

def normalization(mina, maxa, st):
    if 'mare' in st:
        wavem, mare_check =  breakdown3d(100, 64, mina, maxa)
    elif 'highland' in st:
        wavem, mare_check =  breakdown3dh(100, 64, mina, maxa)
    else:
        print('Bad input')
    st = np.min(np.argwhere((np.logical_and(wavem >= 2.80, wavem <= 2.982))))
    en = np.max(np.argwhere((np.logical_and(wavem >= 3.4, wavem <= 3.42))))    
    minimum_wavelength = np.argwhere((mare_check)==np.nanmin(mare_check[st:en]))
    mare_check = 1-mare_check
    normal_base = mare_check / -mare_check[minimum_wavelength[0]]
    return wavem, 1+normal_base

highhotw, highhot = normalization(hmin, hmax, 'highland')
highmedw, highmed = normalization(mmin, mmax, 'highland')
highcoldw, highcold = normalization(cmin, cmax, 'highland')

marehotw, marehot = normalization(hmin, hmax, 'mare')
maremedw, maremed = normalization(mmin, mmax, 'mare')
marecoldw, marecold = normalization(cmin, cmax, 'mare')

fig, ax = plt.subplots(1, 2, figsize=(18, 8), dpi=500)
ax[0].plot(highhotw, highhot, c='red', label='%3.0f < T < %3.0f K' %(hmin, hmax))
ax[0].plot(highmedw, highmed, c='green', label='%3.0f < T < %3.0f K' %(mmin, mmax))
ax[0].plot(highcoldw, highcold, c='blue', label='%3.0f < T < %3.0f K' %(cmin, cmax))
ax[0].set_title('Highland')
ax[0].set_ylim((-0.02, 1.020))
ax[0].set_xlabel('Wavelength ($\mu$m)')
ax[0].set_ylabel('Normalized Reflectance')
ax[0].set_xlim((2.65, 3.65))
ax[0].vlines(2.95, -0.02, 1.02, colors='k', linestyle='dashed')
ax[0].vlines(3.14, -0.02, 1.02, colors='k', linestyle='dashed')
ax[0].legend()

ax[1].plot(marehotw, marehot, c='red', label='%3.0f < T < %3.0f K' %(hmin, hmax))
ax[1].plot(maremedw, maremed, c='green', label='%3.0f < T < %3.0f K' %(mmin, mmax))
ax[1].plot(marecoldw, marecold, c='blue', label='%3.0f < T < %3.0f K' %(cmin, cmax))
ax[1].set_title('Mare')
ax[1].set_ylim((-0.02, 1.02))
ax[1].set_xlim((2.65, 3.65))
ax[1].set_xlabel('Wavelength ($\mu$m)')
ax[1].set_ylabel('Normalized Reflectance')
ax[1].vlines(2.95, -0.02, 1.0200, colors='k', linestyle='dashed')
ax[1].vlines(3.14, -0.02, 1.020, colors='k', linestyle='dashed')
ax[1].legend(loc='lower right')
plt.show()
#%% Figure 10
# Making full data
plt.figure(figsize=(8, 7), dpi=res)
for j in range(0, 80):
    y = [repeats[0, j], repeats[2, j], repeats[4, j]]
    x = [repeats[1, j], repeats[3, j], repeats[5, j]]
    arr_ll = [d33901_tod[0, y[0], x[0]],d34601_tod[0, y[1], x[1]], d35201_tod[0, y[2], x[2]]]
    arr_hibd = [d33901_hibd[0, y[0], x[0]], d34601_hibd[0, y[1], x[1]], d35201_hibd[0, y[2], x[2] ]]
    plt.scatter(d33901_tod[0, y[0], x[0]], d33901_hibd[0, y[0], x[0]],  c=d33901_parameters[0, y[0], x[0]], cmap='coolwarm', s=40, zorder=10, vmin=260, vmax=390)
    plt.scatter(d34601_tod[0, y[1], x[1]], d34601_hibd[0, y[1], x[1]],  c=d34601_parameters[0, y[1], x[1]], cmap='coolwarm', s=40, zorder=10, vmin=260, vmax=390)
    plt.scatter(d35201_tod[0, y[2], x[2]], d35201_hibd[0, y[2], x[2]],  c=d35201_parameters[0, y[2], x[2]], cmap='coolwarm', s=40, zorder=10, vmin=260, vmax=390)
handle= [Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='Mare'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='g', label='Highland'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='Highland')]
plt.colorbar(label='Temperature (K)')
plt.annotate(r'Latitude: [%2.0f:%2.0f$^\circ$S], Longitude: [%2.0f:%2.0f$^\circ$]'%(-np.nanmax(d35201_latlong[1, repeats[4, 0:80], repeats[5, 0:80]]), -np.nanmin(d35201_latlong[1, repeats[4, 0:80], repeats[5, 0:80]]), np.nanmin(d35201_latlong[0, repeats[4, 0:80], repeats[5, 0:80]]), np.nanmax(d35201_latlong[0, repeats[4, 0:80], repeats[5, 0:80]])), (6, 0.032))
plt.ylim((0.03, 0.075))
plt.xlim((5.5, 18.4))
plt.ylabel('IBD')
plt.xlabel(r'TOD (hr)')
plt.show() 

# Making connected
plt.figure(figsize=(8, 7), dpi=res)

for j in range(0, 80):
    y = [repeats[0, j], repeats[2, j], repeats[4, j]]
    x = [repeats[1, j], repeats[3, j], repeats[5, j]]
    if d33901_latlong[1, y[0], x[0]] >= -60 and ~np.isnan(d33901_hibd[0, y[0], x[0]]) and 240 <= d33901_latlong[0, y[0], x[0]] <= 251:
        arr_ll = [d33901_tod[0, y[0], x[0]],d34601_tod[0, y[1], x[1]], d35201_tod[0, y[2], x[2]]]
        arr_hibd = [d33901_hibd[0, y[0], x[0]], d34601_hibd[0, y[1], x[1]], d35201_hibd[0, y[2], x[2] ]]
        plt.plot(arr_ll, arr_hibd, c='k', zorder=1)
        plt.scatter(d33901_tod[0, y[0], x[0]], d33901_hibd[0, y[0], x[0]],  c=d33901_parameters[0, y[0], x[0]], cmap='coolwarm', s=40, zorder=10, vmin=260, vmax=390)
        plt.scatter(d34601_tod[0, y[1], x[1]], d34601_hibd[0, y[1], x[1]],  c=d34601_parameters[0, y[1], x[1]], cmap='coolwarm', s=40, zorder=10, vmin=260, vmax=390)
        plt.scatter(d35201_tod[0, y[2], x[2]], d35201_hibd[0, y[2], x[2]],  c=d35201_parameters[0, y[2], x[2]], cmap='coolwarm', s=40, zorder=10, vmin=260, vmax=390)
        print('%3.0f'%d35201_parameters[0, y[2], x[2]], '%0.3f'%d35201_hibd[0, y[2], x[2]])
        print(d33901_latlong[:, y[0], x[0]])
handle= [Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='Mare'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='g', label='Highland'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='Highland')]
plt.colorbar(label='Temperature (K)')
plt.annotate(r'Latitude: [50:60$^\circ$S], Longitude: [240:250$^\circ$]', (6.0, 0.032)) #(6, 0.032)
plt.ylim((0.03, 0.075))
plt.xlim((5.5, 18.4))
plt.ylabel('IBD')
plt.xlabel(r'TOD (hr)')
plt.show() 

# Making images
fig = plt.figure(figsize=(8, 7), dpi=res)
plt.imshow(d33901_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
for j in range(0, 80):
    y = [repeats[0, j], repeats[2, j], repeats[4, j]]
    x = [repeats[1, j], repeats[3, j], repeats[5, j]]
    if d33901_latlong[1, y[0], x[0]] >= -60 and ~np.isnan(d33901_hibd[0, y[0], x[0]]) and 240 <= d33901_latlong[0, y[0], x[0]] <= 251: 
        y = repeats[0, j]
        x = repeats[1, j]
        plt.scatter(x, y, c='m', s=10, alpha=0.9) 
    else:
        y = repeats[0, j]
        x = repeats[1, j]
        plt.scatter(x, y, c='g', s=10, alpha=0.9) 
plt.gca().invert_yaxis()
plt.axis('off')
plt.show() 

plt.figure(figsize=(8, 7), dpi=res)
plt.imshow(d34601_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
for j in range(0, 80):
    y = [repeats[0, j], repeats[2, j], repeats[4, j]]
    x = [repeats[1, j], repeats[3, j], repeats[5, j]]

    if d34601_latlong[1, y[1], x[1]] >= -60 and ~np.isnan(d34601_hibd[0, y[1], x[1]]) and 240 <= d33901_latlong[0, y[0], x[0]] <= 251: 
        y = repeats[2, j]
        x = repeats[3, j]
        plt.scatter(x, y, c='m', s=10, alpha=0.9) 
    else:
        y = repeats[2, j]
        x = repeats[3, j]
        plt.scatter(x, y, c='g', s=10, alpha=0.9) 
plt.gca().invert_yaxis()
plt.axis('off')
plt.show() 

plt.figure(figsize=(8, 7), dpi=res)
plt.imshow(d35201_im[wvl, :, :], vmin=0, vmax=22, cmap='Greys_r')
for j in range(0, 80):
    y = [repeats[0, j], repeats[2, j], repeats[4, j]]
    x = [repeats[1, j], repeats[3, j], repeats[5, j]] 
    if d35201_latlong[1, y[2], x[2]] >= -60 and ~np.isnan(d35201_hibd[0, y[2], x[2]]) and 240 <= d33901_latlong[0, y[0], x[0]] <= 251: 
        y = repeats[4, j]
        x = repeats[5, j]
        plt.scatter(x, y, c='m', s=10, alpha=0.9) 
    else:
        y = repeats[4, j]
        x = repeats[5, j]
        plt.scatter(x, y, c='g', s=10, alpha=0.9) 
plt.gca().invert_yaxis()
plt.axis('off')
plt.show() 