Data products (all are fits files unless otherwise noted):
Derived data products:
IBD_DOYSC (example: Dec 05 2009 scan 0 is IBD_33900.fit): contains the spatial data cube with integrated 3 micron band depth values
BD_DOYSC: hydration (3 micron) band depth
dDOYSC_bd_new_resmean: 3 micron band depth with a running mean done over the spectral dimension to smooth noise
Parameters_dDOYSC: Temperature, emissivity, 
App_Refl_dDOYSC: Cube (2D spatial, 1D spectral) of Apparent reflectance. 
dDOYSC_pibd: 2 micron integrated band depth (pyroxene)

Calibrated & Geometery:
Calibrated_Cube_dDOYSC: Calibrated (2D spatial, 1D spectral)
Wave_cube_dDOYSC: Calibrated wavelength cube (2D spatial, 1D spectral)
Modtran_d33901: MODTRAN data extrapolated into a cube. 
Satmask_cube_dDOYSC_fixed: Masking uncalibrated data which saturated the instrument per spatial pixel. 
Incidence_ang_DecDA_S: (example: Incidence_ang_Dec05_0.fit) Incidence angle for each observation
pix_to_long_lat_dDOYSC: Latitude longitude map. 
dDOYSC_tod_xymap: Cube (2D spatial, 1D spectral) of Time of Day
d1d2d3_repeats_01_full.txt: X, Y position for each DOY Scan 01 for a Latitude&Longitude repeat observation
