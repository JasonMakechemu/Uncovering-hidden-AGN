
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:03:16 2025

@author: jason
"""

import webbpsf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.nddata
import astropy.io.fits as fits


'''
Define file paths for different bands of observed psfs.

src id 683931
'''
filenames = {
    'F115W': '/Users/jason/Downloads/COSMOS-Web AGN Candidate Cutout f115w 110.fits',
    'F150W': '/Users/jason/Downloads/COSMOS-Web AGN Candidate Cutout f150w 110.fits',
    'F277W': '/Users/jason/Downloads/COSMOS-Web AGN Candidate Cutout f277w 110.fits',
    'F444W': '/Users/jason/Downloads/COSMOS-Web AGN Candidate Cutout f444w 110.fits'
}

# Flux values for each filter (total flux and peak flux)

flux_values = {
    'F115W': 111.99102020263672,
    'F150W': 84.50768280029297,
    'F277W': 36.34436798095703,
    'F444W': 41.25001144
}

peak_flux_values = {
    'F115W': 26.868114471435547,
    'F150W': 105.24082946777344,
    'F277W': 47.82408524,
    'F444W': 108.31375122070312
}


# Set the PSF center and box size
psf_center = (52, 52)  # (X, Y) order
boxsize = 60


'''
check exact coordinates in source extractor. Shift obs image
to exact central coordinates.

make difference image at same scale as obs/sim psf. Add another panel
'''


def plot_data_sim_comparison(obs_psf, obs_psf_err, sim_psf, band, flux, peak_flux):
    '''
    Display observed data, model, and difference between the two;
    and compute a goodness of fit metric, chi^2
    '''
    
    fig, axes = plt.subplots(figsize=(13, 3), ncols=3)
    
    vmax = np.nanmax(obs_psf)
    cmap = matplotlib.cm.gist_heat
    cmap.set_bad(cmap(0))
    
    # Observed PSF
    axes[0].imshow(obs_psf, norm=matplotlib.colors.LogNorm(vmax/1e4, vmax), cmap=cmap, origin='lower')
    axes[0].set_title(f"Observed PSF ({band})")
    axes[0].text(3, 3, f"Flux: {flux:.2f}\nPeak: {peak_flux:.2f}", color='white', fontsize=9)



    # Simulated PSF
    webbpsf.display_psf(sim_psf, ext='DET_DIST', vmax=0.1, vmin=1e-5, ax=axes[1])

    # Scale simulated PSF using total flux
    scalefactor = flux / np.nansum(sim_psf["DET_DIST"].data)  
    difference = obs_psf - (sim_psf["DET_DIST"].data * scalefactor)
    
    
    '''
    Difference Plot ( Fractional Intensity per pixel -> intensity value of a pixel expressed
    as a fraction of the maximum possible intensity)
    
    chisqr is normalised chisqr statistic. Squared difference between observed and 
    expected values. This is divided by corresponding squared error. nansum ignores
    nan values. isfinite only counts non-infinite elements in 'difference'.
    
    '''
    axes[2].imshow(difference, norm=matplotlib.colors.LogNorm(vmax/1e4, vmax), cmap=cmap, origin='lower')
    chisqr = np.nansum(difference**2 / obs_psf_err**2) / np.isfinite(difference).sum()
    axes[2].set_title("Difference")
    axes[2].text(3, 3, f"$\\chi^2$ = {chisqr:.2f}", color='white', fontsize=9)

    for ax in [axes[0], axes[2]]:
        ax.set_xlabel("Pixels")
    
    plt.show()
    
    
#%%

# Loop through each wavelength band and process the data
for band, filename in filenames.items():
    inst = webbpsf.NIRCam()
    inst.filter = band
    #inst.options['charge_diffusion_sigma'] = 0.0295

    # Load observed image data
    with fits.open(filename) as hdul:
        obs_data = hdul[0].data

    
    obs_psf = astropy.nddata.Cutout2D(obs_data, position=psf_center, size=boxsize).data
    
    #calculating an error of obs PSF by taking std_dev of bkg noise
    background_std = np.nanstd(obs_data)
    obs_psf_err = np.full_like(obs_psf, background_std)
    #obs_psf_err = astropy.nddata.Cutout2D(obs_data, position=psf_center, size=boxsize).data
    #obs_psf -= np.nanmedian(obs_psf)  # Background subtraction
    
    # Configure instrument and compute PSF simulated PSF
    inst.detector_position = psf_center
    sim_psf = inst.calc_psf(fov_pixels=boxsize, nlambda=5)
    
    # Retrieve corresponding flux values to make synthetic psf
    flux = flux_values[band]
    peak_flux = peak_flux_values[band]

    # Plot comparison with flux scaling
    plot_data_sim_comparison(obs_psf, obs_psf_err, sim_psf, band, flux, peak_flux)
    
    
    
    



#%%


from astropy.visualization import make_lupton_rgb

# Select four filters for RGB mapping
rgb_filters = ['F115W', 'F150W', 'F277W', 'F444W']  # (R, G, B)



# Function to normalize PSFs for RGB scaling
def normalize_psf(psf):
    return (psf - np.nanmin(psf)) / (np.nanmax(psf) - np.nanmin(psf))



# Dictionary to store processed images
obs_psfs_rgb = {}
sim_psfs_rgb = {}



# Process the selected filters
for band in rgb_filters:
    
    filename = filenames[band]
    inst = webbpsf.NIRCam()
    inst.filter = band

    # Load observed PSF
    with fits.open(filename) as hdul:
        obs_data = hdul[0].data

    obs_psf = astropy.nddata.Cutout2D(obs_data, position=psf_center, size=boxsize).data
    #obs_psf -= np.nanmedian(obs_psf)  # Background subtraction
    obs_psfs_rgb[band] = normalize_psf(obs_psf)

    # Compute simulated PSF
    inst.detector_position = psf_center
    sim_psf = inst.calc_psf(fov_pixels=boxsize, nlambda=5)
    
    # Scale using total flux
    scalefactor = flux_values[band] / np.nansum(sim_psf["DET_DIST"].data)
    sim_psfs_rgb[band] = normalize_psf(sim_psf["DET_DIST"].data * scalefactor)



# Stack RGB images
obs_rgb = make_lupton_rgb(obs_psfs_rgb[rgb_filters[3]]*1.1,
                          obs_psfs_rgb[rgb_filters[2]]*0.75,
                          (obs_psfs_rgb[rgb_filters[0]]+obs_psfs_rgb[rgb_filters[1]])*0.4,
                          minimum=0,
                          Q=8,
                          stretch=0.025)



sim_rgb = make_lupton_rgb(sim_psfs_rgb[rgb_filters[3]]*1.1,
                          sim_psfs_rgb[rgb_filters[2]]*0.75,
                          (sim_psfs_rgb[rgb_filters[0]]+sim_psfs_rgb[rgb_filters[1]])*0.4,
                          minimum=0,
                          Q=8,
                          stretch=0.025)

# Compute residual and normalize for visualization
residual_rgb = np.clip(obs_rgb.astype(float) - sim_rgb.astype(float), 0, 255).astype(np.uint8)




#%%

mse = np.nanmean((obs_rgb.astype(float) - sim_rgb.astype(float))**2)
rmse = np.sqrt(mse)
nrmse = rmse / 255
nrmse_range = rmse / (np.nanmax(obs_rgb) - np.nanmin(obs_rgb))

nrss = np.nansum((obs_rgb - sim_rgb) ** 2) / np.nansum(obs_rgb ** 2)

'''

Images are 8-bit, maximum possible pixel intensity is 255:

An NRMSE of 0.06 means that, on average,
residual error per pixel is 6% of the full 
intensity range.

Interpretation:
NRMSE < 0.05 → Excellent match (very low residuals), bordeline lol

'''

print(mse)
print(rmse)
print(nrmse)
print(nrmse_range)

print(nrss)




#%%

# Plot images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(obs_rgb, interpolation='nearest', origin='lower')
axes[0].set_title("Observed PSF (RGB)")

axes[1].imshow(sim_rgb, interpolation='nearest', origin='lower')
axes[1].set_title("Simulated PSF (RGB)")

axes[2].imshow(residual_rgb, interpolation='nearest', origin='lower')
axes[2].set_title("Residual (Observed - Simulated)")
axes[2].text(3, 3, f"Normalised RMS error (nrmse) = {nrmse:.3f}", color='white', fontsize=11)


for ax in axes:
    ax.axis("off")

# Save the figure before showing it
plt.savefig("psf_comparison.png", dpi=300, bbox_inches='tight')


plt.show()


#%%


# Save each panel separately
plt.imsave("observed_psf.png", obs_rgb, dpi=300, origin='lower')
plt.imsave("simulated_psf.png", sim_rgb, dpi=300, origin='lower')
plt.imsave("residual_psf.png", residual_rgb, dpi=300, origin='lower')


#%%


from astropy.io import fits

# Save individual PSF images
for band in rgb_filters:
    # Save observed PSF
    fits.writeto(f"obs_psf_{band}.fits", obs_psfs_rgb[band], overwrite=True)
    
    # Save simulated PSF
    fits.writeto(f"sim_psf_{band}.fits", sim_psfs_rgb[band], overwrite=True)



# Stack PSFs into a single 3D array
obs_psf_stack = np.array([obs_psfs_rgb[band] for band in rgb_filters])
sim_psf_stack = np.array([sim_psfs_rgb[band] for band in rgb_filters])

# Save combined FITS file
fits.writeto("obs_psf_combined.fits", obs_psf_stack, overwrite=True)
fits.writeto("sim_psf_combined.fits", sim_psf_stack, overwrite=True)


#%%













    
    
    
    
    
    
