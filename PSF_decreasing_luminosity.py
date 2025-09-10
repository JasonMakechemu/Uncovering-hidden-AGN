#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:14:45 2025

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from scipy.ndimage import zoom

# Load synthetic PSF FITS files
i_name = '/Users/jason/Desktop/webbpsf-data/sim_psf_F115W.fits'
i_name_2 = '/Users/jason/Desktop/webbpsf-data/sim_psf_F150w.fits'
g_name = '/Users/jason/Desktop/webbpsf-data/sim_psf_F277w.fits'
r_name = '/Users/jason/Desktop/webbpsf-data/sim_psf_F444w.fits'

# Open the FITS files data
i1 = fits.open(i_name)[0].data
i2 = fits.open(i_name_2)[0].data
blue = i1 + i2
g = fits.open(g_name)[0].data
r = fits.open(r_name)[0].data

# Define target shape
target_shape = (
    max(r.shape[0], g.shape[0], blue.shape[0]),
    max(r.shape[1], g.shape[1], blue.shape[1])
)

# Resize the PSFs to match the largest PSF
r_resized = zoom(r, (target_shape[0] / r.shape[0], target_shape[1] / r.shape[1]), order=3)
g_resized = zoom(g, (target_shape[0] / g.shape[0], target_shape[1] / g.shape[1]), order=3)
blue_resized = zoom(blue, (target_shape[0] / blue.shape[0], target_shape[1] / blue.shape[1]), order=3)

# Ensure dimensions match
blue_cropped_resized = blue_resized[:r_resized.shape[0], :r_resized.shape[1]]

# Define parameters
num_rows = 7  # Color transition from red to blue
num_cols = 7  # Consistent brightness across the grid
luminosity_decay = 0.5  # Decay factor per row

# Generate color weights for transition from red to blue
color_weights = np.linspace(0, 1, num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

image_counter = 1

# Compute total flux in the original images
total_flux = np.sum(r_resized + g_resized + blue_cropped_resized)

for row in range(num_rows):
    # Set row-wise flux scaling factor
    factor = 3 * (luminosity_decay ** row)
    
    for col in range(num_cols):
        # Adjust color blend
        red_weight = 1 - color_weights[col]
        blue_weight = 1 + color_weights[col]
        green_weight = 0.5 + color_weights[col]  # Middle intensity for green

        # Apply initial weight factors
        r_scaled = r_resized * red_weight
        g_scaled = g_resized * green_weight
        blue_scaled = blue_cropped_resized * blue_weight
        
        # Compute total flux in the new image
        new_total_flux = np.sum(r_scaled + g_scaled + blue_scaled)
        
        # Normalize to maintain original total flux
        normalization_factor = total_flux / new_total_flux
        r_normalized = r_scaled * normalization_factor * factor
        g_normalized = g_scaled * normalization_factor * factor
        blue_normalized = blue_scaled * normalization_factor * factor
        
        # Generate RGB image
        rgb_image = make_lupton_rgb(
            r_normalized * 1.1, 
            g_normalized * 0.75, 
            blue_normalized * 0.4, 
            minimum=0, Q=8, stretch=0.025
        )


        
        # Display image
        axes[row, col].imshow(rgb_image, interpolation='nearest', origin='lower')
        axes[row, col].axis("off")
        
        # Save individual image
        plt.imsave(f"psf_image_{image_counter}.png", rgb_image)
        
        # Compute flux of the modified image
        current_flux = np.sum(r_normalized + g_normalized + blue_normalized)
        flux_ratio = current_flux / total_flux  # Ratio compared to the baseline
        
        # Overlay flux ratio on image
        axes[row, col].text(
            5, 5, f"{image_counter}\nFlux: {flux_ratio:.2f}",
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.75, edgecolor='none')
        )
        
        image_counter += 1

plt.tight_layout()
plt.savefig("psf_grid.png", dpi=600, bbox_inches='tight')
plt.show()


