import matplotlib.pyplot as plt

from spectrex import (
    MockImages,
)
# -------------------------------------------------------------------------------------
# --------------------------- Initialize mock class -----------------------------------
# -------------------------------------------------------------------------------------

Dispersed_SHAPE = (20, 600) # Shape of the dispersed image
Direct_SHAPE = (30, 1000) # Shape of the direct image
SOURCE_ORIGIN = (5, 200) # At pixel SOURCE_ORIGIN the (0,0) coordinate of the dispersed image starts
SOURCE_DENSITY = 0.05 # Source density in the images, i.e. number_of_sources/total_pixels
grism="GR150C"
filter="F200W"


mock = MockImages(grism=grism, filter=filter, Dispersed_SHAPE = Dispersed_SHAPE,Direct_SHAPE = Direct_SHAPE, SOURCE_ORIGIN = SOURCE_ORIGIN) # initialization of class

# ------------------------------------------------------------------------------------
# ----------------- Recovery and generation of mock image ----------------------------
# ------------------------------------------------------------------------------------

mock.run_mock_scene_optimized_recovery(
                SOURCE_DENSITY=SOURCE_DENSITY,
                PARITY=True, # Saves parity plots to unittests/Images
                PLOTS=True, # Saves plots to unittests/Images
                NOISE = False, # set true for noise model
            )


# ------------------------------------------------------------------------------------
# ----------------- density plot example. Iteration over densities -------------------
# ------------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 1, figsize=(14, 5)) # plot the error over pixel density

mock.run_densities(
    MAX_DENSITY=0.01,
    STEPS=1,
    ax1= axes,
)

# visualize density plot
axes.set_title("Optimized with Mixing Operator")
plt.tight_layout()
plt.show()