from spectrex import (
    RealImages,
)

# ------------------------------------------------------------------
# --------------------- Initialize Reals class ---------------------
# ------------------------------------------------------------------
GRISM = "GR150C" 
FILTER = "F200W"

DIRECT_FULL_ORIGIN = (1010,300) # At pixel DIRECT_FULL_ORIGIN the (0,0) coordinate of the direct image starts. This is usefull to define a specific snippet from a full-sized 2048*2048 image
Dispersed_SHAPE = (20, 600) # Shape of the dispersed image
Direct_SHAPE = (30, 1000) # Shape of the direct image
SOURCE_ORIGIN = (5, 200) # At pixel SOURCE_ORIGIN the (0,0) coordinate of the dispersed image starts

real = RealImages(grism=GRISM, filter= FILTER,  DIRECT_FULL_ORIGIN = DIRECT_FULL_ORIGIN , Dispersed_SHAPE = Dispersed_SHAPE, Direct_SHAPE = Direct_SHAPE, SOURCE_ORIGIN =  SOURCE_ORIGIN)

# -----------------------------------------------------------------
# ------------- Download matching image pairs pipeline ------------
# -----------------------------------------------------------------

obs = real.query_niriss_program(program=3383) # Search criteria for match
obs = real.add_time_and_position_columns(obs) # Search criteria for match

# all pairs
pairs = real.find_direct_grism_pairs_debug(
    obs,
    target_ra=23.35,
    target_dec=30.49,
    max_target_sep_arcsec=300.0,
    max_pair_sep_arcsec=300.0,
    max_time_delta_min=360.0,
) 

# sort pairs by minimal difference
pairs = sorted(
    pairs,
    key=lambda p: (
        not p[2],   # prefer same_visit=True
        p[1],       # smaller angular separation
        p[0],       # smaller time difference
        
    ),
)

if len(pairs) == 0:
    print("No pairs.")
else:
    for dt_min, sep_arcsec, same_visit, direct, grism in pairs:
        print("\nPAIR")
        print("dt [min]      =", dt_min)
        print("sep [arcsec]  =", sep_arcsec)
        print("same visit    =", same_visit)
        print("direct:", direct["ArchiveFileID"], direct["exp_type"], direct["filter"], direct["niriss_pupil"])
        print("grism: ", grism["ArchiveFileID"], grism["exp_type"], grism["filter"], grism["niriss_pupil"])
        

dt_min, sep_arcsec, same_visit, direct_row, grism_row = pairs[0] # choose 0th, i.e. best from sorted list

# download the pair
direct_fits, dispersed_fits = real.download_pair_with_observations(
    direct_row,
    grism_row,
    download_dir="mast_downloads",
    prefer_suffix="_rate.fits",
)

print("direct_fits    =", direct_fits)
print("dispersed_fits =", dispersed_fits)

# -----------------------------------------------------------------
# -------------------- Running pipeline ---------------------------
# -----------------------------------------------------------------

# initialize direct and dispersed image, given by the downloading pipeline
direct_fits = real.ROOT / "mast_downloads"/"mastDownload"/"JWST"/"jw03383181001_03201_00002_nis"/"jw03383181001_03201_00002_nis_rate.fits"
dispersed_fits = real.ROOT/ "mast_downloads"/"mastDownload"/"JWST"/ "jw03383182001_05201_00003_nis"/"jw03383182001_05201_00003_nis_rate.fits"

# Run the recovery pipeline
result = real.run_real_scene_optimized_recovery(
    direct_fits=direct_fits,
    dispersed_fits=dispersed_fits,
    catalog_files= [real.TESTDATA / "Catalog" / "tri-00-ir.cat.fits"], # used as catalog for centroid of sources 
    wcs_reference_fits=direct_fits,
    ra_col="RA",
    dec_col="DEC",
    instrument_name="NIRISS",
    detector=None,
    noise_factor=3.0,
    min_radius_sigma=0.25,
    max_radius_sigma=6.0,
    fixed_sigma=1.61, # Note: this is F200W specific. In need of a look up table!!!!
    PLOTS=True, # visualizes results
    Save = True, # In case of Astronode, can't visualize directly and saves images to unittests/Images. Only active, when PLOTS = True
)
