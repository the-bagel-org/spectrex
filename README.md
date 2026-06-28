# specTrex

![trex mixed with spectra](docs/assets/logo_new.png)

To run the code with real images, run [test real images](unittests/test_reals.py). Either download the direct and dispersed images at a given position or use your own and run the recovery pipeline afterwards.


To run the code with randomly generated gaussian distributed mock data, run [test mock images](unittests/test_mock.py). Either run the recovery once with a fixed source density or use the iteration through multiple densities to visualize the increasing error.

When defining the sizes of the direct and dispersed image snippets, the direct has to be bigger or equal than the dispersed to ensure better recovery of the boundary cases.

