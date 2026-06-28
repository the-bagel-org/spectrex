"""specTrex — Grism spectra extraction in crowded regions.

Public API
----------
InstrumentConfig
    Instrument configuration loader (grism, filter, sensitivity curves).
EigenspectraBasis
    PCA eigenspectra basis for spectral representation.
ForwardOperatorProtocol
    Protocol that any forward operator must satisfy.
SciPySparseOperator
    scipy-sparse forward operator (Phase 1).
NoiseModel
    Poisson + read-noise model.
SpectralSolver
    LSQR/LSMR solver for grism deconvolution.
JAXOperator
    JAX compact-trace forward operator (Phase 2).
JAXProximalSolver
    FISTA proximal solver with group-L1 regularisation (Phase 2).
"""

import logging as _logging

from spectrex._version import version as __version__
from spectrex.basis import EigenspectraBasis
from spectrex.instrument import InstrumentConfig
from spectrex.operator import ForwardOperatorProtocol, SciPySparseOperator
from spectrex.solver import NoiseModel, SpectralSolver
from spectrex.jax_operator import JAXOperator
from spectrex.jax_solver import JAXProximalSolver
from spectrex.real_images import RealImages
from spectrex.mock_images import MockImages

# Standard library best practice for packages: attach a NullHandler so the
# library never emits "No handlers could be found" warnings when the caller
# hasn't configured logging. Users configure the root "spectrex" logger to
# capture log output from all submodules.
_logging.getLogger("spectrex").addHandler(_logging.NullHandler())

#: Package-level logger.  Users can configure it directly::
#:
#:     import logging, spectrex
#:     logging.getLogger("spectrex").setLevel(logging.DEBUG)
logger: _logging.Logger = _logging.getLogger("spectrex")

__all__ = [
    "__version__",
    "InstrumentConfig",
    "EigenspectraBasis",
    "ForwardOperatorProtocol",
    "SciPySparseOperator",
    "NoiseModel",
    "SpectralSolver",
    "JAXOperator",
    "JAXProximalSolver",
    "RealImages",
    "MockImages",
    "logger",
]
