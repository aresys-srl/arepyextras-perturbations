# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Atmospheric and Geodynamics Perturbations
-----------------------------------------
"""

from importlib import resources as res

from . import resources

__version__ = "1.1.0"

# tropospheric legendre coefficients
tropospheric_coeff_path = res.files(resources).joinpath("troposphere_support", "tropospheric_legendre_coefficients")
tropospheric_legendre_coeff = dict.fromkeys(
    ["anm_bh", "anm_bw", "anm_ch", "anm_cw", "bnm_bh", "bnm_bw", "bnm_ch", "bnm_cw"]
)
for key in tropospheric_legendre_coeff:
    tropospheric_legendre_coeff[key] = tropospheric_coeff_path.joinpath(key + ".txt").read_bytes()

# tropospheric grid data stations
grid_stations_fine = res.files(resources).joinpath("troposphere_support", "gridpoint_coord_1x1.txt")
grid_stations_coarse = res.files(resources).joinpath("troposphere_support", "gridpoint_coord_5x5.txt")
