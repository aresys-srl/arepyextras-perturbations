# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Atmospheric and Geodynamics Perturbations
-----------------------------------------
"""

from importlib import resources as res

from . import resources

__version__ = "1.1.1"

# tropospheric legendre coefficients
# these data has been taken from the data management script hosted by VMF Data Server (vmf3.m) that can be found here:
# https://vmf.geo.tuwien.ac.at/codes/vmf3.m
tropospheric_coeff_path = res.files(resources).joinpath("troposphere_support", "tropospheric_legendre_coefficients")
tropospheric_legendre_coeff = dict.fromkeys(
    ["anm_bh", "anm_bw", "anm_ch", "anm_cw", "bnm_bh", "bnm_bw", "bnm_ch", "bnm_cw"]
)
for key in tropospheric_legendre_coeff:
    tropospheric_legendre_coeff[key] = tropospheric_coeff_path.joinpath(key + ".txt").read_bytes()

# tropospheric grid data stations
# files can be found here:
# https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_1x1.txt
# https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_5x5.txt
grid_stations_fine = res.files(resources).joinpath("troposphere_support", "gridpoint_coord_1x1.txt")
grid_stations_coarse = res.files(resources).joinpath("troposphere_support", "gridpoint_coord_5x5.txt")
