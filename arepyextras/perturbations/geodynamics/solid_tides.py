# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geodynamics Earth Crustal Displacement: Solid Tides submodule
-------------------------------------------------------------

Accounting for solid tides displacement using the IERS Conventions model.
"""

import numpy as np
from arepyextras.iers_solid_tides.wrapper.main import solid_earth_tides_core
from arepytools.geometry.conversions import xyz2llh
from arepytools.io.metadata import PreciseDateTime

SECONDS_IN_MINUTES = 60
SECONDS_IN_HOUR = 3600


def _compute_displacement_unit_vectors(lat_geo_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Computing the unit vectors for each displacement direction, namely north, east and up.

    Parameters
    ----------
    lat_geo_rad : np.ndarray
        array of latitude coordinates geocentric in radians, shape (N,)
    lon_rad : np.ndarray
        array of longitude coordinates in radians, shape (N,)

    Returns
    -------
    np.ndarray
        array of shape (3, N, 3) where first dimension represent the displacement direction [north, east, up],
        second dimension is the size of the input arrays, third dimension is the unit vector coordinate [x, y, z]
    """

    # filling up an array of shape (3, N, 3):
    # first dimension represent the displacement unit vector, namely north [0], east [1] and up [2]
    # second dimension is the number of points in the input arrays
    # third dimension is the number of components of each unit vector (x, y, z)
    displacement_unit_vectors = np.zeros(shape=(3, lat_geo_rad.size, 3))

    # north unit vector
    displacement_unit_vectors[0] = np.array(
        [
            -np.sin(lat_geo_rad) * np.cos(lon_rad),
            -np.sin(lat_geo_rad) * np.sin(lon_rad),
            np.cos(lat_geo_rad),
        ]
    ).T

    # east unit vector
    displacement_unit_vectors[1] = np.array([-np.sin(lon_rad), np.cos(lon_rad), np.zeros_like(lon_rad)]).T

    # up unit vector
    displacement_unit_vectors[2] = np.array(
        [
            np.cos(lat_geo_rad) * np.cos(lon_rad),
            np.cos(lat_geo_rad) * np.sin(lon_rad),
            np.sin(lat_geo_rad),
        ]
    ).T

    return displacement_unit_vectors


def compute_displacement(target_xyz_coords: np.ndarray, acquisition_time: PreciseDateTime) -> np.ndarray:
    """Estimate the input coordinates displacement due to earth tides based on acquisition time using the arepyextras
    iers_solid_tides module.

    Parameters
    ----------
    target_xyz_coords : PreciseDateTime
        the input coordinates on scene, xyz format, shape Nx3
    acquisition_time : PreciseDateTime
        sensor acquisition time of the input coordinates on scene

    Returns
    -------
    np.ndarray
        updated coordinates, same input coordinate but with displacement added
    """

    # calculating the acquisition time in seconds relative to that day
    acq_time_sec = (
        acquisition_time.hour_of_day * SECONDS_IN_HOUR
        + acquisition_time.minute_of_hour * SECONDS_IN_MINUTES
        + acquisition_time.second_of_minute
    )

    # coordinates conversion: geodetic to geocentric
    llh_coordinates = xyz2llh(target_xyz_coords.T).T
    lat_geocentric = np.arctan((1 - 1 / 298.25642) ** 2 * np.tan(llh_coordinates[:, 0]))

    # compute displacement unit vectors along north, east and up
    displacement_unit_vectors = _compute_displacement_unit_vectors(
        lat_geo_rad=lat_geocentric, lon_rad=llh_coordinates[:, 1]
    )

    # creating an empty array of shape (N, 3), columns are: north, east and up
    displacement_interp = np.zeros_like(target_xyz_coords)

    # compute displacement values for each point
    for point_id, _ in enumerate(target_xyz_coords):
        # converting lat and lon to deg and call the SOLID executable
        lat_deg = np.rad2deg(llh_coordinates[point_id, 0])
        lon_deg = np.rad2deg(llh_coordinates[point_id, 1])
        tide_displacement_df = solid_earth_tides_core(
            year=acquisition_time.year,
            month=acquisition_time.month,
            day_of_month=acquisition_time.day_of_the_month,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )

        # evaluate the interpolated displacement at the right time (in seconds relative to midnight of that day)
        # north displacement interpolated value
        displacement_interp[point_id, 0] = np.interp(
            acq_time_sec, tide_displacement_df["time_s"], tide_displacement_df["north"]
        )
        # east displacement interpolated value
        displacement_interp[point_id, 1] = np.interp(
            acq_time_sec, tide_displacement_df["time_s"], tide_displacement_df["east"]
        )
        # up displacement interpolated value
        displacement_interp[point_id, 2] = np.interp(
            acq_time_sec, tide_displacement_df["time_s"], tide_displacement_df["up"]
        )

    # compute displacement vectors
    north_total_displacement = displacement_interp[:, 0].reshape(-1, 1) * displacement_unit_vectors[0]
    east_total_displacement = displacement_interp[:, 1].reshape(-1, 1) * displacement_unit_vectors[1]
    up_total_displacement = displacement_interp[:, 2].reshape(-1, 1) * displacement_unit_vectors[2]

    return north_total_displacement + east_total_displacement + up_total_displacement
