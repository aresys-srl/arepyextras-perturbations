# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geodynamics/solid_tides.py core functionalities"""

import unittest

import numpy as np
from arepytools.io.metadata import PreciseDateTime

from arepyextras.perturbations.geodynamics.solid_tides import compute_displacement


class PlateTectonics(unittest.TestCase):
    """Testing solid_tides.py core functionalities"""

    def setUp(self) -> None:
        # creating test data
        self.pt_pos = np.array(
            [
                (-2468789.77437779, -4626148.4320329, 3620025.27093258),
                (-2467963.52819618, -4626181.75345945, 3620542.41415808),
                (-2467068.51440511, -4626651.66776616, 3620552.45331852),
                (-2466645.20366593, -4626877.04176162, 3620552.92942237),
                (-2466140.52981391, -4627136.54745231, 3620565.21049583),
            ]
        )
        self.time = PreciseDateTime.from_utc_string("16-NOV-2019 04:06:56.329529000000")
        self.displacement_ref = np.array(
            [
                [0.06305905468150182, 0.05339999182683428, -0.049454763905432464],
                [0.0630511871586805, 0.0533805772901235, -0.04944733489296774],
                [0.06304064719973929, 0.05336377369371278, -0.04942786763966205],
                [0.0630356479034235, 0.0533558552189551, -0.04941857281234307],
                [0.06302973059997202, 0.05334632085770409, -0.04940773149090209],
            ]
        )

    def test_compute_displacement(self):
        """Testing compute_displacement function"""
        displacement = compute_displacement(target_xyz_coords=self.pt_pos, acquisition_time=self.time)

        np.testing.assert_array_almost_equal(displacement, self.displacement_ref, 1e-12)


if __name__ == "__main__":
    unittest.main()
