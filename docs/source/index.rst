.. _perturbations_docs_mainpage:

#######################################
Arepyextras Perturbations Documentation
#######################################

.. toctree::
   :maxdepth: 2
   :hidden:

   install  
   reference/api/index
   changelog

Arepyextras Perturbations is the Aresys Python module to compute geophysics displacements and atmospheric signal delays.

This package relies on resources files that have been downloaded and attached to this project in order to properly perform
atmospheric computations. In particular, tropospheric VMF3 model legendre coefficients and stations grid points coordinates
(both for 1x1 and 5x5 grids) have been added as separate files in the resources module and actively use in the code.
These files can be found here:

| VMF3 Legendre coefficients: `https://vmf.geo.tuwien.ac.at/codes/vmf3.m <https://vmf.geo.tuwien.ac.at/codes/vmf3.m>`_
| Station coordinates grid 1x1: `https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_1x1.txt <https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_1x1.txt>`_
| Station coordinates grid 5x5: `https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_5x5.txt <https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_5x5.txt>`_
