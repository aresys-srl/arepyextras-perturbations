Changelog
=========

v1.1.1
------

First official release.

**Additional Features**

- geophysics submodule: added drift velocities optional input to compute plate tectonics displacement vectors
- atmospheric submodule: added `generate_ionospheric_map_filename` and `generate_tropospheric_map_name_for_vmf_data` free functions to generate map filenames starting form acquisition date and other specific inputs
- ionosphere submodule: added support for new CDDIS map format name after GSS week 2238

**Other changes**

- changed resources management policy: pkgutil substituted with `importlib.resources`
- ionosphere: ionosphere height and data correction exponent read directly from map files
- ionosphere: supported analysis centers implementation changed to enum class

**Bug fixing**

- minor code fixes (linting, spelling, formatting)
