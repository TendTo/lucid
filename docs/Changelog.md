# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3]

### Added

- Utility function on `Configuration` class to correctly initialize the configuration after load
- Ability to use user-defined variables in the system dynamics function when using the Python configuration, with a map between the user-defined variable names and their values at each time step
- More variables exposed by `Stats`
- More plot utility functions for visualizing the results
- Prepare support for Python 3.14 (need to upgrade gurobi to 13 first) 

### Changed

- Improved documentation
- Responsive design for the GUI
- Update dependencies version

### Removed

- Support for Python 3.8

## [0.0.2]

### Added

- Contour for gamma and eta planes when plotting the 2D barrier function
- MACRO documentation
- Windows support for HiGHS
- Support for `epsilon`, `b_kappa`, `b_norm`, `feature_sigma_l` and `set_scaling` parameters in the configuration
- PSO optimizer
- SoPlex optimizer
- Monte Carlo sampling to use as a baseline
- Online demo running in the Browser using WebAssembly
- Ability to record internal statistics during tool execution via the `Stats` class
- Ability to save results in `.yaml`, `.json`, `.mat`, `.npz` or `.csv` formats

### Changed

- The default examples in the GUI have been updated to match the ones in the paper
- Optimized the number of points required for plotting eta and gamma planes
- Pylucid loads Gurobi dynamically at runtime, to avoid forcing its installation if it is not needed
- Improved LP problem formulation
- Improved documentation
- Renamed `num_oversample` to `lattice_resolution` for clarity

### Fixed

- Race condition in logger creation
- Bug in the LP formulation

## [0.0.1]

### Added

- Initial release

[0.0.1]: https://github.com/TendTo/lucid/tree/0.0.1
[0.0.2]: https://github.com/TendTo/lucid/compare/0.0.1...0.0.2
[0.0.3]: https://github.com/TendTo/lucid/compare/0.0.2...0.0.3
[NEXT.VERSION]: https://github.com/TendTo/lucid/compare/0.0.3...main
[NEXT.VERSION]: https://github.com/TendTo/lucid
