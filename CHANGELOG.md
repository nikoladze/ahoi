# Changelog

## [0.4.1] - 2020-03-04
### Fixed
- fix importlib.util import

## [0.4] - 2020-03-04
### Added
- new "histogramdd" method which is much faster in case of "cumulative" or
  "orthogonal" selections - e.g. strictly increasing or decreasing cuts

### Fixed
- fix bug for chunkwise filling in mp mode

## [0.3] - 2020-02-22
### Added
- option to trim masks list (skip events that don't pass any combination or have 0 weight)
- on-the-fly calculation of combination index in method "c" (significant speedup)

## [0.2] - 2020-02-17
### Added
- support for chunkwise filling
- support for parallelization via multiprocessing
