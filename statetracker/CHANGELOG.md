# Changelog

All notable changes to StateTracker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-03

Initial release. See [StateTracker documentation](https://statetracker.readthedocs.io) for a full description.

### Added
- Composable discrete states with unidirectional value propagation
- State algebra: product, stack, slice, repeat, shuffle, sample, split, interleave, sync
- Automatic state composition and decomposition through DAG structures
- Conflict detection for incompatible value assignments
- ASCII tree visualization for debugging state dependencies
