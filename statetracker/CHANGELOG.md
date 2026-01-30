# Changelog

All notable changes to StateTracker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflow for automated testing
- Ruff linter configuration
- Dependabot for automated dependency updates

## [0.1.0] - 2026-01-30

Initial release of StateTracker for composable state management.

### Added
- Core `State` class for discrete value tracking
- `Manager` context for state lifecycle management
- Logging support with configurable verbosity levels
- Comprehensive Sphinx documentation with ReadTheDocs integration
- JOSS paper (submitted)

#### Operations
- `product` - Cartesian product of states (all parents active)
- `stack` - Disjoint union of states (one parent active at a time)
- `slice` - Select subset of state values
- `repeat` - Repeat state patterns
- `shuffle` - Randomize state ordering
- `sample` - Sample from state space
- `split` - Split state into multiple parts
- `interleave` - Interleave values from multiple states
- `synced_to()` - Create synchronized parent states

#### Synchronization
- `SynchronizedGroup` for flexible state synchronization
- Automatic value propagation through DAG structure
- Conflict detection with `ConflictingValueAssignmentError`

#### Visualization
- ASCII tree visualization for state DAGs
- `show_tree()` method for debugging state structures

### Performance
- Optimized hot paths (~6% speedup in common operations)

### Fixed
- Value propagation bug when slicing then stacking states
- Iteration order and name propagation for synced states
- Synced state propagation with fixed (stateless) operations
