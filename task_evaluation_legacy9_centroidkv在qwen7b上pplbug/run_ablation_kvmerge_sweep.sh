#!/usr/bin/env bash
# Compatibility alias: focused graph-MIS mymethod ablations.
exec bash "$(dirname "$0")/run_ablation_mymethod_sweep.sh" "$@"
