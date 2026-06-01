#!/usr/bin/env python3
# evalscope_ext/tools/compare_runs.py
# Thin wrapper — delegates to the cerebras_pruner compare_runs implementation.
#
# Usage (as specified in challenge):
#   python -m evalscope_ext.tools.compare_runs \
#       --full ./results_full/ \
#       --pruned ./results_pruned/

from evalscope.benchmarks.cerebras_pruner.tools.compare_runs import main

if __name__ == '__main__':
    main()
