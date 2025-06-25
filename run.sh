#!/bin/bash

# Depending on the first argument, run the Python script with or without the -m option
if [ "$1" == "1" ]; then
    pylucid tests/bindings/pylucid/TestBarrier3.py --seed 42 --gamma 2.0 --time_horizon 5 --num_samples 1000 --lambda 1e-6 --sigma_f 15.0 --sigma_l 1.0 1.0 --num_frequencies 4 --plot --verify --oversample_factor 8.0 --c_coefficient 0.2
elif [ "$1" == "2" ]; then
    pylucid tests/bindings/pylucid/MinScenarioConfig.py --seed 42 --gamma 1.0 --time_horizon 5 --num_samples 1000 --lambda 1e-3 --sigma_f 15.0 --sigma_l 1.75555556 --num_frequencies 4 --plot --verify --oversample_factor 32.0 -v 4 --problem_log_file problem.lp 
elif [ "$1" == "3" ]; then
    pylucid --system_dynamics 'x1 / 2' --X_bounds 'RectSet([-1], [1])' --X_init 'RectSet([-0.5], [0.5])' --X_unsafe 'MultiSet([RectSet([-1], [-0.9]), RectSet([0.9], [1])])' --seed 42 --gamma 1.0 --time_horizon 5 --num_samples 1000 --lambda 1e-3 --sigma_f 15.0 --sigma_l 1.75555556 --num_frequencies 4 --plot --verify --oversample_factor 32.0 -v 4 --problem_log_file problem.lp 
else
    echo "Usage: $0 <1|2>"
    exit 1
fi
