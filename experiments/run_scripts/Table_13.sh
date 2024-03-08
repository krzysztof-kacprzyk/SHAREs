# To produce Table 13, run the following commands:
python share_experiment.py banana classification --n_trials 5
python share_experiment.py breast_cancer classification --n_trials 5
python share_experiment.py breast classification --n_trials 5
python share_experiment.py diabetes classification --n_trials 5
python benchmarks_2.py --algs lr xgb ebm ebm_no_interactions pygam --datasets banana breast breast_cancer diabetes --n_trials 5 --tune
