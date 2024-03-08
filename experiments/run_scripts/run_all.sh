# To produce Table 1, run the following command:
python pysr_stress_strain.py

# To produce Table 2, run the following command:
python fsrd_study.py

# To produce Figure 4, run the following command:
python risk_scores_study.py

# To produce Figure 5, run the following command:
python torque_study.py

# To produce Table_3, run the following command:
python pysr_temperature.py

# To produce Figure 7, run the following command:
python temperature_study.py

# To produce Table 13, run the following commands:
python share_experiment.py banana classification --n_trials 5
python share_experiment.py breast_cancer classification --n_trials 5
python share_experiment.py breast classification --n_trials 5
python share_experiment.py diabetes classification --n_trials 5
python benchmarks_2.py --algs lr xgb ebm ebm_no_interactions pygam --datasets banana breast breast_cancer diabetes --n_trials 5 --tune

# To produce Figure 10, run the following commands:
python robustness_share.py --noise_levels 0.1 1.0 2.0 5.0 10.0 20 100
python robustness_xgb.py --noise_levels 0.1 1.0 2.0 5.0 10.0 20 100 --tune



