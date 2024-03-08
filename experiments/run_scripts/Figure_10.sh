# To produce Figure 10, run the following commands:
python robustness_share.py --noise_levels 0.1 1.0 2.0 5.0 10.0 20 100
python robustness_xgb.py --noise_levels 0.1 1.0 2.0 5.0 10.0 20 100 --tune