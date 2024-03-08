# Shape Arithmetic Expressions
This is the original repository for the paper "Shape Arithmetic Expressions: Advancing Scientific Discovery Beyond Closed-form Equations".

## Clone the repository
Clone the repository using

```
git clone --recurse-submodules -j8 https://github.com/krzysztof-kacprzyk/SHAREs.git 
```

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n shares --file environment.yml
```

## Running all experiments
To run all experiments navigate to experiments using
```
cd experiments
```
and run
```
./run_scripts/run_all.sh
```
Or you can call the scripts individually in run_scripts.

The results will be saved in
```
experiments/results/
```

## Figures and tables
Jupyter notebooks used to create all figures and tables in the paper can be found in experiments/analysis.

## Other information
To properly install PySR follow instructions on https://github.com/MilesCranmer/PySR

## Citations
If you use this code, please cite using the following information.
*Kacprzyk, K. & van der Schaar, M. Shape Arithmetic Expressions: Advancing Scientific Discovery Beyond Closed-form Equations. in Proceedings of The 27th International Conference on Artificial Intelligence and Statistics (PMLR, 2024).*

```
@inproceedings{Kacprzyk.ShapeArithmeticExpressions.2024,
  title = {Shape {{Arithmetic Expressions}}: {{Advancing Scientific Discovery Beyond Closed-form Equations}}},
  booktitle = {Proceedings of {{The}} 27th {{International Conference}} on {{Artificial Intelligence}} and {{Statistics}}},
  author = {Kacprzyk, Krzysztof and {van der Schaar}, Mihaela},
  year = {2024},
}
```
