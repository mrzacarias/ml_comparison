# ML Comparison

Highly based on: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Tester for different ML Algorithms

`checklibs.py`: A small script that checks if you have all the required libs to run `ever_ml.py`

`ml_comparison.py`: The ML checker. Will load a dataset, plot the data (whisker, histogram, scatter matrix), build the models with 6 different algorithms, train the models and make a accuracy prediction for all of them.

Everything is hardcoded right now, so to correctly run:
1. `python -W ignore ml_comparison.py` will do the magic
