# Perceptron

A basic implementation of a perceptron classifier, including application to the iris dataset.

## Contents

The Perceptron folder contains three Python files:

- **Perceptron_algorithm.py**: Contains the perceptron algorithm class, including helper functions.
  
- **iris_0_setosa.py**: Application of the perceptron algorithm to the iris dataset for the classification of setosa vs. non-setosa flower type.
  
- **iris_1_versicolor.py**: Application of the perceptron algorithm to the iris dataset for the classification of versicolor vs. non-versicolor flower type.
  
- **iris_2_virginica.py**: Application of the perceptron algorithm to the iris dataset for the classification of virginica vs. non-virginica flower type.

## Running the Code

To run the code, download the zip file and extract it to a directory of your choice. Execute any of the `iris_0_setosa.py`, `iris_1_versicolor.py`, or `iris_2_virginica.py` files using the terminal or an editor with code execution capabilitys such as vscode..

Running the code will train the perceptron algorithm on the iris dataset. The accuracy of the found model will be printed to console and a 3D plot will be displayed of the datapoints and the found separation plane. A second plot will display the classification errors per iteration during training.

*In the case of `iris_1_versicolor.py`, you might want to run the code more than once to find the best separation plane.

## Dependencies

To run the code you will need the python sklearn, mathplotlib and numpy packages.
Make sure to install the necessary dependencies before running the code.


