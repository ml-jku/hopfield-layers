## General info
This example shows an implementation of the __Classical Hopfield__, __Dense Hopfield__ and __Continuous Hopfield__ model with Numpy only.
Those classes can be found in `models.py`. 

Additional function for plotting and image preprocessing can be found in `utils.py`.

The three notebooks `classical_hopfield.ipynb`, `dense_hopfield.ipynb` and `continuous_hopfield.ipynb` can be used to generate the plots for the [blog](https://ml-jku.github.io/hopfield-layers/).

The input data is contained in the folder `simpsons_faces/` and is a subsample from this [dataset](https://www.kaggle.com/kostastokis/simpsons-faces)

The notebook `continuous_hopfield_pytorch.ipynb` demonstrates the useage of the Hopfield-pytorch layer for the above examples.

## Setup
To install dependencies, execute:
`conda env create -f environment.yml`

To run small initial test, execute in activated environment _hopfield_:
`python models.py`

To run extended experiments, open the according notebooks:
* `classical_hopfield.ipynb`
* `dense_hopfield.ipynb`
* `continuous_hopfield.ipynb`
