# Application of Hopfield-based pooling on Attention-based Deep Multiple Instance Learning

This notebook demonstrates how to apply the Hopfield pooling layer. 
It is based on the PyTorch implementation of the paper [Attention-based Deep Multiple Instance Learning](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) by 
* Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. [arXiv preprint arXiv:1802.04712](https://arxiv.org/pdf/1802.04712.pdf).


## Installation
Download the PyTorch code of Attention-based Deep Multiple Instance Learning (ADMIL) from the accompanying [repository](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) into a directory <code>AttentionDeepMIL</code> on the current directory level ([examples/mnist_bags](.)). Afterwards, line 60 of the file [model.py](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L60) of this repository needs to be modified. Change 
```python 
error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
``` 
to 
```python 
error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
```  

Cell <b>5</b> specifies the parameters defining the data set properties, whereas cell <b>15</b> defines the Hopfield-based pooling network. As a last step, <i>run</i> the notebook.


## Note
* The neural network with Hopfield-based pooling, implemented in cell <b>15</b> of the [mnist_bags_demo.ipynb](mnist_bags_demo.ipynb) notebook is based on the models proposed in [ADMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL).

* The code in the [mnist_bags_demo.ipynb](mnist_bags_demo.ipynb) notebook is based on the [main.py](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/main.py) file from ADMIL.


## Disclaimer
The purpose of this notebook is merely to demonstrate how to use <code>HopfieldPooling</code> layer. In no way it is intended as a comparison of the methods.  
