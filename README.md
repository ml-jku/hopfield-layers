# Hopfield Networks is All You Need

_Hubert Ramsauer<sup>1</sup>,
Bernhard Schäfl<sup>1</sup>,
Johannes Lehner<sup>1</sup>,
Philipp Seidl<sup>1</sup>,
Michael Widrich<sup>1</sup>,
Lukas Gruber<sup>1</sup>,
Markus Holzleitner<sup>1</sup>,
Milena Pavlović<sup>3, 4</sup>,
Geir Kjetil Sandve<sup>4</sup>,
Victor Greiff<sup>3</sup>,
David Kreil<sup>2</sup>,
Michael Kopp<sup>2</sup>,
Günter Klambauer<sup>1</sup>,
Johannes Brandstetter<sup>1</sup>,
Sepp Hochreiter<sup>1, 2</sup>_

<sup>1</sup> ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria  
<sup>2</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)  
<sup>3</sup> Department of Immunology, University of Oslo, Norway  
<sup>4</sup> Department of Informatics, University of Oslo, Norway  

---

The transformer and BERT models pushed the performance on NLP tasks to new levels via their attention mechanism. We show
that this attention mechanism is the update rule of a modern Hopfield network with continuous states. This new Hopfield
network can store exponentially (with the dimension) many patterns,converges with one update, and has exponentially
small retrieval errors. The number of stored patterns must be traded off against convergence speed and retrieval error.
The new Hopfield network has three types of energy minima (fixed points of the update):

1. global fixed point averaging over all patterns,
2. metastable states averaging over a subset of patterns, and
3. fixed points which store a single pattern.

Transformers learn an attention mechanism by constructing an embedding of patterns and queries into an associative
space. Transformer and BERT models operate in their first layers preferably in the global averaging regime, while they
operate in higher layers in metastable states. The gradient in transformers is maximal in the regime of metastable
states, is uniformly distributed when averaging globally, and vanishes when a fixed point is near a stored pattern.
Based on the Hopfield network interpretation, we analyzed learning of transformer and BERT architectures. Learning starts
with attention heads that average and then most of them switch to metastable states. However, the majority of heads in
the first layers still averages and can be replaced by averaging operations like the Gaussian weighting that we propose.
In contrast, heads in the last layers steadily learn and seem to use metastable states to collect information created in
lower layers. These heads seem a promising target for improving transformers. Neural networks that integrate Hopfield
networks that are equivalent to attention heads outperform other methods on immune repertoire classification, where the
Hopfield net stores several hundreds of thousands of patterns.

With _this_ repository, we provide a PyTorch implementation of a new layer called
“Hopfield” which allows to equip deep learning architectures with Hopfield networks as new memory concepts.

## Requirements

The software was developed and tested on the following 64-bit operating systems:

- CentOS Linux release 8.1.1911 (Core)
- MacOS 10.15.5 (Catalina)

As the development environment, [Python](https://www.python.org) 3.8.3 in combination with [pytorch](https://pytorch.org) 1.5.0 was used. More details on how to install PyTorch are available on the [official project page](https://pytorch.org).

## Usage

To get up and running with Hopfield-based networks, only <i>one</i> argument needs to be set, the size (depth) of the input.

```python
hopfield = Hopfield(input_size=...)
```

It is also possible to replace commonly used pooling functions with a Hopfield-based one. Internally, a <i>state pattern</i> is trained, which in turn is used to compute pooling weights with respect to the input.

```python
hopfield_pooling = HopfieldPooling(input_size=...)
```

The usage is as <i>simple</i> as with the main module, but equally <i>powerful</i>.

## Examples

Generally, the Hopfield layer is designed to be used to implement or to substitute different layers like:

- <b>Pooling layers:</b> We consider the Hopfield layer as a pooling layer if only one static state (query) pattern exists. Then, it is de facto a pooling over the sequence, which results from the softmax values applied on the stored patterns. Therefore, our Hopfield layer can act as a pooling layer.

- <b>Permutation equivariant layers:</b> Our Hopfield layer can be used as a plug-in replacement for permutation equivariant layers. Since the Hopfield layer is an associative memory it assumes no dependency between the input patterns.

- <b>GRU & LSTM layers:</b> Our Hopfield layer can be used as a plug-in replacement for GRU & LSTM layers. Optionally, for substituting GRU & LSTM layers, positional encoding might be considered.

- <b>Attention layers:</b>  Our Hopfield layer can act as an attention layer, where state (query) and stored (key) patterns are different, and need to be associated.

The folder [examples](examples/) contains multiple demonstrations on how to use the <code>Hopfield</code> as well as the <code>HopfieldPooling</code> modules. To successfully run the contained [JuPyter Notebooks](https://jupyter.org), additional third-party modules like [pandas](https://pandas.pydata.org) and [seaborn](https://seaborn.pydata.org) are required.

- [Bit Pattern Set](examples/bit_pattern_demo.ipynb): The dataset of this demonstration falls into the category of <i>binary classification</i> tasks in the domain of <i>Multiple Instance Learning (MIL)</i> problems. Each bag comprises a collection of bit pattern instances, wheres each instance is a sequence of <b>0s</b> and <b>1s</b>. The positive class has specific bit patterns injected, which are absent in the negative one. This demonstration shows, that <code>Hopfield</code> and <code>HopfieldPooling</code> are capable of learning and filtering each bag with respect to the class-defining bit patterns.

- [Latch Sequence Set](examples/latch_sequence_demo.ipynb): We study an easy example of learning long-term dependencies by using a simple <i>latch task</i>, see [Hochreiter and Mozer](https://link.springer.com/chapter/10.1007/3-540-44668-0_92). The essence of this task is that a sequence of inputs is presented, beginning with one of two symbols, <b>A</b> or <b>B</b>, and after a variable number of time steps, the model has to output a corresponding symbol. Thus, the task requires memorizing the original input over time. It has to be noted, that both class-defining symbols must only appear at the first position of a sequence. This task was specifically designed to demonstrate the capability of recurrent neural networks to capture long term dependencies. This demonstration shows, that <code>Hopfield</code> and <code>HopfieldPooling</code> adapt extremely fast to this specific task, concentrating only on the first entry of the sequence.

## Disclaimer

Some implementations of this repository are based on existing ones of the official [PyTorch Repository v1.5.0](https://github.com/pytorch/pytorch/tree/v1.5.0) and accordingly extended and modified. In the following, the involved parts are listed:

- The implementation of [HopfieldCore](modules/activation.py#L11) is based on the implementation of [MultiheadAttention](https://github.com/pytorch/pytorch/blob/4ff3872a2099993bf7e8c588f7182f3df777205b/torch/nn/modules/activation.py#L696).
- The implementation of [hopfield_core_forward](modules/functional.py#L8) is based on the implementation of [multi_head_attention_forward](https://github.com/pytorch/pytorch/blob/4ff3872a2099993bf7e8c588f7182f3df777205b/torch/nn/functional.py#L3692).
- The implementation of [HopfieldEncoderLayer](modules/transformer.py#L12) is based on the implementation of [TransformerEncoderLayer](https://github.com/pytorch/pytorch/blob/4ff3872a2099993bf7e8c588f7182f3df777205b/torch/nn/modules/transformer.py#L240).
- The implementation of [HopfieldDecoderLayer](modules/transformer.py#L88) is based on the implementation of [TransformerDecoderLayer](https://github.com/pytorch/pytorch/blob/4ff3872a2099993bf7e8c588f7182f3df777205b/torch/nn/modules/transformer.py#L303).