# MARNNs Can Learn Generalized Dyck Languages
This repository includes a PyTorch implementation of the paper [Memory-Augmented Recurrent Neural Networks Can Learn Generalized Dyck Languages](https://arxiv.org/abs/1911.03329).


## Dependencies
The code was written in `Python 3.6.4` and requires `PyTorch 1.2.0` and a couple of other dependencies. If you would like to run the code locally, please install PyTorch by following the instructions on http://pytorch.org and then run the following command to install the other required packages, which are listed inside `requirements.txt`:
```
pip install -r requirements.txt
```

## Task and Model Parameters
A probabilistic context-free grammar for Dyck-n can be written as follows:

![Stack-RNN Visualizations](https://github.com/suzgunmirac/marnns/blob/master/visualizations/DyckFormulation.png)

where 0 < p, q < 1 and p + q < 1.

In our code, we represent the paremeters `n`, `p`, and `q` in the above formulation by `num_par`, `p_val`, and `q_val`, respectively:
* `num_par`: Number of parentheses pairs.
* `p_val`: p value.
* `q_val`: q value.

We can further specify the number of samples in the training and test corpura:
* `training_window`: Training set length window.
* `training_size`: Number of samples in the training set.
* `test_size`: Number of samples in the test set.

A single layer Stack-RNN (`stack_rnn_softmax`) with 8 hidden units and 5 dimensional memory is chosen as a default architecture, however we can easily change the model parameters with the following set of arguments:
* `model_name`: Model choice (e.g., `stack_rnn_softmax` for Stack-RNN, `baby_ntm_softmax` for Baby-NTM).
* `n_layers`: Number of hidden layers in the network.
* `n_hidden`: Number of hidden units per layer in the network.
* `memory_size`: Size of the stack/memory.
* `memory_dim`: Dimensionality of the stack/memory.

If a stack/memory-augmented model with a softmax-temperature or Gumbel-softmax decision-gate is used, we can also specify the minimum temperature value and the annealing rate: 
* `temp_min`: Minimum temperature value if a model with `softmax-temp` or `gumbel-softmax` is used.
* `anneal_rate`: Annealing rate.

Learning rate and number of epochs can be specified with the following parameters:
* `lr`: Learning rate.
* `n_epoch`: Number of epochs to be trained.

Finally, we can save and load the model weights by using the following arguments. By default, if `load_path` is not specified, the code trains a model from scratch and then saves the model weights in the `models` folder.
* `save_path`: Path to save the weights of a model after the trainig is completed.
* `load_path`: Path to load the weights of a pre-trained model.


## Training
To train a single layer Stack-RNN model with 8 hidden units and 1-dimensional stack to learn the Dyck-2 language:

`python main.py --num_par 2 --model_name stack_rnn_softmax --n_hidden 8 --memory_dim 5 --save_path models/stack_rnn_model_weights.pth`

To train a single layer Baby Neural Turing Machine (Baby-NTM) with 8 hidden units and 5-dimensional memory to learn the Dyck-3 language:

`python main.py --num_par 3 --model_name baby_ntm_softmax --n_hidden 8 --memory_dim 5 --save_path models/baby_ntm_model_weights.pth`

## Evaluation
To evaluate the performance of the Baby-NTM model trained in the previous section, we can simply write:

`python main.py --num_par 3 --model_name baby_ntm_softmax --n_hidden 8 --memory_dim 5 --load_path models/baby_ntm_model_weights.pth`


## Visualizations
### Stack-RNN
The figure below demonstrates how a Stack-RNN model with 8 hidden units and 1-dimensional stack learns to control the strengths of the _PUSH_ and _POP_ operations on a single example as it is to trained to recognize the Dyck-2 language.

![Stack-RNN Visualizations](https://github.com/suzgunmirac/marnns/blob/master/visualizations/StackRNN_Weights.gif)

### Baby-NTM
The figure below demonstrates how a Baby-NTM model with 12 hidden units and 5-dimensional memeory learns to control the strengths of the memory operations on a single example as it is to trained to recognize the Dyck-2 language.

![Baby-NTM Visualizations](https://github.com/suzgunmirac/marnns/blob/master/visualizations/BabyNTM_Weights.gif)

## Notebooks
Please feel free to take a look at our notebooks if you would like to run the code in an interactive session or plot some of the figures in our paper by yourself.

## Related Work
* [`stacknn-core`](https://github.com/viking-sudo-rm/stacknn-core): [William Merrill](https://lambdaviking.com/) has recently released a PyTorch implementation of the differentiable stack and queue models introduced in the paper [Learning to Transduce with Unbounded Memory](http://papers.nips.cc/paper/5648-learning-to-transduce-with-unbounded-memory.pdf). The code is based on the paper [Context-Free Transductions with Neural Stacks](https://arxiv.org/pdf/1809.02836.pdf).