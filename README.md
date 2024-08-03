*Hi there! If you found this already, know that this is still a work in progress. I'll have this done by next week*

# Torch-MLP
*A library that should have been created ages ago*

If you are like me, you have created many MLPs (or FFNs) in PyTorch in the past few years. Perhaps you did something like this:

```python
layers = []
layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
for i in range(n_layers - 2):
    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
    if use_bn:
        layers.append(nn.Batchnorm1d(hidden_dim))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
layers.append(nn.Linear(hidden_dim, out_dim))
mlp = nn.Sequential(*layers)
```
I have written this code in various forms so many times. Something as simple as this should not be repeated over and over again by programmers all over the world. **Just try and search for `.append(nn.Linear(` on GitHub and you'll get nearly 30K matches!** That's insane! This package is meant to replace all of that by making MLPs easily configurable. It has a flexible interface so you can build MLPs in the minimum amount of necessary lines. This reduces clutter, reduces errors and will make your life better. 

## How it works
Let's say you want a simple rectangular MLP of depth 3 with batch normalisation and dropout. You can use:
```python
from torch_mlp import MLP

mlp = MLP(
    d_in=784,
    d_hidden=128,
    d_out=10,
    depth=3,
    act=nn.ReLU,
    norm=nn.BatchNorm1d,
    dropout=0.2,
)
```
That's nice! But what if you want to have more fine-grained control over the hidden dimensionality? No problem:

```python
mlp = MLP(
    dims=[784, 128, 64, 32, 16, 10]
    act=nn.ReLU,
    norm=nn.BatchNorm1d,
    dropout=0.2,
)
```
You are free to use any of these options to set the dimensionality of your activations. 

How about more interesting configurations? For every input argument, you can supply an additional argument with the `_params` suffix to supply extra arguments to the MLP creation module. For example, we could create an MLP with Kaiming normal initialisation and initialise all biases to zero. 
```python
mlp = MLP(
    d_in=784,
    d_hidden=128,
    d_out=10,
    depth=5,
    act=nn.ReLU,
    norm=nn.BatchNorm1d,
    weight_init=nn.init.kaiming_normal_,
    bias_init=nn.init.zeros_,
    act_params={"inplace": True},
    weight_init_params={"mode": "fan_in", "nonlinearity": "relu"},
    norm_params={
        "momentum": 0.42,
        "affine": False,
    },
)

```
Don't know the size of the input? No problem! Just use `d_in=None` or `dims=[None, 128, 10]` the MLP will automatically use a `LazyLinear` for the first layer! What if your inputs are indices in a longtensor? No problem! Just set `embed=True`. What if you want to end the network with a sigmoid, because you want to predict between 0 and 1? No problem! Just set `end_act=nn.Sigmoid` and a sigmoid will be added after the final linear layer. 

