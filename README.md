# Torch-MLP
*A library that should have been created ages ago*

If you are like me, you have created many MLPs (or FFNs or whatever) in the past few years. Perhaps you did something like this:

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
I have written this code in various forms so many times. And it's bullshit. Something as simple as this should not be repeated over and over again by programmers all over the world. Just try and search for `.append(nn.Linear(` on GitHub and you'll get tens of thousands of matches. This package is meant to replace all of that by making MLPs easily configurable. It has a flexible interface so you can build MLPs in the minimum amount of necessary lines. This reduces clutter, reduces errors and will make your life better. 
