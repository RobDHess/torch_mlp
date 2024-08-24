import pytest
import torch
import torch.nn as nn
from torch_mlp import MLP

d_in = 5
d_hidden = 6
d_out = 7
depth = 3
dims = [d_in] + [d_hidden] * depth + [d_out]
batch_size = 2


def test_mlp_creation():
    # Test basic initialization
    mlp1 = MLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out, depth=depth)
    mlp2 = MLP(dims=dims)
    assert mlp1.dims == mlp2.dims


def test_mlp_forward():
    # Test forward pass
    mlp = MLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out, depth=depth)
    x = torch.randn(batch_size, d_in)  # Batch size 1, input size 10
    output = mlp(x)
    assert output.shape == (
        batch_size,
        d_out,
    )  # Output should match the output dimension


def test_mlp_invalid_dimensions():
    # Test initialization with invalid dimensions
    with pytest.raises(AssertionError):
        MLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out, depth=0)  # depth cannot be 0

    with pytest.raises(AssertionError):
        MLP(dims=[None, "20", 5])  # dims should be a list of integers or None


def test_full_mlp():
    # Test MLP with custom layers (e.g., no activation)
    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        act=nn.GELU,
        norm=nn.BatchNorm1d,
        p=0.1,
    )
    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)


def test_embed():
    # Test embedding layer
    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        act=nn.GELU,
        norm=nn.BatchNorm1d,
        p=0.1,
        embed=True,
    )
    x = torch.randint(0, d_in, (batch_size,))
    output = mlp(x)

    assert output.shape == (batch_size, d_out)


def test_lazy():
    # Test lazy initialization
    mlp = MLP(
        d_in=None,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        act=nn.GELU,
        norm=nn.BatchNorm1d,
        p=0.1,
    )
    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)


def test_kwargs():
    linear_kwargs = {"bias": False}
    norm_kwargs = {"momentum": 0.5}
    act_kwargs = {"negative_slope": 0.1}
    dropout_kwargs = {"p": 0.1}

    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        norm=nn.BatchNorm1d,
        act=nn.LeakyReLU,
        linear_kwargs=linear_kwargs,
        dropout_kwargs=dropout_kwargs,
        norm_kwargs=norm_kwargs,
        act_kwargs=act_kwargs,
    )

    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)

    assert (mlp.net[0].bias is not None) == linear_kwargs["bias"]
    assert mlp.net[1].momentum == norm_kwargs["momentum"]
    assert mlp.net[2].negative_slope == act_kwargs["negative_slope"]
    assert mlp.net[3].p == dropout_kwargs["p"]


def test_init():
    weight_init = nn.init.kaiming_normal_
    bias_init = nn.init.constant_

    weight_kwargs = {"a": 0, "mode": "fan_in", "nonlinearity": "leaky_relu"}
    bias_kwargs = {"val": 12}

    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        weight_init=weight_init,
        bias_init=bias_init,
        weight_kwargs=weight_kwargs,
        bias_kwargs=bias_kwargs,
    )

    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)


def test_end_act():
    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        end_act=lambda: nn.Softmax(dim=-1),
    )

    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)
    assert torch.allclose(output.sum(dim=-1), torch.ones(batch_size))


def test_override():
    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        act=nn.ReLU,
        norm=nn.LayerNorm,
        p=0.1,
        dropout_kwargs={"p": 0.2},
        bias=False,
        linear_kwargs={"bias": True},
    )

    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)

    assert mlp.net[3].p == 0.2
    assert mlp.net[0].bias is not None


def test_cuda():
    if torch.cuda.is_available():
        mlp = MLP(
            d_in=d_in,
            d_hidden=d_hidden,
            d_out=d_out,
            depth=depth,
        )
        mlp = mlp.to("cuda")
        x = torch.randn(batch_size, d_in, device="cuda")
        output = mlp(x)
        assert output.shape == (batch_size, d_out)
        assert output.device == torch.device("cuda")


def test_no_end_linear():
    mlp = MLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
        depth=depth,
        end_linear=False,
    )
    x = torch.randn(batch_size, d_in)
    output = mlp(x)
    assert output.shape == (batch_size, d_out)
    assert isinstance(mlp.net[-1], nn.ReLU)
