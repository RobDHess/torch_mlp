import torch
import torch.nn as nn
from typing import Type, Optional, List


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) model with configurable architecture."""

    def __init__(
        self,
        d_in: int = None,
        d_hidden: int = None,
        d_out: int = None,
        depth: int = None,
        dims: List[Optional[int]] = None,
        linear: Type[nn.Module] = nn.Linear,
        norm: Optional[Type[nn.Module]] = None,
        act: Type[nn.Module] = nn.ReLU,
        dropout: Optional[Type[nn.Module]] = nn.Dropout,
        bias: bool = True,
        p: Optional[float] = None,
        embed: bool = False,
        end_linear: bool = True,
        end_act: Optional[Type[nn.Module]] = None,
        linear_kwargs: Optional[dict] = None,
        norm_kwargs: Optional[dict] = None,
        act_kwargs: Optional[dict] = None,
        dropout_kwargs: Optional[dict] = None,
        weight_init: Optional[callable] = None,
        bias_init: Optional[callable] = None,
        weight_kwargs: Optional[dict] = None,
        bias_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the MLP model.

        Parameters:
        d_in (int): Input dimension size.
        d_hidden (int): Hidden layer dimension size.
        d_out (int): Output dimension size.
        depth (int): Number of hidden layers.
        dims (List[Optional[int]]): List of dimensions for each layer.
        linear (Type[nn.Module]): Linear layer class or factory function.
        norm (Optional[Type[nn.Module]]): Normalization layer class or factory function.
        act (Type[nn.Module]): Activation function class or factory function.
        dropout (Optional[Type[nn.Module]]): Dropout layer class or factory function.
        bias (bool): Whether to include bias in linear layers.
        p (Optional[float]): Dropout probability.
        embed (bool): Whether the first layer is an embedding layer.
        end_linear (bool): Whether to include a final linear layer.
        end_act (Optional[Type[nn.Module]]): Activation function for the output layer.
        linear_kwargs (Optional[dict]): Additional arguments for linear layers.
        norm_kwargs (Optional[dict]): Additional arguments for normalization layers.
        act_kwargs (Optional[dict]): Additional arguments for activation functions.
        dropout_kwargs (Optional[dict]): Additional arguments for dropout layers.
        weight_init (Optional[callable]): Function to initialize weights.
        bias_init (Optional[callable]): Function to initialize biases.
        weight_kwargs (Optional[dict]): Additional arguments for weight initialization.
        bias_kwargs (Optional[dict]): Additional arguments for bias initialization.
        """
        super(MLP, self).__init__()
        # Initialize class variables
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.depth = depth
        self.dims = dims
        self.linear = linear
        self.norm = norm
        self.act = act
        self.dropout = dropout
        self.bias = bias
        self.p = p
        self.embed = embed
        self.end_linear = end_linear
        self.end_act = end_act
        self.linear_kwargs = linear_kwargs
        self.norm_kwargs = norm_kwargs
        self.act_kwargs = act_kwargs
        self.dropout_kwargs = dropout_kwargs
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.weight_kwargs = weight_kwargs
        self.bias_kwargs = bias_kwargs

        self.parse_dimensions()
        self.impute_defaults()
        self.check_input()

        self.net = nn.Sequential(*self.build())
        self.initialise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP model.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        return self.net(x)

    def __len__(self) -> int:
        """
        Get the number of layers in the MLP.

        Returns:
        int: Number of layers in the network.
        """
        return len(self.net)

    def __iter__(self):
        """
        Iterate over the layers in the MLP.

        Returns:
        Iterator: Iterator over the layers in the network.
        """
        return iter(self.net)

    def check_input(self) -> None:
        """
        Validate input configurations for the MLP model.

        Ensures that dimensions, dropout probabilities, and other settings are correctly set.
        Raises ValueError or AssertionError if checks fail.
        """
        if self.dims[0] is None and self.embed:
            raise ValueError("d_in must be specified if embedding is used")
        if self.depth is not None:
            assert self.depth > 0, "Depth must be positive"
        if self.d_hidden is not None:
            assert self.d_hidden > 0, "d_hidden must be positive"
        if self.d_out is not None:
            assert self.d_out > 0, "d_out must be positive"
        if self.p is not None:
            assert 0 <= self.p < 1, "p must be in [0, 1)"
        if "p" in self.dropout_kwargs and self.dropout_kwargs["p"] is not None:
            assert 0 <= self.dropout_kwargs["p"] < 1

    def parse_dimensions(self) -> None:
        """
        Parse and validate the dimensions of each layer in the MLP.

        Determines the layer dimensions based on provided parameters or raises
        an error if conflicting configurations are found.
        """
        modes = [
            self.dims is not None,
            (
                self.d_hidden is not None
                and self.d_out is not None
                and self.depth is not None
            ),
        ]

        if sum(modes) != 1:
            raise ValueError(
                "Exactly one of dims or d_hidden, d_out and depth must be specified"
            )

        if self.dims is None:
            self.dims = [self.d_in] + [self.d_hidden] * self.depth + [self.d_out]

        for d in self.dims:
            assert d is None or isinstance(
                d, int
            ), "dims must be a list of integers or None"

    def impute_defaults(self) -> None:
        """
        Set default values for various parameters if they are not explicitly provided.

        Initializes dictionaries for layer arguments and sets bias and dropout defaults.
        """
        if self.linear_kwargs is None:
            self.linear_kwargs = {}
        if self.norm_kwargs is None:
            self.norm_kwargs = {}
        if self.act_kwargs is None:
            self.act_kwargs = {}
        if self.dropout_kwargs is None:
            self.dropout_kwargs = {}
        if self.weight_kwargs is None:
            self.weight_kwargs = {}
        if self.bias_kwargs is None:
            self.bias_kwargs = {}

        if "bias" not in self.linear_kwargs:
            self.linear_kwargs["bias"] = self.bias
        if "p" not in self.dropout_kwargs:
            self.dropout_kwargs["p"] = self.p

        if self.dropout_kwargs["p"] is not None:
            self.dropout_kwargs["p"] = float(self.dropout_kwargs["p"])

    def _linear(self, d_in: int, d_out: int) -> nn.Module:
        """
        Create a linear layer.

        Parameters:
        d_in (int): Input dimension size.
        d_out (int): Output dimension size.

        Returns:
        nn.Module: Linear layer.
        """
        return self.linear(d_in, d_out, **self.linear_kwargs)

    def _norm(self, d_out: int) -> nn.Module:
        """
        Create a normalization layer.

        Parameters:
        d_out (int): Output dimension size.

        Returns:
        nn.Module: Normalization layer.
        """
        return self.norm(d_out, **self.norm_kwargs)

    def _act(self) -> nn.Module:
        """
        Create an activation function.

        Returns:
        nn.Module: Activation function.
        """
        return self.act(**self.act_kwargs)

    def _dropout(self) -> nn.Module:
        """
        Create a dropout layer.

        Returns:
        nn.Module: Dropout layer.
        """
        return self.dropout(**self.dropout_kwargs)

    def _embed(self, d_in: int, d_out: int) -> nn.Embedding:
        """
        Create an embedding layer.

        Parameters:
        d_in (int): Input dimension size.
        d_out (int): Output dimension size.

        Returns:
        nn.Embedding: Embedding layer.
        """
        return nn.Embedding(d_in, d_out)

    def _lazy(self, d_out: int) -> nn.LazyLinear:
        """
        Create a lazy linear layer that infers input dimension at runtime.

        Parameters:
        d_out (int): Output dimension size.

        Returns:
        nn.LazyLinear: Lazy linear layer.
        """
        return nn.LazyLinear(d_out, **self.linear_kwargs)

    def build_layer(
        self,
        d_in: Optional[int],
        d_out: int,
        embed: bool = False,
        lin_only: bool = False,
    ) -> List[nn.Module]:
        """
        Build a layer of the MLP.

        Parameters:
        d_in (Optional[int]): Input dimension size, can be None for lazy layers.
        d_out (int): Output dimension size.
        embed (bool): Whether the layer is an embedding layer.
        lin_only (bool): Whether the layer is only linear without activation or dropout.

        Returns:
        List[nn.Module]: List of layers composing the built layer.
        """
        layer = []
        if embed:
            layer.append(self._embed(d_in, d_out))
        elif d_in is None:
            layer.append(self._lazy(d_out))
        else:
            layer.append(self._linear(d_in, d_out))

        if not lin_only:
            if self.norm is not None:
                layer.append(self._norm(d_out))
            layer.append(self._act())
            if self.dropout is not None:
                p = self.dropout_kwargs.get("p")
                if p is not None and 0 < p < 1:
                    layer.append(self._dropout())
        return layer

    def build(self) -> nn.Sequential:
        """
        Construct the full MLP network by stacking layers.

        The network is built by iterating over the dimensions specified in `dims`.
        Special cases for the first layer (embedding) and the last layer (final linear
        layer without activation/dropout) are handled separately.

        Returns:
        nn.Sequential: A sequential container of the MLP layers.
        """
        net = []
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            # Check first and last layer special cases
            embed = i == 0 and self.embed
            lin_only = i == len(self.dims) - 2 and self.end_linear
            # Build layer
            net.extend(self.build_layer(d_in, d_out, embed, lin_only))
        if self.end_act is not None:
            net.append(self.end_act())
        return nn.Sequential(*net)

    def initialise(self) -> None:
        """
        Initialize the weights and biases of the network.

        Applies the provided weight and bias initialization functions to the layers.
        Lazy layers are skipped until they are instantiated at runtime.
        """
        for module in self.net:
            if isinstance(module, self.linear):
                if isinstance(module, nn.LazyLinear):
                    continue
                if self.weight_init is not None:
                    self.weight_init(module.weight, **self.weight_kwargs)
                if self.bias_init is not None:
                    self.bias_init(module.bias, **self.bias_kwargs)
