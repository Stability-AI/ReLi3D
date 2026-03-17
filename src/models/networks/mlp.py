from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.ops import BiasLayer, MultiplierLayer, get_activation_module
from src.utils.typing import List, Optional, Union


@dataclass
class HeadSpec:
    name: FieldName
    out_channels: int
    n_hidden_layers: int
    output_activation: Optional[str] = None
    init_weights: Optional[str] = None
    init_bias: Optional[str] = None
    out_bias: Optional[Union[float, List[float]]] = None
    out_multiplier: Optional[Union[float, List[float]]] = None


class MultiHeadMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 0
        n_neurons: int = 0
        n_hidden_layers_share: int = 0
        heads: List[HeadSpec] = field(default_factory=list)
        activation: str = "relu"
        bias: bool = True
        weight_init: Optional[str] = "kaiming_uniform"
        bias_init: Optional[str] = None
        chunk_mode: Optional[str] = None
        chunk_size: int = -1
        only_heads: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if self.cfg.only_heads:
            self.shared_layers = torch.nn.Identity()
        else:
            shared_layers = [
                self.make_linear(
                    self.cfg.in_channels,
                    self.cfg.n_neurons,
                    bias=self.cfg.bias,
                    weight_init=self.cfg.weight_init,
                    bias_init=self.cfg.bias_init,
                ),
                get_activation_module(self.cfg.activation),
            ]
            for i in range(self.cfg.n_hidden_layers_share - 1):
                shared_layers += [
                    self.make_linear(
                        self.cfg.n_neurons,
                        self.cfg.n_neurons,
                        bias=self.cfg.bias,
                        weight_init=self.cfg.weight_init,
                        bias_init=self.cfg.bias_init,
                    ),
                    get_activation_module(self.cfg.activation),
                ]
            self.shared_layers = nn.Sequential(*shared_layers)

        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [
                    self.make_linear(
                        (
                            self.cfg.in_channels
                            if i == 0 and self.cfg.only_heads
                            else self.cfg.n_neurons
                        ),
                        self.cfg.n_neurons,
                        bias=self.cfg.bias,
                        weight_init=self.cfg.weight_init,
                        bias_init=self.cfg.bias_init,
                    ),
                    get_activation_module(self.cfg.activation),
                ]
            head_layers.append(
                self.make_linear(
                    self.cfg.n_neurons,
                    head.out_channels,
                    bias=self.cfg.bias,
                    weight_init=self.cfg.weight_init,
                    bias_init=self.cfg.bias_init,
                ),
            )
            if head.out_bias is not None:
                if isinstance(head.out_bias, float) or isinstance(head.out_bias, list):
                    head_layers.append(BiasLayer(head.out_bias))
            if head.out_multiplier is not None:
                if isinstance(head.out_multiplier, float) or isinstance(
                    head.out_multiplier, list
                ):
                    head_layers.append(MultiplierLayer(head.out_multiplier))
            head_layers.append(
                get_activation_module(head.output_activation),
            )
            # ModuleDict only supports `str` keys, due to needing `str` representations
            # for the state_dict (and repr not always being reproducible across runs).
            # Here, we avoid the automatic hidden cast in the `ModuleDict` ctor.
            heads[str(head.name)] = nn.Sequential(*head_layers)

            def get_number(v):
                try:
                    return float(v)
                except ValueError:
                    return v

            if head.init_weights is not None:
                split_str = head.init_weights.split("/")
                method, args = split_str[0], split_str[1:]
                args = [get_number(arg) for arg in args]
                for name, param in heads[head.name].named_parameters():
                    if "weight" in name:
                        getattr(torch.nn.init, method)(param, *args)
            if head.init_bias is not None:
                split_str = head.init_bias.split("/")
                method, args = split_str[0], split_str[1:]
                # Test to check if an element in args is a number and if so convert it to float
                args = [get_number(arg) for arg in args]
                for name, param in heads[head.name].named_parameters():
                    if "bias" in name:
                        getattr(torch.nn.init, method)(param, *args)

        self.heads = nn.ModuleDict(heads)

        if self.cfg.chunk_mode is not None:
            assert self.cfg.chunk_size > 0

        if self.cfg.chunk_mode is None:
            self.chunk_fn = direct_batch_processing
        elif self.cfg.chunk_mode == "deferred":
            self.chunk_fn = apply_batch_deferral
        elif self.cfg.chunk_mode == "checkpointing":
            self.chunk_fn = apply_batch_checkpointing
        else:
            raise NotImplementedError(f"Unknown chunk mode `{self.cfg.chunk_mode}`.")

    def make_linear(
        self,
        dim_in,
        dim_out,
        bias=True,
        weight_init=None,
        bias_init=None,
    ):
        layer = nn.Linear(dim_in, dim_out, bias=bias)

        if weight_init is None:
            pass
        elif weight_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        else:
            raise NotImplementedError

        if bias:
            if bias_init is None:
                pass
            elif bias_init == "zero":
                torch.nn.init.zeros_(layer.bias)
            else:
                raise NotImplementedError

        return layer

    def keys(self):
        return self.heads.keys()

    def get_head_config(self, head_name: FieldName) -> HeadSpec:
        for head in self.cfg.heads:
            if head.name == str(head_name):
                return head
        raise ValueError(f"Head {head_name} not found in MLP.")

    def forward(
        self,
        outputs: OutputsType,
        include: Optional[List] = None,
        exclude: Optional[List] = None,
    ):
        x = outputs[Names.TOKEN]
        inp_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if self.cfg.only_heads:
            shared_features = x
        else:
            shared_features = self.chunk_fn(self.shared_layers, x, self.cfg.chunk_size)

        shared_features = shared_features.reshape(*inp_shape, -1)

        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude.")
        if include is not None:
            heads = [h for h in self.cfg.heads if h.name in [str(i) for i in include]]
        elif exclude is not None:
            heads = [
                h for h in self.cfg.heads if h.name not in [str(i) for i in exclude]
            ]
        else:
            heads = self.cfg.heads

        if self.cfg.only_heads:
            out = {
                head.name: self.chunk_fn(
                    self.heads[head.name], shared_features, self.cfg.chunk_size
                )
                for head in heads
            }
            # reshape to the original shape
            out = {k: v.reshape(*inp_shape, -1) for k, v in out.items()}
        else:
            out = {head.name: self.heads[head.name](shared_features) for head in heads}

        return out

    def consumed_keys(self):
        return super().consumed_keys().union({Names.TOKEN})

    def produced_keys(self):
        return super().produced_keys().union({head.name for head in self.cfg.heads})


def direct_batch_processing(model, x, chunk_size):
    return model(x)


def apply_batch_deferral(model, x, chunk_size):
    """A wrapper around DeferredFunc for ease of calling.

    To properly interface with pytorch paradigms, DeferredFunc's backward pass can't directly
    fill in parameter gradients; it *needs* to treat the parameters as inputs and return an
    explicit parameter list for them (as pytorch doesn't support returning a gradient structure
    for an entire model currently). This wrapper is mostly so we can just write
        `DeferredFunc.apply(model, x, chunk_size)`
    instead of
        `DeferredFunc.apply(model, x, chunk_size, model.parameters())`.
    """
    return DeferredFunc.apply(model, x, chunk_size, *model.parameters())


class DeferredFunc(torch.autograd.Function):
    """Splits model calls across the batch dimension, and backpropagates them separately.

    Most importantly, recalculates the output of the model for each chunk, drastically
    reducing peak memory footprint under the assumption of high-dimensional input and
    high hidden state dimensionality.
    """

    @staticmethod
    def forward(ctx, model, x, chunk_size, *model_parameters):
        """We require a dummy input with the model parameters.

        This is to cleanly interface with PyTorch's graph paradigm and return gradients
        for these parameters as part of `backward` instead of just filling them in in-place.
        As PyTorch does not traverse lists or dicts for these inputs (or e.g. model parameters),
        we pass the model parameters as varargs.
        """
        model_copy = deepcopy(model)
        model_copy.requires_grad_(False)

        ret = []
        x_split = torch.split(x, chunk_size, dim=0)

        # We can't use torch.no_grad() here, as we need to retain a functional graph
        # for the outputs, we only want to keep the model out of the graph.
        with torch.no_grad():
            for cur_x in x_split:
                ret.append(model_copy(cur_x))

        ctx.model = model
        ctx.save_for_backward(x.detach(), torch.as_tensor(chunk_size))

        ret = torch.cat(ret, dim=0)

        return ret

    @staticmethod
    def backward(ctx, grad_output):
        model = ctx.model
        x, chunk_size = ctx.saved_tensors
        chunk_size = chunk_size.item()

        model_copy = deepcopy(model)

        x_split = torch.split(x, chunk_size, dim=0)
        grad_output_split = torch.split(grad_output, chunk_size, 0)
        grad_input_split = []

        with torch.set_grad_enabled(True):
            model_copy.requires_grad_(True)
            model_copy.zero_grad()
            for cur_x, cur_grad_output in zip(x_split, grad_output_split):
                cur_x.requires_grad_(True)
                cur_y = model_copy(cur_x)
                cur_y.backward(cur_grad_output)

                grad_input_split.append(cur_x.grad.clone())

        grad_input = torch.cat(grad_input_split, dim=0)

        param_gradients = [
            param_copy.grad.clone() for param_copy in model_copy.parameters()
        ]

        return None, grad_input, None, *param_gradients


def apply_batch_checkpointing(func, x, chunk_size):
    # use gradient checkpointing to save VRAM
    if chunk_size >= len(x):
        return torch.utils.checkpoint.checkpoint(func, x, use_reentrant=False)

    x_split = torch.split(x, chunk_size, dim=0)

    def cat_and_query(y_all, x):
        return torch.cat([y_all, func(x)])

    y_all = func(x_split[0])
    for cur_x in x_split[1:]:
        y_all = torch.utils.checkpoint.checkpoint(
            cat_and_query, y_all, cur_x, use_reentrant=False
        )

    return y_all
