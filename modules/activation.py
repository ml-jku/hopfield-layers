import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Linear, Module, Parameter
from typing import Optional

from .functional import hopfield_core_forward


class HopfieldCore(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See references: "Hopfield Networks is All You Need" and
                    "Attention Is All You Need" (on which this implementation is partly based on).

    .. math::
        \text{HopfieldHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> hopfield_attn = HopfieldCore(embed_dim, num_heads)
        >>> attn_output, attn_output_weights, attn_matrix = hopfield_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self,
                 embed_dim,                      # type: int
                 num_heads,                      # type: int
                 dropout=0.0,                    # type: float
                 bias=True,                      # type: bool
                 add_bias_kv=False,              # type: bool
                 add_zero_attn=False,            # type: bool
                 kdim=None,                      # type: Optional[int]
                 vdim=None,                      # type: Optional[int]

                 head_dim=None,                  # type: Optional[int]
                 out_dim=None,                   # type: Optional[int]
                 disable_out_projection=False,   # type: bool
                 key_as_static=False,            # type: bool
                 query_as_static=False,          # type: bool
                 value_as_static=False,          # type: bool
                 normalise_pattern=False,        # type: bool
                 normalise_pattern_affine=False  # type: bool
                 ):
        super(HopfieldCore, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        assert (type(key_as_static) == bool) and (type(query_as_static) == bool) and (type(value_as_static) == bool)
        self.key_as_static, self.query_as_static, self.value_as_static = key_as_static, query_as_static, value_as_static
        num_non_static = 3 - (self.key_as_static + self.query_as_static + self.value_as_static)
        assert 0 <= num_non_static < 4

        self.num_heads = num_heads
        self.dropout = dropout
        if head_dim is None:
            self.head_dim = embed_dim // num_heads
            assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads."
        else:
            assert head_dim > 0, "dimension of the association space has to be positive."
            self.head_dim = head_dim
        self.virtual_hopfield_dim = self.num_heads * self.head_dim

        self.__disable_out_projection = disable_out_projection
        self.out_dim = embed_dim if out_dim is None else out_dim
        assert disable_out_projection or (self.out_dim > 0), "output projection dimension has to be positive."

        if normalise_pattern_affine:
            assert normalise_pattern, "affine pattern normalisation without pattern normalisation has no effect."
            self.p_norm_weight = Parameter(torch.Tensor(head_dim))
            self.p_norm_bias = Parameter(torch.Tensor(head_dim))
        else:
            self.register_parameter('p_norm_weight', None)
            self.register_parameter('p_norm_bias', None)
        self.__normalise_pattern = normalise_pattern

        if self._qkv_same_embed_dim is False:
            if query_as_static:
                self.register_parameter('q_proj_weight', None)
            else:
                self.q_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, embed_dim))
            if key_as_static:
                self.register_parameter('k_proj_weight', None)
            else:
                self.k_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, self.kdim))
            if value_as_static:
                self.register_parameter('v_proj_weight', None)
            else:
                self.v_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            if num_non_static > 0:
                self.in_proj_weight = Parameter(torch.empty(num_non_static * self.virtual_hopfield_dim, embed_dim))
            else:
                self.register_parameter('in_proj_weight', None)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias and (num_non_static > 0):
            self.in_proj_bias = Parameter(torch.empty(num_non_static * self.virtual_hopfield_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        if disable_out_projection:
            self.register_parameter('out_proj', None)
        else:
            self.out_proj = Linear(self.virtual_hopfield_dim, self.out_dim, bias=bias)

        self.bias_k, self.bias_v = None, None
        if add_bias_kv:
            if not key_as_static:
                self.bias_k = Parameter(torch.empty(1, 1, self.virtual_hopfield_dim))
            if not value_as_static:
                self.bias_v = Parameter(torch.empty(1, 1, self.virtual_hopfield_dim))
            assert not (self.bias_k is None and self.bias_v is None), r'cannot set key/value bias if both are static.'

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self.p_norm_weight is not None:
            nn.init.ones_(self.p_norm_weight)
            nn.init.zeros_(self.p_norm_bias)

        if self._qkv_same_embed_dim and (self.in_proj_weight is not None):
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            if self.q_proj_weight is not None:
                nn.init.xavier_uniform_(self.q_proj_weight)
            if self.k_proj_weight is not None:
                nn.init.xavier_uniform_(self.k_proj_weight)
            if self.v_proj_weight is not None:
                nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            if not self.__disable_out_projection:
                nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        super(HopfieldCore, self).__setstate__(state)

    def forward(self,
                query,                         # type: Tensor
                key,                           # type: Tensor
                value,                         # type: Tensor
                key_padding_mask=None,         # type: Optional[Tensor]
                need_weights=True,             # type: bool
                attn_mask=None,                # type: Optional[Tensor]

                scaling=None,                  # type: Optional[Tensor]
                update_steps_max=0,            # type: Optional[int]
                update_steps_eps=1e-4,         # type: float
                return_raw_associations=False  # type: bool
                ):
        # type: (...) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
                See "Hopfield Networks is All You Need" for more details in the setting of Hopfield networks.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

            scaling: scaling of association heads, often represented as beta (one entry per head).
            update_steps_max: maximum count of association update steps (None equals to infinity).
            update_steps_eps: minimum difference threshold between two consecutive association update steps.
            return_raw_associations: return raw association (softmax) values, unmodified.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length.

            - scaling: :math:`(num_heads,)`, where num_heads is the amount of heads.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
            - attn_raw: :math:``(N, num_heads, L, S)`, where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        assert self.query_as_static or (query.shape[2] == self.embed_dim), \
            f'query shape[2] of {query.shape[2]} invalid, needs to be {self.embed_dim}.'
        assert (not self.query_as_static) or (self.query_as_static and query.shape[2] == self.head_dim), \
            f'query shape[2] of {query.shape[2]} invalid, needs to be {self.head_dim}'

        assert self.key_as_static or (key.shape[2] == self.kdim), \
            f'key shape[2] of {key.shape[2]} invalid, needs to be {self.kdim}.'
        assert (not self.key_as_static) or (self.key_as_static and key.shape[2] == self.head_dim), \
            f'key shape[2] of {key.shape[2]} invalid, needs to be {self.head_dim}'

        assert self.value_as_static or (value.shape[2] == self.vdim), \
            f'value shape[2] of {value.shape[2]} invalid, needs to be {self.vdim}.'
        assert (not self.value_as_static) or (self.value_as_static and value.shape[2] == self.head_dim), \
            f'value shape[2] of {value.shape[2]} invalid, needs to be {self.head_dim}'

        out_weights, out_bias = None, None
        if not self.__disable_out_projection:
            out_weights, out_bias = self.out_proj.weight, self.out_proj.bias

        if not self._qkv_same_embed_dim:
            return hopfield_core_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_weights, out_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,

                key_as_static=self.key_as_static, query_as_static=self.query_as_static,
                value_as_static=self.value_as_static, normalise_pattern=self.__normalise_pattern,
                p_norm_weight=self.p_norm_weight, p_norm_bias=self.p_norm_bias,
                head_dim=self.head_dim, scaling=scaling,
                update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
                return_raw_associations=return_raw_associations)
        else:
            return hopfield_core_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_weights, out_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,

                key_as_static=self.key_as_static, query_as_static=self.query_as_static,
                value_as_static=self.value_as_static, normalise_pattern=self.__normalise_pattern,
                p_norm_weight=self.p_norm_weight, p_norm_bias=self.p_norm_bias,
                head_dim=self.head_dim, scaling=scaling,
                update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
                return_raw_associations=return_raw_associations)
