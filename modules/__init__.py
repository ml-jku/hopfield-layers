import torch
import torch.nn as nn

from math import sqrt
from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple, Union

from .activation import HopfieldCore


class Hopfield(Module):
    """
    Module with underlying Hopfield association.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,

                 normalise_stored_pattern: bool = True,
                 normalise_stored_pattern_affine: bool = True,
                 normalise_state_pattern: bool = True,
                 normalise_state_pattern_affine: bool = True,
                 normalise_pattern_projection: bool = True,
                 normalise_pattern_projection_affine: bool = True,
                 normalise_hopfield_space: bool = False,
                 normalise_hopfield_space_affine: bool = False,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False
                 ):
        """
        Initialise new instance of a Hopfield module.

        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param normalise_stored_pattern: apply normalisation on stored patterns
        :param normalise_stored_pattern_affine: additionally enable affine normalisation of stored patterns
        :param normalise_state_pattern: apply normalisation on state patterns
        :param normalise_state_pattern_affine: additionally enable affine normalisation of state patterns
        :param normalise_pattern_projection: apply normalisation on the pattern projection
        :param normalise_pattern_projection_affine: additionally enable affine normalisation of pattern projection
        :param normalise_hopfield_space: enable normalisation of patterns in the Hopfield space
        :param normalise_hopfield_space_affine: additionally enable affine normalisation of patterns in Hopfield space
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        """
        super(Hopfield, self).__init__()
        assert type(batch_first) == bool, f'"batch_first" needs to be a boolean, not {type(batch_first)}.'
        assert (association_activation is None) or (type(association_activation) == str)

        # Initialise Hopfield association module.
        self.association_core = HopfieldCore(
            embed_dim=input_size, num_heads=num_heads, dropout=dropout, bias=input_bias,
            add_bias_kv=concat_bias_pattern, add_zero_attn=add_zero_association, kdim=stored_pattern_size,
            vdim=pattern_projection_size, head_dim=hidden_size, out_dim=output_size,
            disable_out_projection=disable_out_projection, key_as_static=stored_pattern_as_static,
            query_as_static=state_pattern_as_static, value_as_static=pattern_projection_as_static,
            normalise_pattern=normalise_hopfield_space, normalise_pattern_affine=normalise_hopfield_space_affine)
        self.association_activation = None
        if association_activation is not None:
            self.association_activation = getattr(torch, association_activation, None)

        # Initialise stored pattern normalisation.
        self.norm_stored_pattern = None
        if normalise_stored_pattern_affine:
            assert normalise_stored_pattern, "affine normalisation without normalisation has no effect."
        if normalise_stored_pattern:
            self.norm_stored_pattern = nn.LayerNorm(
                normalized_shape=self.hidden_size if stored_pattern_as_static else self.association_core.kdim,
                elementwise_affine=normalise_stored_pattern_affine)

        # Initialise state pattern normalisation.
        self.norm_state_pattern = None
        if normalise_state_pattern_affine:
            assert normalise_state_pattern, "affine normalisation without normalisation has no effect."
        if normalise_state_pattern:
            self.norm_state_pattern = nn.LayerNorm(
                normalized_shape=self.hidden_size if state_pattern_as_static else self.association_core.embed_dim,
                elementwise_affine=normalise_state_pattern_affine)

        # Initialise pattern projection normalisation.
        self.norm_pattern_projection = None
        if normalise_pattern_projection_affine:
            assert normalise_pattern_projection, "affine normalisation without normalisation has no effect."
        if normalise_pattern_projection:
            self.norm_pattern_projection = nn.LayerNorm(
                normalized_shape=self.hidden_size if pattern_projection_as_static else self.association_core.vdim,
                elementwise_affine=normalise_pattern_projection_affine)

        # Initialise remaining auxiliary properties.
        assert self.association_core.head_dim > 0, f'invalid hidden dimension encountered.'
        self.__batch_first = batch_first
        self.__scaling = (1.0 / sqrt(self.association_core.head_dim)) if scaling is None else scaling
        self.__update_steps_max = update_steps_max
        self.__update_steps_eps = update_steps_eps

    def _maybe_transpose(self, *args: Tuple[Tensor, ...]) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Eventually transpose specified data.

        :param args: tensors to eventually transpose (dependent on the state of "batch_first")
        :return: eventually transposed tensors
        """
        if self.__batch_first:
            transposed_result = tuple(_.transpose(0, 1) for _ in args)
            return transposed_result[0] if len(transposed_result) == 1 else transposed_result
        return args

    def _associate(self, data: Union[Tensor, Tuple[Tensor, Tensor, Tensor]], return_raw_associations: bool = False,
                   stored_pattern_padding_mask: Optional[Tensor] = None,
                   association_mask: Optional[Tensor] = None) -> Tuple[Optional[Tensor], ...]:
        """
        Apply Hopfield association module on specified data.

        :param data: data to be processed by Hopfield core module
        :return: Hopfield-processed input data
        """
        assert (type(data) == Tensor) or ((type(data) == tuple) and (len(data) == 3)), \
            r'either one tensor to be used as "stored pattern", "state pattern" and' \
            r' "pattern_projection" must be provided, or three separate ones.'
        if type(data) == Tensor:
            stored_pattern, state_pattern, pattern_projection = data, data, data
        else:
            stored_pattern, state_pattern, pattern_projection = data

        # Optionally transpose data.
        stored_pattern, state_pattern, pattern_projection = self._maybe_transpose(
            stored_pattern, state_pattern, pattern_projection)

        # Optionally apply stored pattern normalisation.
        if self.norm_stored_pattern is not None:
            stored_pattern = self.norm_stored_pattern.forward(input=stored_pattern.reshape(
                shape=(-1, stored_pattern.shape[2]))).reshape(shape=stored_pattern.shape)

        # Optionally apply state pattern normalisation.
        if self.norm_state_pattern is not None:
            state_pattern = self.norm_state_pattern.forward(input=state_pattern.reshape(
                shape=(-1, state_pattern.shape[2]))).reshape(shape=state_pattern.shape)

        # Optionally apply pattern projection normalisation.
        if self.norm_pattern_projection is not None:
            pattern_projection = self.norm_pattern_projection.forward(input=pattern_projection.reshape(
                shape=(-1, pattern_projection.shape[2]))).reshape(shape=pattern_projection.shape)

        # Apply Hopfield association and optional activation function.
        return self.association_core.forward(
            query=state_pattern, key=stored_pattern, value=pattern_projection,
            key_padding_mask=stored_pattern_padding_mask, need_weights=False, attn_mask=association_mask,
            scaling=self.__scaling, update_steps_max=self.__update_steps_max, update_steps_eps=self.__update_steps_eps,
            return_raw_associations=return_raw_associations)

    def forward(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                stored_pattern_padding_mask: Optional[Tensor] = None,
                association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Hopfield association on specified data.

        :param input: data to be processed by Hopfield association module
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: Hopfield-processed input data
        """
        association_output = self._maybe_transpose(self._associate(
            data=input, return_raw_associations=False,
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask)[0])
        if self.association_activation is not None:
            association_output = self.association_activation.forward(association_output)
        return association_output

    def get_association_matrix(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        """
        Fetch Hopfield association matrix gathered by passing through the specified data.

        :param input: data to be passed through the Hopfield association
        :return: association matrix as computed by the Hopfield core module
        """
        with torch.no_grad():
            return self._associate(data=input, return_raw_associations=True)[2]

    @property
    def batch_first(self) -> bool:
        return self.__batch_first

    @property
    def scaling(self) -> Union[float, Tensor]:
        return self.__scaling.clone() if type(self.__scaling) == Tensor else self.__scaling

    @property
    def stored_pattern_dim(self) -> int:
        return self.association_core.kdim

    @property
    def state_pattern_dim(self) -> int:
        return self.association_core.embed_dim

    @property
    def pattern_projection_dim(self) -> int:
        return self.association_core.vdim

    @property
    def input_size(self) -> int:
        return self.state_pattern_dim

    @property
    def hidden_size(self) -> int:
        return self.association_core.head_dim

    @property
    def output_size(self):
        return self.association_core.out_dim

    @property
    def update_steps_max(self) -> Optional[Union[int, Tensor]]:
        return self.__update_steps_max.clone() if type(self.__update_steps_max) == Tensor else self.__update_steps_max

    @property
    def update_steps_eps(self) -> Optional[Union[float, Tensor]]:
        return self.__update_steps_eps.clone() if type(self.__update_steps_eps) == Tensor else self.__update_steps_eps


class StatePattern(Module):
    """
    Wrapper class to be used in case of static state patterns.
    """

    def __init__(self,
                 size: int,
                 batch_first: bool = True,
                 trainable: bool = True
                 ):
        """
        Initialise a new instance of a state pattern wrapper.

        :param size: depth of the state pattern
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param trainable: state pattern is trainable
        """
        super(StatePattern, self).__init__()
        self.__batch_first = batch_first
        self.state_pattern = nn.Parameter(torch.empty(size=(1, 1, size)), requires_grad=trainable)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialise parameters of state pattern tensor.

        :return: None
        """
        nn.init.xavier_uniform_(self.state_pattern)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Wrap internal state pattern and external input.

        :param input: external input to pass-through as stored pattern and pattern projection
        :return: tuple containing stored and state pattern as well as pattern projection
        """
        state_pattern_expanded = self.state_pattern.expand(size=(1, input.shape[0], self.state_pattern.shape[2]))
        if self.__batch_first:
            state_pattern_expanded = state_pattern_expanded.transpose(0, 1)
        return input, state_pattern_expanded, input

    @property
    def batch_first(self) -> bool:
        return self.__batch_first

    @property
    def output_size(self) -> int:
        return self.state_pattern.shape[2]


class HopfieldPooling(Module):
    """
    Wrapper class encapsulating "StatePattern" and "Hopfield" in one combined module
    to be used as a Hopfield-based pooling layer.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,

                 normalise_stored_pattern: bool = True,
                 normalise_stored_pattern_affine: bool = True,
                 normalise_state_pattern: bool = True,
                 normalise_state_pattern_affine: bool = True,
                 normalise_pattern_projection: bool = True,
                 normalise_pattern_projection_affine: bool = True,
                 normalise_hopfield_space: bool = False,
                 normalise_hopfield_space_affine: bool = False,
                 stored_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False,
                 trainable: bool = True
                 ):
        """
        Initialise a new instance of a Hopfield-based pooling layer.

        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param normalise_stored_pattern: apply normalisation on stored patterns
        :param normalise_stored_pattern_affine: additionally enable affine normalisation of stored patterns
        :param normalise_state_pattern: apply normalisation on state patterns
        :param normalise_state_pattern_affine: additionally enable affine normalisation of state patterns
        :param normalise_pattern_projection: apply normalisation on the pattern projection
        :param normalise_pattern_projection_affine: additionally enable affine normalisation of pattern projection
        :param normalise_hopfield_space: enable normalisation of patterns in the Hopfield space
        :param normalise_hopfield_space_affine: additionally enable affine normalisation of patterns in Hopfield space
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        :param trainable: state pattern used for pooling is trainable
        """
        super(HopfieldPooling, self).__init__()
        self.hopfield = Hopfield(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_heads=num_heads,
            scaling=scaling, update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
            normalise_stored_pattern=normalise_stored_pattern,
            normalise_stored_pattern_affine=normalise_stored_pattern_affine,
            normalise_state_pattern=normalise_state_pattern,
            normalise_state_pattern_affine=normalise_state_pattern_affine,
            normalise_pattern_projection=normalise_pattern_projection,
            normalise_pattern_projection_affine=normalise_pattern_projection_affine,
            normalise_hopfield_space=normalise_hopfield_space,
            normalise_hopfield_space_affine=normalise_hopfield_space_affine,
            stored_pattern_as_static=stored_pattern_as_static, state_pattern_as_static=True,
            pattern_projection_as_static=pattern_projection_as_static, stored_pattern_size=stored_pattern_size,
            pattern_projection_size=pattern_projection_size, batch_first=batch_first,
            association_activation=association_activation, dropout=dropout, input_bias=input_bias,
            concat_bias_pattern=concat_bias_pattern, add_zero_association=add_zero_association,
            disable_out_projection=disable_out_projection)
        self.pooling_weights = StatePattern(
            size=self.hopfield.hidden_size, batch_first=batch_first, trainable=trainable)

    def forward(self, input: Tensor) -> Tensor:
        """
        Compute Hopfield-based pooling on specified data.

        :param input: data to be pooled
        :return: Hopfield-pooled input data
        """
        return self.hopfield.forward(input=self.pooling_weights.forward(input=input)).flatten(start_dim=1)

    def get_association_matrix(self, input: Tensor) -> Tensor:
        """
        Fetch Hopfield association matrix used for pooling gathered by passing through the specified data.

        :param input: data to be passed through the Hopfield association
        :return: association matrix as computed by the Hopfield core module
        """
        with torch.no_grad():
            return self.hopfield.get_association_matrix(input=self.pooling_weights.forward(input=input))

    @property
    def batch_first(self) -> bool:
        return self.hopfield.batch_first

    @property
    def scaling(self) -> Union[float, Tensor]:
        return self.hopfield.scaling

    @property
    def stored_pattern_dim(self) -> int:
        return self.hopfield.stored_pattern_dim

    @property
    def state_pattern_dim(self) -> int:
        return self.hopfield.state_pattern_dim

    @property
    def pattern_projection_dim(self) -> int:
        return self.hopfield.pattern_projection_dim

    @property
    def input_size(self) -> int:
        return self.hopfield.input_size

    @property
    def hidden_size(self) -> int:
        return self.hopfield.hidden_size

    @property
    def output_size(self):
        return self.hopfield.output_size

    @property
    def update_steps_max(self) -> Optional[Union[int, Tensor]]:
        return self.hopfield.update_steps_max

    @property
    def update_steps_eps(self) -> Optional[Union[float, Tensor]]:
        return self.hopfield.update_steps_eps
