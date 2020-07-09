import torch
import torch.nn as nn

from torch.tensor import Tensor
from typing import Optional, Tuple, Union


def hopfield_core_forward(query,                           # type: Tensor
                          key,                             # type: Tensor
                          value,                           # type: Tensor
                          embed_dim_to_check,              # type: int
                          num_heads,                       # type: int
                          in_proj_weight,                  # type: Optional[Tensor]
                          in_proj_bias,                    # type: Optional[Tensor]
                          bias_k,                          # type: Optional[Tensor]
                          bias_v,                          # type: Optional[Tensor]
                          add_zero_attn,                   # type: bool
                          dropout_p,                       # type: float
                          out_proj_weight,                 # type: Tensor
                          out_proj_bias,                   # type: Tensor
                          training=True,                   # type: bool
                          key_padding_mask=None,           # type: Optional[Tensor]
                          need_weights=True,               # type: bool
                          attn_mask=None,                  # type: Optional[Tensor]
                          use_separate_proj_weight=False,  # type: bool
                          q_proj_weight=None,              # type: Optional[Tensor]
                          k_proj_weight=None,              # type: Optional[Tensor]
                          v_proj_weight=None,              # type: Optional[Tensor]
                          static_k=None,                   # type: Optional[Tensor]
                          static_v=None,                   # type: Optional[Tensor]

                          key_as_static=False,             # type: bool
                          query_as_static=False,           # type: bool
                          value_as_static=False,           # type: bool
                          normalise_pattern=False,         # type: bool
                          p_norm_weight=None,              # type: Optional[Tensor]
                          p_norm_bias=None,                # type: Optional[Tensor]
                          head_dim=None,                   # type: Optional[int]
                          scaling=None,                    # type: Optional[Union[float, Tensor]]
                          update_steps_max=0,              # type: Optional[Union[int, Tensor]]
                          update_steps_eps=1e-4,           # type: Union[float, Tensor]
                          return_raw_associations=False    # type: bool
                          ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
            See "Hopfield Networks is All You Need" for more details in the setting of Hopfield networks.
        embed_dim_to_check: total dimension of the model (in case of default head dimension).
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.

        key_as_static: interpret specified key as being static.
        query_as_static: interpret specified key as being static.
        value_as_static: interpret specified key as being static.
        normalise_pattern: enable normalisation of patterns.
        p_norm_weight, p_norm_bias: pattern normalisation weight and bias.
        head_dim: dimensionality of each head.
        scaling: scaling of association heads, often represented as beta (one entry per head).
        update_steps_max: maximum count of association update steps (None equals to infinity).
        update_steps_eps: minimum difference threshold between two consecutive association update steps.
        return_raw_associations: return raw association (softmax) values, unmodified.

    Shape:
        Inputs:
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
        - static_k: :math:`(N*num_heads, S, head_dim)`, where S is the source sequence length, N is the batch size.
        - static_v: :math:`(N*num_heads, S, head_dim)`, where S is the source sequence length, N is the batch size.

        - scaling: :math:`(num_heads,)`, where num_heads is the amount of heads.

        Outputs:
        - attn_output: :math:`(L, N, E)`, where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)`, where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        - attn_raw: :math:``(N, num_heads, L, S)`, where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and nn.functional.has_torch_function(tens_ops):
            return nn.functional.handle_torch_function(
                hopfield_core_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v,
                key_as_static=key_as_static, query_as_static=query_as_static,
                value_as_static=value_as_static, normalise_pattern=normalise_pattern,
                p_norm_weight=p_norm_weight, p_norm_bias=p_norm_bias,
                head_dim=head_dim, scaling=scaling, update_steps_max=update_steps_max,
                update_steps_eps=update_steps_eps, return_raw_associations=return_raw_associations)
    tgt_len, (bsz, embed_dim) = query.shape[0], value.shape[1:]
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    assert (scaling is None) or (type(scaling) in (float, torch.Tensor))
    if type(scaling) == torch.Tensor:
        assert scaling.ndimension() == 1 and scaling.shape[0] == num_heads, "only one entry per head."

    assert (update_steps_max is None) or (type(update_steps_max) in (int, torch.Tensor))
    if type(update_steps_max) == torch.Tensor:
        assert update_steps_max.ndimension() == 1 and update_steps_max.shape[0] == num_heads, "only one entry per head."
    elif type(update_steps_max) == int:
        update_steps_max = torch.tensor([update_steps_max] * num_heads, dtype=torch.int32, device=query.device)
    elif update_steps_max is None:
        update_steps_max = -torch.ones(size=(num_heads,), dtype=torch.int32, device=query.device)

    assert type(update_steps_eps) in (float, torch.Tensor)
    if type(update_steps_eps) == torch.Tensor:
        assert update_steps_eps.ndimension() == 1 and update_steps_eps.shape[0] == num_heads, "only one entry per head."
        assert (update_steps_eps <= 0.0).sum() == 0, "only positive thresholds allowed."
        update_steps_eps = update_steps_eps.to(device=query.device)
    elif type(update_steps_eps) == float:
        assert update_steps_eps > 0, "only positive thresholds allowed."
        update_steps_eps = torch.tensor([update_steps_eps] * num_heads, dtype=query.dtype, device=query.device)

    if head_dim is None:
        hopfield_dim = embed_dim // num_heads
        assert head_dim * num_heads == hopfield_dim, r'embed_dim must be divisible by num_heads.'
    else:
        hopfield_dim = num_heads * head_dim

    q, k, v, xi, src_len = None, None, None, None, 0
    update_step, xi_old, xi_difference_norm = 0, None, float(r'+inf')
    update_active_heads = torch.tensor([[[True]]] * num_heads * bsz, device=query.device)
    assert update_active_heads.any(), "at least one head needs to be active."

    ####################################################################################################################
    #                                         BEGIN HOPFIELD UPDATE ITERATION                                          #
    ####################################################################################################################

    while update_active_heads.any():

        # They query is already projected into the "Hopfield" space at "update_step" equals 0.
        # No more projection necessary if "update_step" greater than 0.
        if update_step == 0:
            if not use_separate_proj_weight:

                if torch.equal(query, key) and torch.equal(key, value) and not (
                        key_as_static or query_as_static or value_as_static):
                    # self-attention
                    q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

                elif torch.equal(key, value) and not (key_as_static or value_as_static):
                    # encoder-decoder attention
                    _start, _end = 0, hopfield_dim
                    if query_as_static:
                        q = query.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        q = nn.functional.linear(query, _w, _b)
                        _start = hopfield_dim
                    _end = None

                    if key is None:
                        assert value is None
                        k = None
                        v = None
                    else:

                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

                else:
                    _start, _end = 0, hopfield_dim
                    if query_as_static:
                        q = query.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        q = nn.functional.linear(query, _w, _b)
                        _start += hopfield_dim
                        _end += hopfield_dim

                    if key_as_static:
                        k = key.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        k = nn.functional.linear(key, _w, _b)
                        _start += hopfield_dim
                        _end += hopfield_dim

                    if value_as_static:
                        v = value.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        v = nn.functional.linear(value, _w, _b)
            else:
                if query_as_static:
                    q = query.repeat(1, num_heads, 1)
                else:
                    q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
                    len1, len2 = q_proj_weight_non_opt.size()
                    assert len1 == hopfield_dim and len2 == query.size(-1)
                    if in_proj_bias is not None:
                        q = nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias[0:hopfield_dim])
                    else:
                        q = nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias)

                if key_as_static:
                    k = key.repeat(1, num_heads, 1)
                else:
                    k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
                    len1, len2 = k_proj_weight_non_opt.size()
                    assert len1 == hopfield_dim and len2 == key.size(-1)
                    if in_proj_bias is not None:
                        k = nn.functional.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(hopfield_dim * 2)])
                    else:
                        k = nn.functional.linear(key, k_proj_weight_non_opt, in_proj_bias)

                if value_as_static:
                    v = value.repeat(1, num_heads, 1)
                else:
                    v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
                    len1, len2 = v_proj_weight_non_opt.size()
                    assert len1 == hopfield_dim and len2 == value.size(-1)
                    if in_proj_bias is not None:
                        v = nn.functional.linear(value, v_proj_weight_non_opt, in_proj_bias[(hopfield_dim * 2):])
                    else:
                        v = nn.functional.linear(value, v_proj_weight_non_opt, in_proj_bias)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                    if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                        raise RuntimeError('The size of the 2D attn_mask is not correct.')
                elif attn_mask.dim() == 3:
                    if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                        raise RuntimeError('The size of the 3D attn_mask is not correct.')
                else:
                    raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
                # attn_mask's dim is 3 now.

            # Optionally normalise patterns.
            if normalise_pattern:
                q = torch.nn.functional.layer_norm(
                    input=q.reshape(shape=(-1, head_dim)), normalized_shape=(head_dim,),
                    weight=p_norm_weight, bias=p_norm_bias).reshape(shape=q.shape)
                k = torch.nn.functional.layer_norm(
                    input=k.reshape(shape=(-1, head_dim)), normalized_shape=(head_dim,),
                    weight=p_norm_weight, bias=p_norm_bias).reshape(shape=k.shape)

        else:
            active_xi = xi.masked_select(mask=update_active_heads).view(size=(-1, *xi.shape[1:]))
            active_k = k.masked_select(mask=update_active_heads).view(size=(-1, *k.shape[1:]))
            q = torch.masked_scatter(input=q, mask=update_active_heads, source=torch.bmm(active_xi, active_k))

        # Optionally scale association heads (each head separately).
        if type(scaling) == float:
            q *= scaling
        elif type(scaling) == torch.Tensor:
            q *= scaling.repeat(repeats=(head_dim,))

        if update_step == 0:
            if bias_k is not None and bias_v is not None:
                if static_k is None and static_v is None and key_as_static is None and value_as_static is None:
                    k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                    v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                    if attn_mask is not None:
                        attn_mask = nn.functional.pad(attn_mask, [0, 1])
                    if key_padding_mask is not None:
                        key_padding_mask = nn.functional.pad(key_padding_mask, [0, 1])
                else:
                    assert static_k is None, "bias cannot be added to static key."
                    assert static_v is None, "bias cannot be added to static value."
                    assert not key_as_static, "bias cannot be added to static key."
                    assert not value_as_static, "bias cannot be added to static value."
            else:
                assert bias_k is None
                assert bias_v is None

            q = q.contiguous().view(tgt_len, -1, head_dim).transpose(0, 1)
            if k is not None:
                k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            if v is not None:
                v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

            if static_k is not None:
                assert static_k.size(0) == bsz * num_heads
                assert static_k.size(2) == head_dim
                k = static_k

            if static_v is not None:
                assert static_v.size(0) == bsz * num_heads
                assert static_v.size(2) == head_dim
                v = static_v

            src_len = k.size(1)

            if key_padding_mask is not None:
                assert key_padding_mask.size(0) == bsz
                assert key_padding_mask.size(1) == src_len

            if add_zero_attn:
                src_len += 1
                k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
                v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
                if attn_mask is not None:
                    attn_mask = nn.functional.pad(attn_mask, [0, 1])
                if key_padding_mask is not None:
                    key_padding_mask = nn.functional.pad(key_padding_mask, [0, 1])

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        # Compute new xi for Hopfield retrieve iterations.
        if xi is None:
            xi = nn.functional.softmax(attn_output_weights, dim=-1)
        else:
            xi = torch.masked_scatter(input=xi, mask=update_active_heads, source=nn.functional.softmax(
                attn_output_weights.masked_select(mask=update_active_heads).view(size=(-1, *xi.shape[1:])), dim=-1))

        # Compute threshold-based stopping criterion for Hopfield retrieve iterations.
        with torch.no_grad():
            xi_active = xi.view(size=(bsz, num_heads, tgt_len, src_len))
            update_active_heads = (update_step < update_steps_max) | (update_steps_max < 0)
            if xi_old is not None:
                update_active_heads &= ((xi_old - xi_active).norm(p=2, dim=(2, 3)).max(axis=0)[0]) > update_steps_eps
            update_active_heads = update_active_heads.unsqueeze(dim=1).unsqueeze(dim=2).repeat(repeats=(bsz, 1, 1))
            xi_old = xi_active
        update_step += 1

    ####################################################################################################################
    #                                          END HOPFIELD UPDATE ITERATION                                           #
    ####################################################################################################################

    attn_output_weights = nn.functional.dropout(xi, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, hopfield_dim)
    if out_proj_weight is not None:
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    xi = xi.view(bsz, num_heads, tgt_len, src_len) if return_raw_associations else None
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, xi
    else:
        return attn_output, None, xi
