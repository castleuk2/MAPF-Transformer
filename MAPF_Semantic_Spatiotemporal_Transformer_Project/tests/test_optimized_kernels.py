import torch

from mapf_sst.attention import MultiheadAttentionWithBias


def test_fused_attention_matches_explicit_formula():
    torch.manual_seed(17)
    module = MultiheadAttentionWithBias(32, 4, 0.0).double().eval()
    x = torch.randn(2, 11, 32, dtype=torch.double)
    bias = torch.randn(2, 4, 11, 11, dtype=torch.double) * 0.1
    padding = torch.zeros(2, 11, dtype=torch.bool)
    padding[1, -2:] = True

    q = module._split(module.q_proj(x))
    k = module._split(module.k_proj(x))
    v = module._split(module.v_proj(x))
    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale + bias
    scores = scores.masked_fill(
        padding[:, None, None, :], torch.finfo(scores.dtype).min
    )
    attended = torch.matmul(torch.softmax(scores, dim=-1), v)
    attended = attended.transpose(1, 2).contiguous().view(2, 11, 32)
    expected = module.out_proj(attended).masked_fill(padding[..., None], 0.0)

    actual, weights = module(
        x, x, x, attn_bias=bias,
        key_padding_mask=padding, query_padding_mask=padding,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert weights is None
