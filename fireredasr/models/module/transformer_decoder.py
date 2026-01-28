from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None

try:
    import torchair as tng
    from torchair import CompilerConfig
except ModuleNotFoundError:  # pragma: no cover
    tng = None
    CompilerConfig = None


class TransformerDecoder(nn.Module):
    def __init__(
            self, sos_id, eos_id, pad_id, odim,
            n_layers, n_head, d_model,
            residual_dropout=0.1, pe_maxlen=5000):
        super().__init__()
        self.INF = 1e10
        # parameters
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        # Components
        self.tgt_word_emb = nn.Embedding(odim, d_model, padding_idx=self.pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = DecoderLayer(d_model, n_head, residual_dropout)
            self.layer_stack.append(block)

        self.tgt_word_prj = nn.Linear(d_model, odim, bias=False)
        self.layer_norm_out = nn.LayerNorm(d_model)

        self.tgt_word_prj.weight = self.tgt_word_emb.weight
        self.scale = (d_model ** 0.5)

        self.run_kernel_v1 = self._run_kernel_v1
        self.run_kernel_v2 = self._run_kernel_v2
        self._prefer_kernel_v2 = False

    def compile_kernel(self, backend=None, dynamic=True, fullgraph=False, mode="reduce-overhead"):
        if isinstance(mode, str) and mode.lower() in {"none", "default", "auto"}:
            mode = None
        if backend is None and tng is not None and CompilerConfig is not None:
            config = CompilerConfig()
            config.experimental_config.frozen_parameter = True
            if mode is not None:
                config.mode = mode
            backend = tng.get_npu_backend(compiler_config=config)

        if backend is None:
            self.run_kernel_v1 = torch.compile(self._run_kernel_v1, dynamic=dynamic, fullgraph=fullgraph)
            self.run_kernel_v2 = torch.compile(self._run_kernel_v2, dynamic=dynamic, fullgraph=fullgraph)
        else:
            self.run_kernel_v1 = torch.compile(
                self._run_kernel_v1, dynamic=dynamic, fullgraph=fullgraph, backend=backend
            )
            self.run_kernel_v2 = torch.compile(
                self._run_kernel_v2, dynamic=dynamic, fullgraph=fullgraph, backend=backend
            )
        self._prefer_kernel_v2 = True

    def reset_kernel(self):
        self.run_kernel_v1 = self._run_kernel_v1
        self.run_kernel_v2 = self._run_kernel_v2
        self._prefer_kernel_v2 = False

    def _run_kernel_v1(self, dec_output, encoder_outputs, tgt_mask, src_mask, caches):
        i = 0
        for dec_layer in self.layer_stack:
            dec_output = dec_layer.forward(
                dec_output,
                encoder_outputs,
                tgt_mask,
                src_mask,
                cache=caches[i],
            )
            caches[i] = dec_output
            i += 1
        return dec_output, caches

    def _run_kernel_v2(self, dec_output, encoder_outputs, tgt_mask, src_mask, caches):
        i = 0
        for dec_layer in self.layer_stack:
            dec_output = dec_layer.forward_v2(
                dec_output,
                encoder_outputs,
                tgt_mask,
                src_mask,
                cache=caches[i],
            )
            i += 1
        return dec_output, caches

    def batch_beam_search(self, encoder_outputs, src_masks,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
                   disable_early_stop: bool = False):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        use_version_2 = self._prefer_kernel_v2 or (Ti > 152)
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
        stride = B * torch.arange(N, device=device).view(N, 1).repeat(1, B).view(N * B)
        if use_version_2:
            caches = [torch.zeros((N * B, maxlen, H), device=device, dtype=encoder_outputs.dtype) for _ in range(self.n_layers)]
        else:
            caches = [torch.empty((N * B, 0, H), device=device, dtype=encoder_outputs.dtype) for _ in range(self.n_layers)]
        scores_temp = torch.tensor([0.0] + [-self.INF] * (B - 1), device=device).float()
        scores = scores_temp.repeat(N).view(N * B, 1)
        mask_score = scores_temp.view(1, B).repeat(N * B, 1)
        is_finished = torch.zeros_like(scores)

        # Autoregressive Prediction
        for t in range(maxlen):
            # In incremental decoding (q = last token), there is no "future token" in K/V,
            # so we can skip the causal mask entirely.
            tgt_mask = None

            dec_output = self.dropout(
                self.tgt_word_emb(ys) * self.scale +
                self.positional_encoding(ys))

            if use_version_2:
                dec_output, caches = self.run_kernel_v2(dec_output, encoder_outputs, tgt_mask, src_mask, caches)
            else:
                dec_output, caches = self.run_kernel_v1(dec_output, encoder_outputs, tgt_mask, src_mask, caches)

            dec_output = self.layer_norm_out(dec_output)

            t_logit = self.tgt_word_prj(dec_output[:, -1])
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished, mask_score=mask_score)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            # Accumulated
            scores = scores + t_topB_scores

            # Pruning
            scores = scores.view(N, B*B)
            scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
            scores = scores.view(-1, 1)

            topB_row_number_in_each_B_rows_of_ys = torch.floor_divide(topB_score_ids, B).view(N * B)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

            # Update ys
            ys = ys[topB_row_number_in_ys]
            t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
            ys = torch.cat((ys, t_ys), dim=1)

            # Update caches
            caches = [cache[topB_row_number_in_ys] for cache in caches]

            # Update finished state
            is_finished = t_ys.eq(self.eos_id)
            if not disable_early_stop:
                if is_finished.sum().item() == N * B:
                    break

        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(N*B, -1)[index.view(-1)]
        nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)

        # result
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                new_hyp = {
                    "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    def ignored_target_position_is_0(self, padded_targets, ignore_id):
        # NOTE: torch_npu/torchair GE converter may fail on `aten.ne.Scalar` when the scalar
        # becomes symbolic during Dynamo tracing. Force Tensor-Tensor compare on the same device.
        ignore = padded_targets.new_full((), ignore_id)
        mask = torch.ne(padded_targets, ignore)
        mask = mask.unsqueeze(dim=1)
        T = padded_targets.size(-1)
        upper_tri_0_mask = self.upper_triangular_is_0(T, device=mask.device).unsqueeze(0).to(mask.dtype)
        return mask.to(torch.uint8) & upper_tri_0_mask.to(torch.uint8)

    def upper_triangular_is_0(self, size, device=None):
        ones = torch.ones((size, size), device=device)
        tri_left_ones = torch.tril(ones)
        return tri_left_ones.to(torch.uint8)

    def set_finished_beam_score_to_zero(self, scores, is_finished, mask_score=None):
        NB, B = scores.size()
        is_finished = is_finished.float()
        if mask_score is None:
            mask_score = torch.tensor([0.0] + [-self.INF] * (B - 1), device=scores.device).float()
            mask_score = mask_score.view(1, B).repeat(NB, 1)
        return scores * (1 - is_finished) + mask_score * is_finished

    def set_finished_beam_y_to_eos(self, ys, is_finished):
        is_finished = is_finished.long()
        return ys * (1 - is_finished) + self.eos_id * is_finished

    def get_ys_lengths(self, ys):
        N, B, Tmax = ys.size()
        eos = ys.new_full((), self.eos_id)
        ys_lengths = torch.sum(torch.ne(ys, eos), dim=-1)
        return ys_lengths.int()



class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model*4, dropout)

    def forward(self, dec_input, enc_output, self_attn_mask, cross_attn_mask,
                cache=None):
        x = dec_input
        residual = x
        x = self.self_attn_norm(x)
        if cache is not None:
            xq = x[:, -1:, :]
            residual = residual[:, -1:, :]
            if self_attn_mask is not None:
                self_attn_mask = self_attn_mask[:, -1:, :]
        else:
            xq = x
        x = self.self_attn(xq, x, x, mask=self_attn_mask)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, enc_output, enc_output, mask=cross_attn_mask)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x

    def forward_v2(self, dec_input, enc_output, self_attn_mask, cross_attn_mask, cache):
        t = dec_input.size(1) - 1
        x = dec_input
        residual = x
        x = self.self_attn_norm(x)
        xq = x[:, -1:, :]
        residual = residual[:, -1:, :]
        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask[:, -1:, :]
        x = self.self_attn(xq, x, x, mask=self_attn_mask)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, enc_output, enc_output, mask=cross_attn_mask)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        cache[:, t : t + 1, :] = x
        return cache[:, : t + 1, :]


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)

        self._use_sdpa_eager = hasattr(F, "scaled_dot_product_attention")
        self.attention = DecoderScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _is_compiling() -> bool:
        try:
            import torch._dynamo  # noqa: WPS433

            return bool(torch._dynamo.is_compiling())
        except Exception:  # pragma: no cover
            pass
        try:
            return bool(torch.compiler.is_compiling())
        except Exception:  # pragma: no cover
            return False

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # NOTE: torch_npu's SDPA may not be traceable under Dynamo with symbolic sizes
        # (e.g., it may call storage_offset() in a way FakeTensor can't support). Use SDPA
        # only in eager; use the matmul/bmm-based attention under torch.compile.
        if self._use_sdpa_eager and (not self._is_compiling()):
            attn_mask = None
            if mask is not None:
                # incoming mask is "valid positions" (True/1 keeps, False/0 masks)
                mask_bool = mask.to(torch.bool)
                attn_mask = (~mask_bool).unsqueeze(1)  # [B,1,1,S], broadcast over heads and query length
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        else:
            if mask is not None:
                mask = mask.unsqueeze(1)
            output = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.fc(output)
        output = self.dropout(output)

        return output


class DecoderScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.INF = float("inf")

    def forward(self, q, k, v, mask=None):
        # Avoid 4D torch.matmul broadcasting paths (may trigger aten.expand/reshape patterns
        # that some TorchAIR backends cannot lower under dynamic shapes).
        bs, n_head, lq, d_k = q.shape
        lk = k.size(2)
        q_ = q.reshape(bs * n_head, lq, d_k)
        k_ = k.reshape(bs * n_head, lk, d_k)
        attn = torch.bmm(q_, k_.transpose(1, 2)).view(bs, n_head, lq, lk) / self.temperature
        if mask is not None:
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        attn_ = attn.reshape(bs * n_head, lq, lk)
        v_ = v.reshape(bs * n_head, lk, d_k)
        output = torch.bmm(attn_, v_).view(bs, n_head, lq, d_k)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = NpuGELU() if torch_npu is not None else nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.act(self.w_1(x)))
        output = self.dropout(output)
        return output


class NpuGELU(nn.Module):
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        if torch_npu is None:  # pragma: no cover
            return F.gelu(x, approximate=self.approximate)
        return torch_npu.npu_gelu(x, approximate=self.approximate)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)
        return self.pe[:, :length].clone().detach()
