import copy
import math

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import config


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(q, k, v, d_k, positional_bias=None, mask=None, dropout=None,
              memory_decay=False, memory_gamma=None, ltime=None):
    # ltime shape [batch, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [bs, nh, s, s]
    bs, nhead, seqlen = scores.size(0), scores.size(1), scores.size(2)

    if mask is not None:
        mask = mask.unsqueeze(1)

    if memory_decay and memory_gamma is not None and ltime is not None:
        time_seq = torch.cumsum(ltime.float(), dim=-1) - ltime.float()  # [bs, s]
        index_seq = torch.arange(seqlen).unsqueeze(-2).to(q.device)

        dist_seq = time_seq + index_seq

        with torch.no_grad():
            if mask is not None:
                scores_ = scores.masked_fill(mask, 1e-9)
            scores_ = F.softmax(scores_, dim=-1)
            distcum_scores = torch.cumsum(scores_, dim=-1)
            distotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_diff = dist_seq[:, None, :] - dist_seq[:, :, None]
            position_effect = torch.abs(position_diff)[:, None, :, :].type(torch.FloatTensor).to(q.device)
            dist_scores = torch.clamp((distotal_scores - distcum_scores) * position_effect, min=0.)
            dist_scores = dist_scores.sqrt().detach()

        m = nn.Softplus()
        memory_gamma = -1. * m(memory_gamma)
        total_effect = torch.clamp(torch.clamp((dist_scores * memory_gamma).exp(), min=1e-5), max=1e5)
        scores = total_effect * scores

    if positional_bias is not None:
        scores = scores + positional_bias

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    scores = F.softmax(scores, dim=-1)  # [bs, nh, s, s]

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(num_heads, config.MAX_SEQ, 1))
        self.m_srfe = MemorySRFE(embed_dim, num_heads)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, ltime=None, gamma=None, positional_bias=None,
                attn_mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if gamma is not None:
            gamma = self.m_srfe(gamma) + self.gammas
        else:
            gamma = self.gammas

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, positional_bias, attn_mask, self.dropout,
                           memory_decay=True, memory_gamma=gamma, ltime=ltime)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class BaseSRFE(nn.Module):
    def __init__(self, in_dim, n_head, dropout):
        super(BaseSRFE, self).__init__()
        assert in_dim % n_head == 0
        self.in_dim = in_dim // n_head
        self.n_head = n_head
        self.attention = MultiHeadAttention(embed_dim=in_dim, num_heads=n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(in_dim)

    def forward(self, x, pos_embed, mask):
        out = x
        att_out = self.attention(out, out, out, positional_bias=pos_embed, attn_mask=mask)
        out = out + self.dropout(att_out)
        out = self.layernorm(out)

        return x


class MemorySRFE(nn.Module):
    def __init__(self, in_dim, n_head):
        super(MemorySRFE, self).__init__()
        assert in_dim % n_head == 0
        self.in_dim = in_dim // n_head
        self.n_head = n_head
        self.linear1 = nn.Linear(self.in_dim, 1)

    def forward(self, x):
        bs = x.size(0)

        x = x.view(bs, -1, self.n_head, self.in_dim) \
            .transpose(1, 2) \
            .contiguous()
        x = self.linear1(x)
        return x


class PerformanceSRFE(nn.Module):
    def __init__(self, d_model, d_piv):
        super(PerformanceSRFE, self).__init__()
        self.linear1 = nn.Linear(d_model, 128)
        self.linear2 = nn.Linear(128, d_piv)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)

        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(FFN, self).__init__()
        self.lr1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU()
        self.lr2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lr1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lr2(x)
        return x


class PivotFFN(nn.Module):
    def __init__(self, d_model, d_ffn, d_piv, dropout):
        super(PivotFFN, self).__init__()
        self.p_srfe = PerformanceSRFE(d_model, d_piv)
        self.lr1 = nn.Bilinear(d_piv, d_model, d_ffn)
        self.lr2 = nn.Bilinear(d_piv, d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pivot):
        pivot = self.p_srfe(pivot)

        x = F.gelu(self.lr1(pivot, x))
        x = self.dropout(x)
        x = self.lr2(pivot, x)
        return x


class LANAEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout, max_seq):
        super(LANAEncoder, self).__init__()
        self.max_seq = max_seq

        self.multi_attn = MultiHeadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FFN(d_model, d_ffn, dropout)

    def forward(self, x, pos_embed, mask):
        out = x
        att_out = self.multi_attn(out, out, out, positional_bias=pos_embed, attn_mask=mask)
        out = out + self.dropout1(att_out)
        out = self.layernorm1(out)

        ffn_out = self.ffn(out)
        out = self.layernorm2(out + self.dropout2(ffn_out))

        return out


class LANADecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout, max_seq):
        super(LANADecoder, self).__init__()
        self.max_seq = max_seq

        self.multi_attn_1 = MultiHeadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.multi_attn_2 = MultiHeadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = FFN(d_model, d_ffn, dropout)

    def forward(self, x, memory, ltime, status, pos_embed, mask1, mask2):
        out = x
        att_out_1 = self.multi_attn_1(out, out, out, ltime=ltime,
                                      positional_bias=pos_embed, attn_mask=mask1)
        out = out + self.dropout1(att_out_1)
        out = self.layernorm1(out)

        att_out_2 = self.multi_attn_2(out, memory, memory, ltime=ltime,
                                      gamma=status, positional_bias=pos_embed, attn_mask=mask2)
        out = out + self.dropout2(att_out_2)
        out = self.layernorm2(out)

        ffn_out = self.ffn(out)
        out = self.layernorm3(out + self.dropout3(ffn_out))

        return out


class PositionalBias(nn.Module):
    def __init__(self, max_seq, embed_dim, num_heads, bidirectional=True, num_buckets=32, max_distance=config.MAX_SEQ):
        super(PositionalBias, self).__init__()
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.pos_embed = nn.Embedding(max_seq, embed_dim)  # Encoder position Embedding
        self.pos_query_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_key_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_layernorm = nn.LayerNorm(embed_dim)

        self.relative_attention_bias = nn.Embedding(32, config.N_HEADS)

    def forward(self, pos_seq):
        bs = pos_seq.size(0)

        pos_embed = self.pos_embed(pos_seq)
        pos_embed = self.pos_layernorm(pos_embed)

        pos_query = self.pos_query_linear(pos_embed)
        pos_key = self.pos_key_linear(pos_embed)

        pos_query = pos_query.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        pos_key = pos_key.view(bs, -1, self.h, self.d_k).transpose(1, 2)

        absolute_bias = torch.matmul(pos_query, pos_key.transpose(-2, -1)) / math.sqrt(self.d_k)
        relative_position = pos_seq[:, None, :] - pos_seq[:, :, None]

        relative_buckets = 0
        num_buckets = self.num_buckets
        if self.bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_bias = torch.abs(relative_position)
        else:
            relative_bias = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_bias < max_exact

        relative_bias_if_large = max_exact + (
                torch.log(relative_bias.float() / max_exact)
                / math.log(self.max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_bias_if_large = torch.min(
            relative_bias_if_large, torch.full_like(relative_bias_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_bias, relative_bias_if_large)
        relative_position_buckets = relative_buckets.to(pos_seq.device)

        relative_bias = self.relative_attention_bias(relative_position_buckets)
        relative_bias = relative_bias.permute(0, 3, 1, 2)

        position_bias = absolute_bias + relative_bias
        return position_bias


class LANA(nn.Module):
    def __init__(self, d_model, n_head, n_encoder, n_decoder, dim_feedforward, dropout,
                 max_seq, n_exercises, n_parts, n_resp, n_etime, n_ltime_s, n_ltime_m, n_ltime_d):
        super(LANA, self).__init__()
        self.max_seq = max_seq

        self.pos_embed = PositionalBias(max_seq, d_model, n_head, bidirectional=False, num_buckets=32,
                                        max_distance=max_seq)

        self.encoder_resp_embed = nn.Embedding(n_resp + 2, d_model,
                                               padding_idx=config.PAD)  # Answer Embedding, 0 for padding
        self.encoder_eid_embed = nn.Embedding(n_exercises + 2, d_model,
                                              padding_idx=config.PAD)  # Exercise ID Embedding, 0 for padding
        self.encoder_part_embed = nn.Embedding(n_parts + 2, d_model,
                                               padding_idx=config.PAD)  # Part Embedding, 0 for padding
        self.encoder_p_explan_embed = nn.Embedding(2 + 2, d_model, padding_idx=config.PAD)
        self.encoder_linear = nn.Linear(4 * d_model, d_model)
        self.encoder_layernorm = nn.LayerNorm(d_model)
        self.encoder_dropout = nn.Dropout(dropout)

        self.decoder_resp_embed = nn.Embedding(n_resp + 2, d_model,
                                               padding_idx=config.PAD)  # Answer Embedding, 0 for padding
        self.decoder_etime_embed = nn.Embedding(n_etime + 3, d_model, padding_idx=config.PAD)  # Elapsed time Embedding
        self.decoder_ltime_embed_s = nn.Embedding(n_ltime_s + 3, d_model,
                                                  padding_idx=config.PAD)  # Lag time Embedding 1
        self.decoder_ltime_embed_m = nn.Embedding(n_ltime_m + 3, d_model,
                                                  padding_idx=config.PAD)  # Lag time Embedding 2
        self.decoder_ltime_embed_h = nn.Embedding(n_ltime_d + 3, d_model,
                                                  padding_idx=config.PAD)  # Lag time Embedding 3
        self.decoder_linear = nn.Linear(5 * d_model, d_model)
        self.decoder_layernorm = nn.LayerNorm(d_model)
        self.decoder_dropout = nn.Dropout(dropout)

        self.encoder = get_clones(LANAEncoder(d_model, n_head, dim_feedforward, dropout, max_seq), n_encoder)
        self.srfe = BaseSRFE(d_model, n_head, dropout)
        self.decoder = get_clones(LANADecoder(d_model, n_head, dim_feedforward, dropout, max_seq), n_decoder)

        self.layernorm_out = nn.LayerNorm(d_model)
        self.ffn = PivotFFN(d_model, dim_feedforward, config.D_PIV, dropout)
        self.classifier = nn.Linear(d_model, 1)

    def get_pos_seq(self):
        return torch.arange(self.max_seq).unsqueeze(0)

    def _get_param_from_input(self, input):
        return (input["content_id"].long(),
                input["part"].long(),
                input["correctness"].long(),
                input["elapsed_time"].long(),
                input["lag_time_s"].long(),
                input["lag_time_m"].long(),
                input["lag_time_d"].long(),
                input["prior_explan"].long())

    def forward(self, input):
        exercise_seq, part_seq, resp_seq, etime_seq, ltime_s_seq, ltime_m_seq, ltime_d_seq, p_explan_seq = self._get_param_from_input(
            input)

        ltime = ltime_m_seq.clone()

        pos_embed = self.pos_embed(self.get_pos_seq().to(exercise_seq.device))

        # encoder embedding
        inter_seq = self.encoder_resp_embed(resp_seq)
        exercise_seq = self.encoder_eid_embed(exercise_seq)
        part_seq = self.encoder_part_embed(part_seq)
        p_explan_seq = self.encoder_p_explan_embed(p_explan_seq)
        encoder_input = torch.cat([exercise_seq, part_seq, p_explan_seq, inter_seq], dim=-1)
        encoder_input = self.encoder_linear(encoder_input)
        encoder_input = self.encoder_layernorm(encoder_input)
        encoder_input = self.encoder_dropout(encoder_input)

        # decoder embedding
        resp_seq = self.decoder_resp_embed(resp_seq)
        etime_seq = self.decoder_etime_embed(etime_seq)
        ltime_s_seq = self.decoder_ltime_embed_s(ltime_s_seq)
        ltime_m_seq = self.decoder_ltime_embed_m(ltime_m_seq)
        ltime_d_seq = self.decoder_ltime_embed_h(ltime_d_seq)
        decoder_input = torch.cat([resp_seq, etime_seq, ltime_s_seq, ltime_m_seq, ltime_d_seq], dim=-1)
        decoder_input = self.decoder_linear(decoder_input)
        decoder_input = self.decoder_layernorm(decoder_input)
        decoder_input = self.decoder_dropout(decoder_input)

        attn_mask = future_mask(self.max_seq).to(exercise_seq.device)
        # encoding
        encoding = encoder_input
        for mod in self.encoder:
            encoding = mod(encoding, pos_embed, attn_mask)

        srfe = encoding.clone()
        srfe = self.srfe(srfe, pos_embed, attn_mask)

        # decoding
        decoding = decoder_input
        for mod in self.decoder:
            decoding = mod(decoding, encoding, ltime, srfe, pos_embed,
                           attn_mask, attn_mask)

        predict = self.ffn(decoding, srfe)
        predict = self.layernorm_out(predict + decoding)
        predict = self.classifier(predict)
        return predict.squeeze(-1)
