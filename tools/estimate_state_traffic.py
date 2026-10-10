"""Estimate the attention state each model reads and writes per decode step, from executed shapes.

  python tools/estimate_state_traffic.py           # write reports/five-model-rerun-h100-v1/state_estimate.csv
  python tools/estimate_state_traffic.py --check   # fail if that file is stale

Standard library only. This is the labelled fallback for Kan's HBM question: the rented node refused
GPU performance counters, so nothing here is a hardware measurement. Every number is **Estimate**
(arithmetic on shapes), except the two columns copied from the live-state snapshots, which are the
gauge-times-pool approximation of the session.

What is counted: attention state only (stored keys/values or latents, index entries, window buffers,
recurrent state), per GPU, for one decode step of one request at context L. Each of the eight GPUs
holds its own slice or its own copy, so the node moves eight times these values. Weights (experts,
projections) are not state and are not counted; neither is activation traffic.

Shapes come from each pinned checkpoint config and from the cache definitions of vLLM
a98247ab4db686ee03c66d5feb3c761e52a2f8ab (vllm/models/deepseek_v4, deepseek_v41, glm5next;
vllm/model_executor/models/mimo_v2.py, qwen3_next.py; vllm/v1/kv_cache_interface.py). Assumptions
that the source does not settle are named in the `assumptions` column.
"""
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'five-model-rerun-h100-v1'
BASE = ROOT / 'reports' / STUDY
OUT = f'reports/{STUDY}/state_estimate.csv'
CONTEXTS = (('1k', 1024), ('16k', 16384), ('64k', 65536), ('128k', 131072))

# Bytes of one stored state on one GPU.
MLA_FP8 = 584        # DeepSeek fp8_ds_mla record: 448 B NoPE + 128 B RoPE + 8 B scales (deepseek_v4/attention.py)
INDEX_FP8 = 132      # FP8 index key: index_head_dim 128 + 4 B of block scales (0731; GLM uses the same layout)
INDEX_FP4 = 68       # FP4 index key: 128/2 + 128/32 (V4.1, _indexer_k_cache_head_dim)
MIMO_KV = 2 * (192 + 128)    # one KV head per GPU, key 192 + value 128 dims, BF16
QWEN_KV = 2 * (256 + 256)    # one KV head per GPU, key 256 + value 256 dims, BF16
GLM_LATENT = 2 * 512         # kv_lora_rank 512 latent, BF16
QWEN_RECURRENT = 6 * 128 * 128 * 2     # 48 value heads / 8 GPUs, 128 x 128, BF16 (gated_delta_net_state_shape)
GLM_RECURRENT = 8 * 128 * 128 * 4      # 64 heads / 8 GPUs, 128 x 128, float32 (kda_state_shape, kda_state_dtype)


def v4_0731(L):
    """43 layers: 2 window-only, 21 with one state per 4 tokens, 20 with one state per 128 tokens;
    window 128, index_topk 512 (config compress_ratios, sliding_window, index_topk)."""
    c4, c128 = L // 4, L // 128
    return dict(
        window=43 * min(L, 128) * MLA_FP8,
        selected=21 * min(512, c4) * MLA_FP8 + 20 * min(512, c128) * MLA_FP8,
        search=21 * c4 * INDEX_FP8 + 20 * c128 * INDEX_FP8, full=0, recurrent_read=0, recurrent_write=0,
        stored_per_token=21 * (MLA_FP8 + INDEX_FP8) / 4 + 20 * (MLA_FP8 + INDEX_FP8) / 128,
        assumptions='the indexer reads every index entry of the request at each step; page alignment ignored')


def v41(L):
    """40 layers: KV kept at 4 source layers (3 at one state per 2 tokens, 1 per token), index keys at 8
    source layers; window 128, index_topk 512, candidate pool 2,048 blocks of 8 (config kv_source_layer_ids,
    index_source_layer_ids, compress_ratios, candidate_topk_blocks, candidate_block_size)."""
    half = L // 2
    return dict(
        window=40 * min(L, 128) * MLA_FP8,
        selected=18 * min(512, half) * MLA_FP8 + 20 * min(512, L) * MLA_FP8,
        search=(3 * half + L + 4 * min(L, 2048 * 8)) * INDEX_FP4, full=0, recurrent_read=0, recurrent_write=0,
        stored_per_token=3 * MLA_FP8 / 2 + MLA_FP8 + (3 / 2 + 5) * INDEX_FP4,
        assumptions='the four index sources up to layer 20 read every index entry and the four deeper ones read '
                    'only the candidate pool; every layer gathers its own 512 states from a shared source; '
                    'Engram tables live in host memory and are not counted')


def mimo(L):
    """48 layers: 39 with a 128-token window, 9 with full attention (config hybrid_layer_pattern)."""
    return dict(window=39 * min(L, 128) * MIMO_KV, selected=0, search=0, full=9 * L * MIMO_KV,
                recurrent_read=0, recurrent_write=0, stored_per_token=9 * MIMO_KV,
                assumptions='full-attention layers read all stored keys and values of the request')


def qwen(L, index_per_token):
    """48 layers: 36 recurrent, 12 that look up at most 2,048 stored positions through an index with one
    entry per 4 tokens (config layer_types, indexer_budget, indexer_compress_ratio)."""
    return dict(window=0, selected=12 * min(L, 2048) * QWEN_KV, search=12 * L * index_per_token, full=0,
                recurrent_read=36 * QWEN_RECURRENT, recurrent_write=36 * QWEN_RECURRENT,
                stored_per_token=12 * (QWEN_KV + index_per_token), stored_from_shapes=12 * QWEN_KV,
                assumptions='the from-shapes column holds keys and values only; index bytes per token are not '
                            'derived from source and are taken as the measured growth minus that part; the '
                            'indexer reads every index entry; each recurrent state is read and rewritten whole')


def glm(L):
    """45 layers: 34 recurrent, 11 sparse with index_topk 2,048 and one index entry per 4 tokens
    (config linear_attn_config, index_topk, index_kpool)."""
    return dict(window=0, selected=11 * min(L, 2048) * GLM_LATENT, search=11 * (L // 4) * INDEX_FP8, full=0,
                recurrent_read=34 * GLM_RECURRENT, recurrent_write=34 * GLM_RECURRENT,
                stored_per_token=11 * (GLM_LATENT + INDEX_FP8 / 4),
                assumptions='the indexer reads every pooled index entry; each recurrent state is read and '
                            'rewritten whole; tail buffers and page padding ignored')


def measured():
    """Growth of live state per token (64K to 128K, one request) and the fixed part, from memory.csv."""
    p = BASE / 'memory.csv'
    out = {}
    if not p.is_file():
        return out
    with p.open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['live_sequences'] == '1' and r.get('kv_usage_frac_decoding')]
    for mk in {r['model'] for r in rows}:
        pt = {r['bucket']: (float(r['live_tokens_server_prompt_plus_generated']),
                            float(r['kv_usage_frac_decoding']) * float(r['pool_gib_per_rank']) * 2**30)
              for r in rows if r['model'] == mk}
        if '64k' in pt and '128k' in pt:
            slope = (pt['128k'][1] - pt['64k'][1]) / (pt['128k'][0] - pt['64k'][0])
            out[mk] = (slope, pt['128k'][1] - slope * pt['128k'][0])
    return out


def generate():
    meas = measured()
    rows = []
    for mk, fn in (('v4-0731', v4_0731), ('v41', v41), ('mimo-v26', mimo), ('qwen-38-bf16', None), ('glm-53', glm)):
        slope, fixed = meas.get(mk, (None, None))
        for bucket, L in CONTEXTS:
            if mk == 'qwen-38-bf16':
                if slope is None:
                    continue  # its index entry size is taken from the measured growth
                e = qwen(L, max(0.0, slope / 12 - QWEN_KV))
            else:
                e = fn(L)
            reads = e['window'] + e['selected'] + e['search'] + e['full'] + e['recurrent_read']
            writes = e['stored_per_token'] + e['recurrent_write']
            rows.append(dict(
                study=STUDY, model=mk, label='Estimate', context=bucket, context_tokens=L, unit='bytes per GPU per decode step, one request',
                read_window_bytes=round(e['window']), read_selected_bytes=round(e['selected']),
                read_search_index_bytes=round(e['search']), read_full_attention_bytes=round(e['full']),
                read_recurrent_bytes=round(e['recurrent_read']), read_total_bytes=round(reads),
                write_new_token_bytes=round(e['stored_per_token']), write_recurrent_bytes=round(e['recurrent_write']),
                write_total_bytes=round(writes), read_plus_write_bytes=round(reads + writes),
                stored_bytes_per_token_from_shapes=round(e.get('stored_from_shapes', e['stored_per_token']), 1),
                stored_bytes_per_token_measured='' if slope is None else round(slope, 1),
                shapes_over_measured='' if slope is None else round(e.get('stored_from_shapes', e['stored_per_token']) / slope, 3),
                fixed_bytes_per_request_measured='' if fixed is None else round(fixed),
                assumptions=e['assumptions']))
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    return {OUT: buf.getvalue()}


def main():
    stale = False
    for rel, body in generate().items():
        p = ROOT / rel
        if '--check' in sys.argv:
            if not p.is_file() or p.read_text(encoding='utf-8') != body:
                print('stale:', rel); stale = True
        elif BASE.is_dir():
            p.write_text(body, encoding='utf-8', newline='\n')
            print('wrote', rel)
    return 1 if stale else 0


if __name__ == '__main__':
    sys.exit(main())
