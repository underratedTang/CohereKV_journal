#!/usr/bin/env python3
"""CPU-only regression for six-arm held-out aggregation and ZIP contract."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import tempfile
import zipfile
from aggregate_group_d2_quick import ARMS,aggregate,validate_plan
from package_group_d2_sweep import pack
from result_csv import save_csv
from method_registry import selected_methods


def fixture(root):
    assert selected_methods('k_only_importance','coherekv_inf_certificate,group_d2,group_d2_importance_matched')==ARMS
    sid='h0p80_ea0p6_ee0p6_b0p10_k8'
    setting=dict(setting_id=sid,lambda_history='0.80',lambda_ea='0.60',lambda_ee='0.60',
                 budget='0.10',pair_window='8')
    save_csv(root/'sweep_plan.csv',[setting])
    for seq in (1,2):
        f=root/'settings'/sid/'sequences'/f'seq_{seq}'/'ppl_continuous'
        f.mkdir(parents=True)
        cfg=dict(model='Qwen/Qwen2.5-7B',dataset='wikitext2',
            cycles=3,unscored_len=8,scored_len=4,p_values='2',geometry_modes='exact_p2',
            reference_mode='frozen_train',sink_size=1,recent_protect_size=2,score_window=2,
            recency_beta=1.,prefill_chunk_size=4,distance_chunk_size=64,
            graph_backend='streamed',torch_dtype='float32',attn_implementation='eager',
            device_map='auto',fusion_sample_topk=0,sequence_index=seq,
            token_start=seq*36,token_end=(seq+1)*36,lambda_history=0.8,
            group_d2_pair_window=8,budgets='0.10',methods=list(ARMS),
            resolved_lambda_pairs=[dict(lambda_ea=.6,lambda_ee=.6,ee_over_ea=1.)],
            scored_stream_sha256=f'fixture_stream_{seq}',reference_sha256='fixture_calibration')
        (f/'continuous_config.json').write_text(json.dumps(cfg))
        rows=[];cycles=[];heads=[]
        for ix,m in enumerate(ARMS):
            nll=1.0+ix*.1+seq*.01
            accepted=0 if m in ('full_cache','eviction_only') else 12
            routed=0 if m in ('full_cache','eviction_only') else 18
            remerged=0 if m in ('full_cache','eviction_only') else 3
            rows.append(dict(method=m,p=(math.inf if m=='coherekv_inf_certificate' else 2.),
                geometry_mode=('minkowski' if m=='coherekv_inf_certificate' else 'exact_p2'),
                nll_per_token=nll,ppl=math.exp(nll),scored_tokens=12,
                accepted_slots=accepted,routed_slots=routed,
                accepted_previously_merged_slots=remerged,group_d2_real_violations=0,
                group_d2_matched_quota_shortfall=0,
                lambda_ea=.6,lambda_ee=.6,budget=.1))
            for rd in range(1,4):
                cycles.append(dict(method=m,cycle=rd,scored_tokens=4,
                    cycle_nll_sum=nll*4,cycle_nll_per_token=nll))
                if m!='full_cache':
                    for h in range(2):
                        heads.append(dict(method=m,round=rd,layer_idx=0,kv_head_idx=h,
                            accepted_slots=0 if m=='eviction_only' else 2,
                            routed_slots=0 if m=='eviction_only' else 3,
                            accepted_previously_merged_slots=0 if m=='eviction_only' else (1 if h==0 else 0),
                            max_radius_over_delta=('0.70' if m=='coherekv_inf_certificate' else ''),
                            max_diameter_over_delta=('0.9' if m=='coherekv_inf_certificate' else ''),
                            real_history_d2_over_delta_max=('0.8' if m=='group_d2' else '')))
        save_csv(f/'ppl_continuous.csv',rows)
        save_csv(f/'ppl_continuous_cycles.csv',cycles)
        save_csv(f/'ppl_continuous_head_rounds.csv',heads)
        (f/'continuous_run_complete.json').write_text(json.dumps(dict(config=cfg,n_results=6,n_cycles=18)))
    return sid


def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        sid=fixture(root)
        assert len(validate_plan(root))==1
        settings,ranked=aggregate(root,[1,2])
        assert len(settings)==len(ranked)==1
        assert int(settings[0]['scored_tokens'])==24
        rounds=root/'group_d2_sweep_rounds.csv'
        assert rounds.exists() and len(rounds.read_text().splitlines())==1+3*6
        zpath=pack(root)
        with zipfile.ZipFile(zpath) as z:
            assert z.testzip() is None
            assert 'group_d2_sweep_rounds.csv' in z.namelist()
            assert f'settings/{sid}/group_d2_round_metrics.csv' in z.namelist()
        # Strictly reject incomplete cycle rows despite a completion marker.
        f=root/'settings'/sid/'sequences'/'seq_2'/'ppl_continuous'/'ppl_continuous_cycles.csv'
        content=f.read_text();f.write_text('\n'.join(content.splitlines()[:-1])+'\n')
        try:aggregate(root,[1,2])
        except RuntimeError as exc:assert 'Missing method/cycle/head' in str(exc) or 'cycles' in str(exc)
        else:raise AssertionError('Missing scored cycle silently accepted')
    print('[PASS] six-arm held-out aggregate, complete per-round data, ZIP, rejection tests')

if __name__=='__main__':main()
