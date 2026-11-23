#!/usr/bin/env python3
"""CPU-only regression for multi-window PPL aggregation and strict validation."""
import json
import math
import tempfile
from pathlib import Path
from aggregate_continuous_sequences import aggregate, read_csv, save_csv


def make_window(root,index,method_nll):
    folder=root/'sequences'/f'seq_{index}'/'ppl_continuous'
    folder.mkdir(parents=True)
    cfg={'sequence_index':index,'token_start':index*10,'token_end':(index+1)*10,
         'cycles':2,'unscored_len':3,'scored_len':2,'model':'test','dataset':'wikitext2',
         'reference_sha256':'f'*64,'scored_stream_sha256':format(index+1,'064x'),
         'output_dir':str(folder),'p_values':'2','lambda_ea':'0.6','lambda_ee':'0.6',
         'budgets':'0.1','methods':list(method_nll)}
    (folder/'continuous_config.json').write_text(json.dumps(cfg))
    results=[];cycle_rows=[]
    for method,nll in method_nll.items():
        common={'method':method,'p':2.,'budget':0.1,'lambda_ea':0.6,'lambda_ee':0.6,
                'sequence_index':index,'token_start':index*10,'token_end':(index+1)*10,
                'stream_sha256':cfg['scored_stream_sha256']}
        results.append(dict(common,scored_tokens=4,ppl=math.exp(nll),nll_per_token=nll,
                    compress_rounds=2,routed_slots=4,accepted_slots=2,rejected_slots=2,
                    dropped_slots=2,mis_conflict_edges=0,geometry_over_delta=0,
                    max_before_len=10,max_after_len=1,seconds=1.0,reused_baseline=False))
        for cycle in (1,2):
            cycle_rows.append(dict(common,cycle=cycle,logical_tokens=cycle*5,
                scored_tokens=2,cycle_nll_sum=2*nll,cycle_nll_per_token=nll,
                cycle_ppl=math.exp(nll),cumulative_nll_per_token=nll,
                cumulative_ppl=math.exp(nll),compression_rounds=cycle,
                routed_slots=4,accepted_slots=2,rejected_slots=2,
                geometry_over_delta=0,physical_kv_len=5))
    save_csv(folder/'ppl_continuous.csv',results)
    save_csv(folder/'ppl_continuous_cycles.csv',cycle_rows)
    (folder/'continuous_run_complete.json').write_text(json.dumps(dict(config=cfg,
        n_results=len(results),n_cycles=len(cycle_rows))))
    return folder


def test_multi_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        make_window(root,1,{'full_cache':1.0,'eviction_only':2.0,'merge_all':1.8})
        make_window(root,2,{'full_cache':2.0,'eviction_only':3.0,'merge_all':2.2})
        manifest=aggregate(root,[1,2]);assert manifest['scored_tokens_per_method']==8
        rows={r['method']:r for r in read_csv(root/'aggregate'/'ppl_continuous.csv')}
        assert abs(float(rows['eviction_only']['nll_per_token'])-2.5)<1e-12
        assert abs(float(rows['merge_all']['ppl'])-math.exp(2.0))<1e-12
        assert abs(float(rows['full_cache']['nll_per_token'])-1.5)<1e-12
        assert len(read_csv(root/'aggregate'/'ppl_continuous_cycles.csv'))==6
        assert len(read_csv(root/'aggregate'/'ppl_continuous_per_sequence.csv'))==6
        assert len(read_csv(root/'aggregate'/'ppl_continuous_paired_deltas.csv'))==6
        for invalid in ([1,1],[1,-1],[1,3]):
            try:aggregate(root,invalid)
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Should reject invalid windows '+str(invalid))
        assert not (root/'aggregate'/'aggregate_complete.json').exists(), 'Stale success marker'
        print('[OK] multi-sequence weighted NLL/PPL, cycle accounting, and bad-window rejection')


if __name__=='__main__':test_multi_sequence()
