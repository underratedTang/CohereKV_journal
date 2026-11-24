#!/usr/bin/env python3
"""CPU tests of conference routing, native full-cache scoring and multi-window pooling."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from conference_mis import conference_routes,conference_weights
from sequence_windows import choose_indices
from aggregate_continuous_sequences import aggregate,SUM_FIELDS,FLOAT_SUM_FIELDS
from method_registry import METHODS,methods_for_p,select_methods,attention_for_method

class NativeCache:
    def __init__(self):self.key_cache=[]

class NativeModel:
    def __call__(self,ids,*,use_cache,past_key_values):
        cache=past_key_values
        old=cache.key_cache[0].shape[-2] if cache.key_cache else 0
        cache.key_cache=[torch.zeros(1,1,old+ids.shape[-1],1)]
        # At each position predicts next integer mod 16 with larger logit.
        logits=torch.zeros(1,ids.shape[-1],16)
        logits.scatter_(2,((ids+1)%16).unsqueeze(-1),3.)
        return SimpleNamespace(logits=logits,past_key_values=cache)

def load_evaluator():
    tree=ast.parse(Path(__file__).with_name('evaluate_ppl_continuous.py').read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_steps','evaluate_continuous')]
    ns=dict(torch=torch,math=math,DynamicCache=NativeCache)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'evaluation','exec'),ns)
    return ns['evaluate_continuous']

class Tests(unittest.TestCase):
    def test_windows_exact_multiple(self):
        self.assertEqual(choose_indices([0,1,2,3],30,10),([0,1,2],3))
        with self.assertRaises(ValueError):choose_indices([0,0],30,10)
        with self.assertRaises(ValueError):choose_indices([3],30,10)

    def test_conference_chooses_compatible_not_nearest(self):
        mu=torch.tensor([[0.],[5.],[2.]],dtype=torch.float64)
        r=torch.tensor([3.,0.,0.]);d=torch.tensor([6.,0.,0.]);keep=torch.tensor([0,1]);evict=torch.tensor([2])
        owner,routed,hard,gamma,physical=conference_routes(mu,mu,r,d,keep,evict,4.,4.)
        self.assertEqual(owner.item(),1);self.assertTrue(routed.item());self.assertTrue(hard.item())
        self.assertEqual(gamma.item(),3.);self.assertEqual(physical.item(),3.)
        # Ties use smaller physical index, independent of keep array order.
        mu=torch.tensor([[0.],[4.],[2.]],dtype=torch.float64)
        r=torch.zeros(3);d=r;keep=torch.tensor([1,0])
        owner,*_=conference_routes(mu,mu,r,d,keep,evict,4.,4.)
        self.assertEqual(keep[owner].item(),0)

    def test_conference_maximal_selection(self):
        mu=torch.tensor([[0.],[-3.],[0.],[3.]],dtype=torch.float64)
        r=torch.zeros(4);d=r;keep=torch.tensor([0]);evict=torch.tensor([1,2,3])
        best,route,hard,gamma,_=conference_routes(mu,mu,r,d,keep,evict,4.,4.)
        plan=conference_weights(mu,r,d,keep,evict,best,route,hard,gamma,4.,4.,math.inf)
        self.assertEqual(plan.hard_edges,1)
        self.assertEqual(plan.positive_slots,2)
        self.assertEqual(plan.weights[1].item(),1.)
        self.assertEqual(plan.weights[0].item()+plan.weights[2].item(),1.)

    def test_full_cache_same_scored_positions(self):
        ev=load_evaluator();ids=torch.arange(18).reshape(1,-1)%16
        ppl,rows=ev(NativeModel(),ids,3,4,2,torch.device('cpu'),prefill_chunk_size=2,score_window=2)
        expected=math.log(math.exp(3)+15)-3
        self.assertAlmostEqual(math.log(ppl),expected,places=6)
        self.assertEqual([r['physical_kv_len'] for r in rows],[6,12,18])
        self.assertEqual([r['scored_tokens'] for r in rows],[2,2,2])
        self.assertTrue(all(r['compression_rounds']==0 for r in rows))
        # Change only one scored target: only that prediction should lose 3 NLL
        # (also affects next-token prediction at the next scored position).
        ids2=ids.clone();ids2[0,5]=13
        _,changed=ev(NativeModel(),ids2,3,4,2,torch.device('cpu'),prefill_chunk_size=2,score_window=2)
        self.assertAlmostEqual(changed[0]['cycle_nll_sum']-rows[0]['cycle_nll_sum'],3.,places=6)
        self.assertAlmostEqual(changed[1]['cycle_nll_sum']-rows[1]['cycle_nll_sum'],0.,places=6)

    def test_method_switches(self):
        self.assertEqual(len(METHODS),8)
        self.assertEqual(len(methods_for_p(2)),8)
        self.assertEqual(len(methods_for_p(math.inf)),6)
        for m in METHODS:
            self.assertEqual(methods_for_p(math.inf,select_methods(m)),(m,))
        for bad in ('','unknown','full_cache,full_cache'):
            with self.assertRaises(ValueError):select_methods(bad)
        self.assertEqual(attention_for_method('coherekv_soft_mis_log_count'),'log_count')
        self.assertEqual(attention_for_method('coherekv_soft_mis_importance_log_count'),'log_count')

    def test_aggregate_pools_and_deduplicates(self):
        self._check_aggregate(METHODS,16)

    def test_aggregate_no_baselines(self):
        self._check_aggregate(('coherekv_soft_mis_importance_log_count',),4)

    def test_aggregate_full_only(self):
        self._check_aggregate(('full_cache',),1)

    def _check_aggregate(self,selected,expected_rows):
        def save(path,rows):
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for i,nll in enumerate([math.log(4),math.log(16)]):
                d=root/'sequences'/f'seq_{i}'/'ppl_continuous';d.mkdir(parents=True)
                sha=hashlib.sha256(str(i).encode()).hexdigest()
                cfg=dict(sequence_index=i,cycles=1,unscored_len=4,scored_len=2,token_start=i*6,token_end=(i+1)*6,
                         scored_stream_sha256=sha,output_dir=str(d),methods=list(selected),methods_by_p={str(p):list(methods_for_p(p,selected)) for p in (2.,math.inf)},
                         budgets='0.1',lambda_ea='0.5',lambda_ee='0.1,0.5',split='validation')
                rows=[];cr=[]
                for p in [2.,math.inf]:
                    for ee in [.1,.5]:
                        for m in methods_for_p(p,selected):
                            candidate=m.startswith('coherekv_soft_mis');conf=m=='coherekv_conference';full=m=='full_cache'
                            x=nll if full else nll+.1
                            common=dict(method=m,p=p,budget=.1,lambda_ea=.5,lambda_ee=ee,
                                        actual_p=p if candidate else math.inf if conf else '',actual_budget='' if full else .1,
                                        actual_lambda_ea='' if m in ('full_cache','eviction_only') else .5,
                                        actual_lambda_ee=ee if candidate else .5 if conf else '',
                                        actual_reference_mode='online_first' if conf else 'none' if m in ('full_cache','eviction_only') else 'frozen_train',
                                        sequence_index=i,token_start=i*6,token_end=(i+1)*6,stream_sha256=sha,scored_tokens=2)
                            vals={k:0 for k in (*SUM_FIELDS,*FLOAT_SUM_FIELDS)}
                            rows.append(dict(**common,ppl=math.exp(x),nll_per_token=x,nll_sum=2*x,compress_rounds=0 if full else 1,
                                             **vals,importance_retention_rate='',max_before_len=6,max_after_len=6 if full else 1,seconds=1.,reused_baseline=False))
                            cr.append(dict(**common,cycle=1,cycle_nll_sum=2*x,cycle_nll_per_token=x,cycle_ppl=math.exp(x),
                                           cumulative_nll_per_token=x,cumulative_ppl=math.exp(x),compression_rounds=0 if full else 1,
                                           **{k:v for k,v in vals.items() if k!='mis_conflict_edges'},physical_kv_len=6,importance_retention_rate=''))
                (d/'continuous_config.json').write_text(json.dumps(cfg))
                (d/'continuous_run_complete.json').write_text(json.dumps(dict(config=cfg,n_results=len(rows),n_cycles=len(cr))))
                save(d/'ppl_continuous.csv',rows);save(d/'ppl_continuous_cycles.csv',cr)
            aggregate(root,[0,1])
            with (root/'aggregate'/'ppl_sweep_unique.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),expected_rows)
            self.assertEqual({r['method'] for r in rows},set(selected))
            if 'full_cache' in selected:
                full=next(r for r in rows if r['method']=='full_cache')
                self.assertAlmostEqual(float(full['overall_ppl']),8.)
                self.assertAlmostEqual(float(full['mean_ppl']),10.)
                self.assertAlmostEqual(float(full['std_ppl']),math.sqrt(72))
                self.assertEqual(int(full['scored_tokens']),4)
                for r in rows:
                    self.assertAlmostEqual(float(r['delta_nll_vs_full']),0. if r['method']=='full_cache' else .1)
            else:
                self.assertTrue(all(r['delta_nll_vs_full']=='' for r in rows))
                self.assertTrue(all(math.isclose(float(r['overall_ppl']),8.*math.exp(.1)) for r in rows))
            # Plot actual available arms (font requirement is tested on the server).
            from plot_continuous_ppl import heatmaps,baseline_plot,best_plot,read_rows
            agg=root/'aggregate';fig=agg/'figures'
            heatmaps(read_rows(agg/'ppl_continuous.csv'),fig,'ppl',40)
            heatmaps(read_rows(agg/'ppl_continuous.csv'),fig,'delta',40,field='delta_nll_vs_full')
            baseline_plot(rows,fig,40)
            best_plot(read_rows(agg/'ppl_sweep_best_by_method_p.csv'),read_rows(agg/'ppl_continuous_paired_deltas.csv'),fig,40)
            self.assertTrue(list(fig.glob('best_ppl*.png')))
            self.assertTrue(list(fig.glob('best_ppl*.pdf')))
            if 'full_cache' not in selected:self.assertFalse(list(fig.glob('best_paired*.png')))

if __name__=='__main__':unittest.main()
