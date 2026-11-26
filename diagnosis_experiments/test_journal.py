#!/usr/bin/env python3
"""CPU geometry, calibration, sweep, real cache-class (stub runtime) and pooling tests."""
import ast,csv,json,math,tempfile,unittest,contextlib,io
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from journal_mis import route,hard_mis,count_mean,propagate,relation_distance,merge_all_baseline,node_weights,net_gain
from calibrate_median_reference import key_scales,aggregate_median_scales,choose_starts,validate_payload
from method_registry import METHODS, PRIORITY_MODES
from sweep_config import build_configs
from aggregate_continuous_sequences import aggregate,SUM_FIELDS,FLOAT_SUM_FIELDS


def args(**kw):
    return NS(**(dict(methods=','.join(METHODS),budgets='.1',alpha_ea='0,.25,.5,.75,1',alpha_ee='0,.25,.5,.75,1',lambda_ea='.5',lambda_ee='.5',reference_modes='frozen_train',priority_modes='degree')|kw))

class FakeBase:
    def __init__(self,*,model,stats,**kw):
        self._ppl_model=model;self._stats=stats;self._forward_calls=0;self._ppl_logical_seen_tokens=0;self.key_cache=[];self.value_cache=[]

def replace(cache,kv):cache.key_cache=[k for k,v in kv];cache.value_cache=[v for k,v in kv]
def split(keys,imp,pos,B,sink,recent):return torch.arange(B).expand(keys.shape[0],-1),torch.arange(B,keys.shape[1]).expand(keys.shape[0],-1)
def importance(score,ptr,beta):return torch.ones(score.shape[0],score.shape[-1],device=score.device)
def reference(keys):return key_scales(keys).reshape(1,-1,1)

def load_cache():
    tree=ast.parse(Path(__file__).with_name('evaluate_ppl_continuous.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ContinuousPCache')
    env=dict(globals(),KVCompressingDynamicCache=FakeBase,TASK_ROOT=Path('.'),
             _replace_dynamic_cache_inplace=replace,recent_window_importance=importance,
             select_split=split,reference_scale_from_original_keys=reference)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'cache','exec'),env)
    return env['ContinuousPCache']

class JournalTests(unittest.TestCase):
    def test_star_conflict_and_modes(self):
        k=torch.tensor([[0.],[-.8],[.8]],dtype=torch.float64);r=torch.zeros(3,dtype=torch.float64);d=r.clone()
        keep=torch.tensor([0]);evict=torch.tensor([1,2]);b,ok,g=route(k,k,r,d,keep,evict,1.,1.)
        self.assertTrue(ok.all())
        chosen,e,_=hard_mis(k,k,r,d,evict,b,ok,g,1.,0.);self.assertEqual(e,1);self.assertEqual(int(chosen.sum()),1)
        # Only the historical cross radius makes this pair incompatible.
        k=torch.tensor([[0.],[.2],[.3]],dtype=torch.float64);r=torch.tensor([0.,.4,.4],dtype=torch.float64);d=r.clone()
        b,ok,g=route(k,k,r,d,keep,evict,1.,1.);self.assertTrue(ok.all())
        for alpha,expected in [(0.,2),(.5,2),(1.,1)]:
            chosen,e,_=hard_mis(k,k,r,d,evict,b,ok,g,.5,alpha);self.assertEqual(int(chosen.sum()),expected)
        # History gate searches a farther compatible anchor rather than dropping nearest.
        k=torch.tensor([[0.],[.3],[.1]],dtype=torch.float64);r=torch.tensor([1.,0.,0.],dtype=torch.float64);d=r.clone()
        b,ok,_=route(k,k,r,d,torch.tensor([0,1]),torch.tensor([2]),.5,1.);self.assertTrue(ok.item());self.assertEqual(b.item(),1)

    def test_weighted_priority(self):
        # Geometric star: leaves mutually compatible, center conflicts with all.
        k=torch.tensor([[0.],[-.8],[-.7],[-.6],[.8]],dtype=torch.float64)
        r=torch.zeros(5,dtype=torch.float64);d=r.clone()
        keep=torch.tensor([0]);evict=torch.tensor([1,2,3,4]);b,ok,g=route(k,k,r,d,keep,evict,1.,0.)
        counts=torch.tensor([1.,1.,1.,1.,20.]);imp=torch.tensor([1.,.1,.15,.2,.8])
        expected={'degree':[True,True,True,False], 'count':[False,False,False,True],
                  'importance':[False,False,False,True], 'count_importance':[False,False,False,True]}
        default,e,_=hard_mis(k,k,r,d,evict,b,ok,g,1.,0.)
        self.assertEqual(e,3)
        for mode in PRIORITY_MODES:
            w=node_weights(mode,counts,imp)
            chosen,e,_=hard_mis(k,k,r,d,evict,b,ok,g,1.,0.,weights=w)
            self.assertEqual(chosen.tolist(),expected[mode]);self.assertEqual(e,3)
            if mode=='degree':self.assertTrue(torch.equal(chosen,default))
            chosen,e,_=hard_mis(k,k,r,d,evict,b,ok,g,2.,0.,weights=w)
            self.assertTrue(chosen.all());self.assertEqual(e,0)
        # Explicitly verify count multiplication, then test signed gains.
        torch.testing.assert_close(node_weights('count_importance',counts,imp),counts.double()*imp.double())
        graph=np.array([[False,True],[True,False]])
        np.testing.assert_allclose(net_gain(graph,[.1,.2]),[-.1,.1])
        # Importance alone and count*importance can choose different nodes.
        k=torch.tensor([[0.],[-.8],[.8]],dtype=torch.float64);r=torch.zeros(3,dtype=torch.float64)
        evict=torch.tensor([1,2]);b,ok,g=route(k,k,r,r,keep,evict,1.,0.)
        counts=torch.tensor([1.,1.,10.]);imp=torch.tensor([1.,.8,.2])
        for mode,expected in [('importance',[True,False]),('count_importance',[False,True])]:
            chosen,_,_=hard_mis(k,k,r,r,evict,b,ok,g,1.,0.,weights=node_weights(mode,counts,imp))
            self.assertEqual(chosen.tolist(),expected)
        # Zero or negative net gain must not stop maximal selection.
        chosen,_,_=hard_mis(k,k,r,r,evict,b,ok,g,1.,0.,weights=torch.zeros(3))
        self.assertEqual(int(chosen.sum()),1)

    def test_priority_sweep(self):
        c=build_configs(args(alpha_ea='0',alpha_ee='1',priority_modes=','.join(PRIORITY_MODES)))
        self.assertEqual(len(c),7)
        self.assertEqual([r['priority_mode'] for r in c[:3]],['','',''])
        self.assertEqual([r['priority_mode'] for r in c[3:]],list(PRIORITY_MODES))
        self.assertEqual(len(build_configs(args(alpha_ea='0',alpha_ee='1',priority_modes='degree,degree'))),4)
        self.assertRaises(ValueError,build_configs,args(priority_modes='invalid'))

    def test_weight_endpoints_and_certification(self):
        w=torch.tensor([2.,1.],dtype=torch.float64);m=torch.tensor([[0.],[1.]],dtype=torch.float64)
        r=torch.tensor([.5,0.],dtype=torch.float64);d=torch.tensor([1.,0.],dtype=torch.float64)
        left=torch.tensor([0]);right=torch.tensor([1])
        # Same geometry for all weights; alpha=0 ignores BOTH D and R.
        for alpha,expected in [(0.,1.),(.25,1.125),(.5,1.25),(1.,1.5)]:
            self.assertAlmostEqual(relation_distance(m,m,r,d,left,right,alpha).item(),expected)
        self.assertRaises(ValueError,relation_distance,m,m,r,d,left,right,1.1)
        # Relaxed gate may admit a merge that exceeds its threshold, but the
        # full certificate must retain the true historical extent.
        best,ok,_=route(m,m,r,d,left,right,1.1,0.)
        self.assertTrue(ok.item())
        _,ok,_=route(m,m,r,d,left,right,1.1,1.)
        self.assertFalse(ok.item())
        mean,rad,diam=propagate(w,m,r,d)
        self.assertAlmostEqual(mean.item(),1/3);self.assertAlmostEqual(diam.item(),1.5)
        raw=torch.tensor([-.5,.5,1.],dtype=torch.float64)
        self.assertGreaterEqual(rad.item()+1e-12,float((raw-mean).abs().max()))
        self.assertGreaterEqual(diam.item()+1e-12,float(raw.max()-raw.min()))
        _,rr,dd=propagate(w[:1],m[:1],r[:1],d[:1]);self.assertEqual(rr.item(),.5);self.assertEqual(dd.item(),1.)
        # Internal D must be scaled too, not silently retained at alpha=0.
        d=torch.tensor([4.,0.],dtype=torch.float64)
        self.assertAlmostEqual(relation_distance(m,m,r,d,left,right,0.).item(),1.)
        self.assertAlmostEqual(relation_distance(m,m,r,d,left,right,.5).item(),2.)

    def test_count_mean(self):
        k=torch.tensor([[0.],[2.],[4.]]);v=k*3;w=torch.tensor([2.,3.,4.]);keep=torch.tensor([0]);evict=torch.tensor([1,2]);best=torch.tensor([0,0]);accepted=torch.tensor([True,False])
        a,b,c=count_mean(k,v,w,keep,evict,best,accepted);self.assertAlmostEqual(a.item(),1.2,places=6);self.assertAlmostEqual(b.item(),3.6,places=6);self.assertEqual(c.item(),5.)
        _,_,c,_,ok=merge_all_baseline(k,v,w,keep,evict,3.);self.assertEqual(c.item(),5.);self.assertEqual(ok.tolist(),[True,False])

    def test_calibration(self):
        k=torch.tensor([[[[0.],[1.],[3.]],[[0.],[2.],[6.]]]],dtype=torch.float64)
        s=key_scales(k);expected=torch.stack([torch.pdist(k[0,h]).square().mean().sqrt() for h in range(2)])
        torch.testing.assert_close(s,expected)
        med,_,_,stack=aggregate_median_scales([s[None],(s*3)[None]]);torch.testing.assert_close(med,s[None]*2)
        payload=dict(schema='diagnosis_median_reference_v1',aggregation='median_across_sequences',frozen=True,calibration_split='train',num_layers=1,num_kv_heads=2,calibration_num_sequences=2,layer_head_scales=med.tolist(),sequence_layer_head_scales=stack.tolist())
        np.testing.assert_allclose(validate_payload(payload),med.numpy());starts=choose_starts(100,4,10,42);self.assertEqual(len(set(starts)),4);self.assertEqual(starts,choose_starts(100,4,10,42))
        payload['calibration_split']='test';self.assertRaises(ValueError,validate_payload,payload)

    def test_sweeps(self):
        c=build_configs(args());self.assertEqual(len(c),28)
        self.assertEqual(len({tuple(v for k,v in x.items() if k!='config_id') for x in c}),len(c))
        for method,expected in [('full_cache',1),('eviction_only',1),('merge_all',1),('coherekv_journal',25)]:
            self.assertEqual(sum(x['method']==method for x in c),expected)
        self.assertTrue(any(x['method']=='coherekv_journal' and x['alpha_ea']==x['alpha_ee']==1 for x in c))
        d=build_configs(args(lambda_ea='.1,.3,.5,.7',lambda_ee='.1,.3,.5,.7'))
        self.assertEqual(len(d),406)
        self.assertEqual(sum(x['method']=='coherekv_journal' for x in d),400)
        p=build_configs(args(alpha_ea='0',alpha_ee='0',reference_modes='online_first,frozen_train',budgets='.1,.2'))
        self.assertEqual(len(p),11)
        self.assertEqual(sum(x['method']=='coherekv_journal' for x in p),4)
        self.assertTrue(all(x['alpha_ea']==x['alpha_ee']==0 for x in p if x['method']=='coherekv_journal'))
        self.assertEqual(len(build_configs(args(alpha_ea='0,0.0',alpha_ee='0,0.0'))),4)
        self.assertRaises(ValueError,build_configs,args(alpha_ea='1.2'))
        self.assertRaises(ValueError,build_configs,args(alpha_ee=''))
        self.assertRaises(ValueError,build_configs,args(lambda_ea='0'))

    def test_real_cache_class_four_rounds(self):
        klass=load_cache();schemas=[]
        for strategy in ('eviction_only','merge_all','coherekv_journal'):
            for alpha_ea,alpha_ee,mode in [(0.,0.,'degree'),(.5,1.,'count'),(1.,0.,'importance'),(1.,1.,'count_importance')] if strategy=='coherekv_journal' else [(1.,1.,'degree')]:
                model=NS(model=NS(layers=[NS(self_attn=NS())]));stats={}
                obj=klass(model=model,method=strategy,budget=.5,priority_mode=mode,alpha_ea=alpha_ea,alpha_ee=alpha_ee,lambda_ea=10.,lambda_ee=10.,reference=np.array([[10.]]),unscored_len=7,scored_len=2,reference_mode='frozen_train',sink_size=0,recent_protect_size=0,stats=stats)
                for n in range(4):
                    start=0 if n==0 else 6+(n-1)*9;end=6+n*9
                    raw=torch.arange(start,end,dtype=torch.float32).view(1,1,-1,1)*.01
                    if n:raw=torch.cat([obj.key_cache[0],raw],dim=-2)
                    obj.key_cache=[raw];obj.value_cache=[raw.clone()];obj._ppl_logical_seen_tokens=end
                    obj.maybe_compress((torch.ones(1,1,2,raw.shape[-2]),),None)
                    self.assertEqual(obj.compress_rounds,n+1);self.assertEqual(obj.key_cache[0].shape[-2],3)
                    self.assertEqual(int(obj.counts[0].sum()),3 if strategy=='eviction_only' else end)
                    self.assertTrue(torch.isfinite(obj.geom_radii[0]).all());self.assertTrue(torch.equal(model.model.layers[0].self_attn._mymethod_counts_kv,obj.counts[0]))
                schemas.extend(set(r) for r in stats['head_rounds'])
                if strategy=='coherekv_journal':
                    self.assertTrue(stats['head_rounds'][0]['history_certified'])
                    self.assertEqual(stats['head_rounds'][0]['threshold_guaranteed'],alpha_ea==alpha_ee==1.)
        self.assertTrue(all(s==schemas[0] for s in schemas))

    def test_pooling_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs=build_configs(args(alpha_ea='.5',alpha_ee='0',methods='full_cache,coherekv_journal',budgets='.5',priority_modes='degree,count_importance'))
            def write(path,rows):
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            for idx in (0,1):
                src=root/'sequences'/f'seq_{idx}'/'ppl_continuous';src.mkdir(parents=True)
                sha=str(idx+1)*64;cfg=dict(sequence_index=idx,token_start=idx*18,token_end=(idx+1)*18,scored_stream_sha256=sha,output_dir=str(src),cycles=2,unscored_len=7,scored_len=2,configurations=configs)
                results=[];cycles=[]
                for c in configs:
                    common=dict(**c,sequence_index=idx,token_start=idx*18,token_end=(idx+1)*18,stream_sha256=sha,cycles=2,unscored_len=7,scored_len=2)
                    nll=1.+idx+(.1 if c['method']!='full_cache' else 0.);full=c['method']=='full_cache';fields={f:0 for f in SUM_FIELDS+FLOAT_SUM_FIELDS}
                    results.append(dict(**common,**fields,scored_tokens=4,ppl=math.exp(nll),nll_sum=4*nll,nll_per_token=nll,seconds=1.,compress_rounds=0 if full else 2,max_before_len=18,max_after_len=18 if full else 3))
                    for j in (1,2):cycles.append(dict(**common,**fields,cycle=j,logical_tokens=j*9,scored_tokens=2,cycle_nll_sum=2*nll,cycle_nll_per_token=nll,cycle_ppl=math.exp(nll),cumulative_nll_per_token=nll,cumulative_ppl=math.exp(nll),compression_rounds=0 if full else j,physical_kv_len=j*9 if full else 6))
                write(src/'ppl_continuous.csv',results);write(src/'ppl_continuous_cycles.csv',cycles)
                (src/'continuous_config.json').write_text(json.dumps(cfg));(src/'continuous_run_complete.json').write_text(json.dumps(dict(config=cfg,n_results=len(results),n_cycles=len(cycles))))
            with contextlib.redirect_stdout(io.StringIO()):aggregate(root,[0,1])
            with (root/'aggregate'/'ppl_sweep_unique.csv').open() as f:rr=list(csv.DictReader(f))
            self.assertAlmostEqual(float(rr[0]['overall_ppl']),math.exp(1.5));self.assertAlmostEqual(float(rr[1]['delta_nll_vs_full']),.1)
            self.assertEqual({r['priority_mode'] for r in rr},{'', 'degree','count_importance'})
            self.assertRaises(ValueError,aggregate,root,[0,0])

if __name__=='__main__':unittest.main(verbosity=2)
