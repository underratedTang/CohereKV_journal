#!/usr/bin/env python3
"""CPU-only numerical / routing / parallel Soft MIS tests; no model download."""
import math
import unittest
import torch
from soft_mis import (soft_anchor_routes,soft_mis_weights as independent_weights,fuse_soft_slots,
                      merge_weighted_geometry)


def soft_mis_weights(*args, **kwargs):
    # Explicit legacy diagonal (p=gamma) for regression tests.
    return independent_weights(*args,gamma=args[-1],**kwargs)


class SoftMISTests(unittest.TestCase):
    def test_decoupled_ee_decay(self):
        k=torch.tensor([[0.],[-2.],[2.]],dtype=torch.float32)
        z=torch.zeros(3,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1,2])
        best,routed,hard,cert,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,3.,3.)
        args=(k.double(),z,z,keep,evict,best,routed,hard,cert,3.,3.)
        for p in (1.,2.,math.inf):
            for g in (1.,2.,4.,math.inf):
                weights=independent_weights(*args,p,gamma=g).weights
                self.assertAlmostEqual(weights.max().item(),1.)
                self.assertAlmostEqual(weights.min().item(),0. if math.isinf(g) else (.75)**g,places=7)

    def test_decoupled_ea_decay(self):
        k=torch.tensor([[0.],[.5]],dtype=torch.float32)
        radii=torch.tensor([1.5,0.],dtype=torch.float64)
        z=torch.zeros(2,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1])
        best,routed,hard,cert,_=soft_anchor_routes(k,k.double(),radii,z,keep,evict,1.,1.)
        self.assertTrue(routed.item());self.assertFalse(hard.item())
        args=(k.double(),radii,z,keep,evict,best,routed,hard,cert,1.,1.)
        for p in (1.,2.,math.inf):
            for g in (1.,2.,4.,math.inf):
                self.assertAlmostEqual(independent_weights(*args,p,gamma=g).weights.item(),
                                       0. if math.isinf(g) else .5**g,places=8)

    def test_two_conflicting_evicts_soft_vs_hard(self):
        # E1/E2 both near anchor, but E-E distance=4 > 3.
        k=torch.tensor([[0.],[-2.],[2.]],dtype=torch.float32)
        z=torch.zeros(3,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1,2])
        best, routed, hard, gamma, _=soft_anchor_routes(
            k,k.double(),z,z,keep,evict,3.,3.)
        self.assertEqual(routed.tolist(),[True,True])
        self.assertEqual(hard.tolist(),[True,True])
        plan=soft_mis_weights(k.double(),z,z,keep,evict,best,
                              routed,hard,gamma,3.,3.,2)
        self.assertEqual(plan.hard_edges,1)
        self.assertEqual(plan.parallel_rounds,2)
        got=plan.weights.tolist()
        self.assertAlmostEqual(max(got),1.)
        self.assertAlmostEqual(min(got),(3/4)**2,places=8)
        hard_plan=soft_mis_weights(k.double(),z,z,keep,evict,best,
                                   routed,hard,gamma,3.,3.,math.inf)
        self.assertEqual(sorted(hard_plan.weights.tolist()),[0.,1.])

    def test_no_conflict_is_parallel(self):
        k=torch.tensor([[0.,0.],[0.5,0.],[-0.5,0.]],dtype=torch.float32)
        z=torch.zeros(3,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1,2])
        best,routed,hard,gamma,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,2.,2.)
        plan=soft_mis_weights(k.double(),z,z,keep,evict,best,routed,hard,gamma,2.,2.,2)
        self.assertEqual(plan.parallel_rounds,1)
        self.assertEqual(plan.weights.tolist(),[1.,1.])

    def test_physical_nearest_wins_over_historical_compatibility(self):
        # The nearer physical Anchor (A0) has a very broad historical radius;
        # A1 is history-compatible. v2 must still pick A0.
        k=torch.tensor([[0.], [2.], [.75]],dtype=torch.float32)
        centers=k.double()
        rad=torch.tensor([2.,0.,0.],dtype=torch.float64)
        dia=torch.zeros(3,dtype=torch.float64)
        keep=torch.tensor([0,1]);evict=torch.tensor([2])
        best,routed,hard,gamma,physical=soft_anchor_routes(
            k,centers,rad,dia,keep,evict,1.5,2.)
        self.assertEqual(best.tolist(),[0])
        self.assertEqual(routed.tolist(),[True])
        self.assertEqual(hard.tolist(),[False])
        self.assertAlmostEqual(physical.item(),.75,places=6)
        self.assertGreater(gamma.item(),1.5)
        # Historical incompatibility is translated into positive soft weight,
        # not a different anchor or a routing rejection.
        plan=soft_mis_weights(centers,rad,dia,keep,evict,best,routed,
                              hard,gamma,1.5,2.,2)
        self.assertTrue(0.<float(plan.weights[0])<1.)

    def test_physical_gate_drops_even_with_close_historical_centers(self):
        # Large physical L2 distance cannot be rescued by old/history center.
        k=torch.tensor([[0.],[5.]],dtype=torch.float32)
        centers=torch.zeros((2,1),dtype=torch.float64)
        zeros=torch.zeros(2,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1])
        best,routed,hard,gamma,physical=soft_anchor_routes(
            k,centers,zeros,zeros,keep,evict,4.,4.)
        self.assertEqual(best.tolist(),[0])
        self.assertEqual(routed.tolist(),[False])
        self.assertEqual(hard.tolist(),[False])
        self.assertEqual(physical.item(),5.)
        plan=soft_mis_weights(centers,zeros,zeros,keep,evict,best,
                              routed,hard,gamma,4.,4.,2)
        self.assertEqual(plan.weights.tolist(),[0.])

    def test_routing_matches_merge_all_physical_top1(self):
        from p_geometry import physical_l2_routes
        torch.manual_seed(11)
        k=torch.randn((53,6),dtype=torch.float32)
        keep=torch.tensor([0,4,8,12,20]);evict=torch.tensor([i for i in range(53) if i not in keep.tolist()])
        z=torch.zeros(len(k),dtype=torch.float64)
        m1,r1,h1,gamma,physical=soft_anchor_routes(
            k,k.double(),z,z,keep,evict,2.4,2.4,chunk_size=7)
        m2,r2,_=physical_l2_routes(k,keep,evict,2.4,7)
        self.assertTrue(torch.equal(m1,m2))
        self.assertTrue(torch.equal(r1,r2))
        self.assertTrue(torch.equal(h1,r1))

    def test_history_compatible_farther_anchor_is_not_selected(self):
        # This explicitly rules out the old certificate-compatible-first route.
        k=torch.tensor([[0.],[2.],[.75]])
        centers=k.double();radius=torch.tensor([2.,0.,0.],dtype=torch.float64)
        zero=torch.zeros(3,dtype=torch.float64)
        out=soft_anchor_routes(k,centers,radius,zero,torch.tensor([0,1]),
                               torch.tensor([2]),1.5,2.)
        self.assertEqual(out[0].tolist(),[0])

    def test_exact_weighted_p2_center_and_radius(self):
        # Three raw singleton Keys with mass 1, .3 and .4.
        keys=torch.tensor([[0.,0.],[3.,0.],[0.,4.]],dtype=torch.float64)
        masses=torch.tensor([1.,.3,.4],dtype=torch.float64)
        z=torch.zeros(3,dtype=torch.float64)
        W,mu,R,D=merge_weighted_geometry(masses,keys,z,z,2)
        manual_mu=(masses[:,None]*keys).sum(0)/masses.sum()
        var=(masses*(keys-manual_mu).square().sum(-1)).sum()/masses.sum()
        self.assertTrue(torch.allclose(mu,manual_mu,atol=1e-12))
        self.assertAlmostEqual(R.square().item(),var.item(),places=10)
        self.assertAlmostEqual(D.square().item(),2*var.item(),places=10)
        self.assertAlmostEqual(W.item(),1.7)
        # Round-trip: aggregate first two, then merge third; same exact stats.
        W1,mu1,R1,D1=merge_weighted_geometry(masses[:2],keys[:2],z[:2],z[:2],2)
        W2,mu2,R2,D2=merge_weighted_geometry(
            torch.stack((W1,masses[2])),torch.stack((mu1,keys[2])),
            torch.stack((R1,z[2])),torch.stack((D1,z[2])),2)
        self.assertTrue(torch.allclose(mu2,mu,atol=1e-12))
        self.assertAlmostEqual(R2.item(),R.item(),places=10)
        self.assertAlmostEqual(D2.item(),D.item(),places=10)

    def test_p4_is_safe_bound(self):
        # Merging two weighted singleton histories must upper-bound exact p4.
        keys=torch.tensor([[0.],[2.],[5.]],dtype=torch.float64)
        W0,mu0,R0,D0=merge_weighted_geometry(torch.ones(2),keys[:2],
                       torch.zeros(2),torch.zeros(2),4)
        W1,mu1,R1,D1=merge_weighted_geometry(
            torch.tensor([W0.item(),1.]),torch.stack((mu0,keys[2])),
            torch.stack((R0,torch.tensor(0.))),torch.stack((D0,torch.tensor(0.))),4)
        truth=torch.pow(((keys-keys.mean()).abs().pow(4)).mean(),0.25)
        self.assertGreaterEqual(R1.item()+1e-8,truth.item())
        self.assertGreaterEqual(D1.item()+1e-8,torch.pow((keys[:,None]-keys[None,:]).abs().pow(4).mean(),.25).item())

    def test_count_based_fusion_and_true_count(self):
        # C_A=2, C_1=3, C_2=5. A's 2 source tokens and an Evict's 5
        # source tokens are treated as full group masses for this round.
        k=torch.tensor([[0.],[2.],[4.]],dtype=torch.float32)
        v=2*k+1
        keep=torch.tensor([0]);evict=torch.tensor([1,2])
        best=torch.zeros(2,dtype=torch.long)
        count=torch.tensor([2.,3.,5.])
        g=torch.tensor([1.,.25])
        kk,vv,cc,den=fuse_soft_slots(k,v,count,keep,evict,best,g)
        self.assertAlmostEqual(den.item(),2.+3.+.25*5.)
        self.assertAlmostEqual(cc.item(),10.)
        self.assertAlmostEqual(kk.item(),(2*0+3*2+.25*5*4)/(2+3+.25*5),places=6)
        self.assertAlmostEqual(vv.item(),2*kk.item()+1.,places=6)
        # A zero-weight Evict makes no contribution to Count or K/V.
        _,_,cc2,den2=fuse_soft_slots(k,v,count,keep,evict,best,torch.tensor([1.,0.]))
        self.assertEqual(cc2.item(),5.)
        self.assertAlmostEqual(den2.item(),5.)

    def test_count_is_used_instead_of_old_W(self):
        # The previous implementation would use (W_A,W_E)=(20,10).
        # The new implementation MUST use the actual (C_A,C_E)=(100,10).
        k=torch.tensor([[0.],[10.]],dtype=torch.float32)
        c=torch.tensor([100.,10.])
        f=torch.tensor([.5])
        kk,vv,cc,den=fuse_soft_slots(k,k,c,torch.tensor([0]),
                                   torch.tensor([1]),torch.tensor([0]),f)
        self.assertAlmostEqual(kk.item(),50/105,places=6)
        self.assertAlmostEqual(vv.item(),50/105,places=6)
        self.assertAlmostEqual(den.item(),105.)
        self.assertAlmostEqual(cc.item(),110.)

    def test_unit_weights_agree_with_count_mean(self):
        k=torch.tensor([[0.,0.],[10.,0.],[1.,2.],[9.,4.]],dtype=torch.float32)
        v=k*3.+2.
        count=torch.tensor([3.,4.,2.,5.])
        keep=torch.tensor([0,1]); evict=torch.tensor([2,3])
        best=torch.tensor([0,1]); weights=torch.ones(2)
        kk,vv,cc,den=fuse_soft_slots(k,v,count,keep,evict,best,weights)
        expected=torch.stack(((3*k[0]+2*k[2])/5.,(4*k[1]+5*k[3])/9.))
        self.assertTrue(torch.allclose(kk,expected,atol=1e-6))
        self.assertTrue(torch.allclose(vv,expected*3+2,atol=1e-6))
        self.assertEqual(cc.tolist(),[5.,9.])
        self.assertEqual(den.tolist(),[5.,9.])

    def test_geometry_rebases_from_real_count_next_round(self):
        # Round one takes the second singleton with a 0.25 contribution.
        z=torch.zeros(2,dtype=torch.float64)
        k=torch.tensor([[0.],[4.]],dtype=torch.float64)
        temporary,mu,R,D=merge_weighted_geometry(torch.tensor([1.,.25]),k,z,z,2)
        self.assertAlmostEqual(temporary.item(),1.25)
        self.assertAlmostEqual(mu.item(),.8)
        # Its NEXT-round Count is 2, not 1.25. A new singleton at Key 6
        # must be fused with 2:1, not 1.25:1; the updated history radius
        # is likewise propagated with Count=2 rather than old mass 1.25.
        nxt,mu2,R2,D2=merge_weighted_geometry(
            torch.tensor([2.,1.]),torch.stack((mu,k[1]+2)),
            torch.stack((R,z[0])),torch.stack((D,z[0])),2)
        self.assertAlmostEqual(nxt.item(),3.)
        self.assertAlmostEqual(mu2.item(),(2*.8+6)/3.,places=10)
        expected_r2=(2*(R.item()**2+(.8-mu2.item())**2)
                     +(6-mu2.item())**2)/3.
        self.assertAlmostEqual(R2.square().item(),expected_r2,places=10)
        self.assertAlmostEqual(D2.square().item(),2*expected_r2,places=10)

    def test_different_owners(self):
        k=torch.tensor([[0.],[10.],[1.],[9.]],dtype=torch.float32)
        z=torch.zeros(4,dtype=torch.float64)
        keep=torch.tensor([0,1]); evict=torch.tensor([2,3])
        best,routed,hard,gamma,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,2,2)
        self.assertEqual(best.tolist(),[0,1])
        plan=soft_mis_weights(k.double(),z,z,keep,evict,best,routed,hard,gamma,2,2,2)
        self.assertEqual(plan.weights.tolist(),[1.,1.])
        self.assertEqual(plan.parallel_rounds,2)  # one independent layer per owner

    def test_importance_mass_delta_changes_conflict_winner(self):
        # Degree-first chooses E1 (closer), importance-mass priority E2.
        k=torch.tensor([[0.],[-1.],[2.]])
        z=torch.zeros(3,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1,2])
        best,routed,hard,gamma,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,3.,3.)
        args=(k.double(),z,z,keep,evict,best,routed,hard,gamma,3.,2.)
        counts=torch.tensor([1.,1.,4.]);importance=torch.tensor([1.,.8,.4])
        for p in (2.,math.inf):
            old=soft_mis_weights(*args,p).weights
            new=soft_mis_weights(*args,p,priority='importance_mass_delta',
                                 counts=counts,importance=importance).weights
            self.assertAlmostEqual(old[0].item(),1.)
            self.assertAlmostEqual(new[1].item(),1.)
            self.assertLess(new[0].item(),old[0].item())
            self.assertGreater(new[1].item(),old[1].item())
            q=(counts*importance)[evict].double()
            self.assertGreater(float(q@new),float(q@old))
            if math.isinf(p):self.assertEqual(new.tolist(),[0.,1.])

    def test_importance_delta_counts_conflicting_neighbors(self):
        # Path: leaves each q=3, center q=4. The center has the largest
        # individual importance but net score -2 vs each leaf -1.
        # E1 and E3 are compatible; both conflict with E2.
        k=torch.tensor([[0.,0.],[-.5,0.],[0.,2.],[.5,0.]])
        z=torch.zeros(4,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1,2,3])
        best,routed,hard,gamma,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,3.,3.)
        plan=soft_mis_weights(k.double(),z,z,keep,evict,best,routed,hard,gamma,
                              3.,1.5,math.inf,priority='importance_mass_delta',
                              counts=torch.ones(4),importance=torch.tensor([1.,3.,4.,3.]))
        self.assertEqual(plan.weights.tolist(),[1.,0.,1.])
        plan=soft_mis_weights(k.double(),z,z,keep,evict,best,routed,hard,gamma,
                              3.,1.5,math.inf,priority='importance_mass_delta',
                              counts=torch.ones(4),importance=torch.tensor([1.,3.,9.,3.]))
        self.assertEqual(plan.weights.tolist(),[0.,1.,0.])

    def test_continuous_importance_uses_edge_strength(self):
        # Path with equal edge cross^2=4.25, delta^2=2.25.
        # Binary net prefers leaves (3-4=-1 > 4-6=-2), whereas
        # continuous net prefers center (4-6*c > 3-4*c).
        k=torch.tensor([[0.,0.],[-.5,0.],[0.,2.],[.5,0.]])
        z=torch.zeros(4,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1,2,3])
        best,routed,hard,gamma,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,3.,3.)
        args=(k.double(),z,z,keep,evict,best,routed,hard,gamma,3.,1.5)
        kw=dict(priority='importance_mass_delta',counts=torch.ones(4),importance=torch.tensor([1.,3.,4.,3.]))
        for chunk in (1,2,256):
            plan=soft_mis_weights(*args,2.,pair_chunk=chunk,**kw)
            torch.testing.assert_close(plan.weights,torch.tensor([2.25/4.25,1.,2.25/4.25],dtype=torch.float64),rtol=1e-6,atol=1e-7)
        self.assertEqual(soft_mis_weights(*args,math.inf,**kw).weights.tolist(),[1.,0.,1.])

    def test_unit_node_mass_recovers_degree_priority(self):
        # Verify exact equality on multiple graphs and historical soft-only
        # candidates, not just a two-node example.
        for seed in range(8):
            gen=torch.Generator().manual_seed(seed)
            k=torch.randn(25,3,generator=gen)
            centers=k.double()+.04*torch.randn(25,3,generator=gen).double()
            radii=torch.rand(25,generator=gen).double()*.25
            diameters=radii*2
            keep=torch.arange(4);evict=torch.arange(4,25)
            best,routed,hard,gamma,_=soft_anchor_routes(k,centers,radii,diameters,keep,evict,2.,2.)
            args=(centers,radii,diameters,keep,evict,best,routed,hard,gamma,2.,1.)
            for p in (math.inf,):
                a=soft_mis_weights(*args,p,pair_chunk=5)
                b=soft_mis_weights(*args,p,pair_chunk=5,priority='importance_mass_delta',
                                   counts=torch.ones(25),importance=torch.ones(25))
                torch.testing.assert_close(a.weights,b.weights,rtol=0,atol=0)
                self.assertEqual(a.parallel_rounds,b.parallel_rounds)
                self.assertEqual(a.ee_soft_slots,b.ee_soft_slots)

    def test_importance_priority_requires_valid_inputs(self):
        k=torch.tensor([[0.],[1.]])
        z=torch.zeros(2,dtype=torch.float64)
        keep=torch.tensor([0]);evict=torch.tensor([1])
        best,routed,hard,gamma,_=soft_anchor_routes(k,k.double(),z,z,keep,evict,2.,2.)
        args=(k.double(),z,z,keep,evict,best,routed,hard,gamma,2.,1.,2.)
        with self.assertRaises(ValueError):
            soft_mis_weights(*args,priority='importance_mass_delta')
        with self.assertRaises(ValueError):
            soft_mis_weights(*args,priority='importance_mass_delta',counts=torch.ones(2),importance=torch.tensor([1.,-1.]))


if __name__=='__main__':unittest.main()
