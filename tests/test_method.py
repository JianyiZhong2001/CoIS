"""Mathematical regressions for the extracted method and packaging adapter."""
import itertools
import unittest
import torch

from cois import cois_loss, source_context_probabilities
from cois.losses.ecoc import similarities, segmentation_loss
from cois.losses.context_interval import target_loss
from cois.losses.context_additive import mixed_loss
from cois.losses.interval_code_set import interval_candidates, matched_topk, set_mass_loss, code_set_loss


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2333)
        torch.set_num_threads(2)
        self.codes = torch.tensor([[0,0,0,0], [0,1,1,0], [1,0,1,0],
                                   [1,1,0,1], [1,1,1,1]], dtype=torch.float64)

    def test_pairwise_bounds_against_exhaustive_corners(self):
        corners = torch.tensor(list(itertools.product([False, True], repeat=4)))
        for _ in range(20):
            contexts = torch.rand(2,1,4,1,1, dtype=torch.float64)
            low, high = contexts.amin(0)[0,:,0,0], contexts.amax(0)[0,:,0,0]
            points = torch.where(corners, high, low)[:,:,None,None]
            scores = similarities(points, self.codes)[:,:,0,0]
            brute = ((scores[:,:,None]-scores[:,None,:]).amax(0) >= -1e-6).all(1)
            original = similarities(contexts[0], self.codes).argmax(1)
            brute[original.item()] = True
            actual, _ = interval_candidates(contexts, self.codes, original)
            self.assertTrue(torch.equal(actual[0,:,0,0], brute))
            interior = low + (high-low)*torch.rand(100,4,dtype=torch.float64)
            winners = similarities(interior[:,:,None,None], self.codes).argmax(1)
            self.assertTrue(actual[0,:,0,0][winners].all())

    def test_interval_gradient_direction_and_mask(self):
        logits = torch.tensor([[[[-2.,0.,2.,2.]]]], dtype=torch.float64, requires_grad=True)
        contexts = torch.stack([torch.full_like(logits,.3),torch.full_like(logits,.7)]).requires_grad_()
        weights = torch.tensor([[[1.,1.,1.,0.]]], dtype=torch.float64)
        loss, _ = target_loss(logits, contexts, weights, "interval")
        grad, teacher_grad = torch.autograd.grad(loss, (logits,contexts), allow_unused=True)
        self.assertLess(grad[0,0,0,0],0)
        self.assertEqual(grad[0,0,0,1],0)
        self.assertGreater(grad[0,0,0,2],0)
        self.assertEqual(grad[0,0,0,3],0)
        self.assertIsNone(teacher_grad)

    def test_set_mass_full_set_zero_and_target_masking(self):
        logits = torch.randn(2,4,3,3,dtype=torch.float64,requires_grad=True)
        selected = torch.zeros(2,5,3,3,dtype=torch.bool)
        selected[:,:2] = True
        weights = torch.ones(2,3,3,dtype=torch.float64)
        weights[1] = 0
        loss = set_mass_loss(logits, selected, weights, self.codes, .5)
        grad, = torch.autograd.grad(loss, logits)
        self.assertEqual(grad[1].abs().sum(),0)
        self.assertGreater(grad[0].abs().sum(),0)
        self.assertLess(set_mass_loss(logits.detach()-.01*grad, selected, weights, self.codes,.5),loss)
        zero = set_mass_loss(logits, torch.ones_like(selected), weights, self.codes,.5)
        zero_grad, = torch.autograd.grad(zero,logits)
        self.assertEqual(zero,0)
        self.assertEqual(zero_grad.abs().sum(),0)

    def test_teacher_topk_preserves_count_and_rank(self):
        contexts = torch.rand(2,2,4,3,3,dtype=torch.float64)
        original = torch.rand(2,4,3,3,dtype=torch.float64)
        winners = similarities(original,self.codes).argmax(1)
        candidates,_ = interval_candidates(contexts,self.codes,winners)
        selected = matched_topk(candidates,original,self.codes)
        self.assertTrue(torch.equal(selected.sum(1),candidates.sum(1)))
        self.assertTrue(selected.gather(1,winners[:,None]).all())
        rank = similarities(original,self.codes).argsort(dim=1,descending=True,stable=True)
        expected = torch.arange(5)[None,:,None,None] < candidates.sum(1,keepdim=True)
        self.assertTrue(torch.equal(selected.gather(1,rank),expected))

    def test_adapter_matches_original_training_composition_and_gradients(self):
        source = tuple(torch.randn(2,4,3,3,dtype=torch.float64,requires_grad=True) for _ in range(2))
        mixed = tuple(torch.randn_like(x,requires_grad=True) for x in source)
        bits = torch.randint(0,2,source[0].shape).double()
        weights = torch.ones(2,3,3,dtype=torch.float64)
        target_weights = weights.clone(); target_weights[:,:,0] = 0
        contexts = torch.rand(2,2,4,3,3,dtype=torch.float64,requires_grad=True)
        original = torch.rand(2,4,3,3,dtype=torch.float64,requires_grad=True)
        source_loss,_ = segmentation_loss(source,bits,weights,self.codes)
        mixed_loss_value,_ = mixed_loss(mixed,bits,weights,target_weights,contexts,self.codes,"interval")
        extra,_ = code_set_loss(mixed,contexts,original,similarities(original,self.codes).argmax(1),
                                target_weights,self.codes,"topk_set")
        expected = 2*source_loss + 2*(mixed_loss_value+extra)
        actual,terms = cois_loss(source,mixed,bits,weights,bits,weights,target_weights,contexts,original,self.codes)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        grads_a = torch.autograd.grad(actual,source+mixed,retain_graph=True)
        grads_b = torch.autograd.grad(expected,source+mixed,retain_graph=True)
        for a,b in zip(grads_a,grads_b):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertTrue(all(x is None for x in torch.autograd.grad(actual,(contexts,original),allow_unused=True)))
        baseline,_ = cois_loss(source,mixed,bits,weights,bits,weights,target_weights,contexts,original,self.codes,cis=False,igss=False)
        plain,_ = segmentation_loss(mixed,bits,weights,self.codes)
        torch.testing.assert_close(baseline,2*source_loss+2*plain)

    def test_context_preserves_teacher_bn_and_modes(self):
        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.dropout = torch.nn.Dropout(.5)
            def forward(self,x,return_feat=False):
                y = self.dropout(self.bn(x))
                return y,y
        teacher = Teacher().train()
        teacher.dropout.eval()
        modes = [x.training for x in teacher.modules()]
        state = {k:v.clone() for k,v in teacher.state_dict().items()}
        target = torch.randn(2,3,3,3)
        sources = torch.randn_like(target)
        mask = torch.zeros(2,3,3,dtype=torch.bool)
        probabilities = source_context_probabilities(teacher,target,sources,mask)
        torch.testing.assert_close(probabilities[0],probabilities[1],atol=0,rtol=0)
        self.assertFalse(probabilities.requires_grad)
        self.assertEqual(modes,[x.training for x in teacher.modules()])
        for k,v in state.items():
            torch.testing.assert_close(v,teacher.state_dict()[k],atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main()
