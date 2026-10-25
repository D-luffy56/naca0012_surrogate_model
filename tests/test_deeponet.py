"""Run with python -m unittest discover -s tests -v."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import io
import numpy as np
import torch
from coefficient_deeponet import CoefficientDeepONet, fit_coefficients
from sst_deeponet_physics import make_residuals

class DeepONetTests(unittest.TestCase):
    def test_shape_single_batch_and_gradients(self):
        torch.manual_seed(42)
        net=CoefficientDeepONet(5,latent=8);x=torch.randn(3,2)
        self.assertEqual(net(x).shape,(3,5))
        torch.testing.assert_close(net(x[0]),net(x)[0])
        net(x).square().mean().backward()
        for part in [net.branch,net.trunk]:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in part.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters()),0)
        with self.assertRaises(ValueError): net(torch.randn(3,7))
    def test_roundtrip(self):
        net=CoefficientDeepONet(5,latent=8);other=CoefficientDeepONet(5,latent=8)
        buf=io.BytesIO();torch.save(net.state_dict(),buf);buf.seek(0)
        other.load_state_dict(torch.load(buf,weights_only=True))
        x=torch.randn(2,2);torch.testing.assert_close(net(x),other(x),rtol=0,atol=0)
    def test_training_reduces_known_target_error(self):
        rng=np.random.default_rng(11);x=rng.uniform(-1,1,(30,2)).astype('float32')
        y=np.column_stack((x[:,0]+x[:,1],x[:,0]*x[:,1],x[:,0]**2)).astype('float32')
        net,_=fit_coefficients(x,y,hidden_units=(12,12),latent=8,adam_steps=100,lbfgs_steps=100)
        with torch.no_grad(): mse=float(((net(torch.tensor(x))-torch.tensor(y))**2).mean())
        self.assertLess(mse,.001)
    def test_physics_backpropagates_through_pod(self):
        n=9;matrix=torch.zeros(n,n)
        for i in range(n): matrix[i,i]=-1;matrix[i,(i+1)%n]=1
        residual=make_residuals(matrix.to_sparse(),(.7*matrix).to_sparse(),np.ones(n))
        net=CoefficientDeepONet(3,hidden_units=(8,8),latent=4)
        modes=torch.randn(3,6*n)*.001
        fields=(torch.ones(6*n)+net(torch.ones(2,2))@modes).reshape(2,6,n)
        terms=residual(fields,torch.tensor([2e6,3e6]));loss=sum(t.square().mean() for t in terms.values())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.parameters()),0)

if __name__=='__main__': unittest.main()
