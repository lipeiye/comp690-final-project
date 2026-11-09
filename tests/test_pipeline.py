from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mibci import data,deploy
from mibci.costs import cost_summary
from mibci.model import build_eegnet
from mibci.prune import prune_pointwise
from mibci.splits import fold_masks
from mibci.train import ChannelStandardizer,predict,set_seed


class PipelineInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_heldout_session_requires_explicit_final_evaluation(self):
        with self.assertRaises(PermissionError):data.load_session(1,2)

    def test_run_folds_are_disjoint_and_cover_every_trial_once(self):
        runs=np.repeat(np.arange(6),48)
        heldout=[]
        for fold in range(3):
            train,val=fold_masks(runs,fold)
            self.assertFalse(np.any(train&val))
            self.assertTrue(np.all(train|val))
            self.assertFalse(set(runs[train])&set(runs[val]))
            heldout.append(val)
        np.testing.assert_array_equal(np.sum(heldout,axis=0),np.ones(len(runs)))

    def test_validation_values_do_not_change_training_normalization(self):
        rng=np.random.default_rng(1)
        train=rng.normal(size=(8,3,500)).astype(np.float32)
        scaler=ChannelStandardizer().fit(train)
        mean,std=scaler.mean.copy(),scaler.std.copy()
        scaler.transform(np.full((4,3,500),1e6,dtype=np.float32))
        np.testing.assert_array_equal(scaler.mean,mean)
        np.testing.assert_array_equal(scaler.std,std)
        np.testing.assert_allclose(scaler.transform(train).mean(axis=(0,2)),0,atol=1e-6)

    def test_structured_pruning_reduces_real_tensors_and_keeps_valid_outputs(self):
        set_seed(0);model=build_eegnet(22).eval()
        pruned=prune_pointwise(model,.5).eval()
        self.assertEqual(pruned.F2,8)
        self.assertEqual(pruned.conv_separable_point.weight.shape[0],8)
        self.assertEqual(pruned.bnorm_2.num_features,8)
        self.assertEqual(sum(p.numel() for p in pruned.parameters()),1796)
        logits=predict(pruned,np.zeros((2,22,500),dtype=np.float32))
        self.assertEqual(logits.shape,(2,4));self.assertTrue(np.isfinite(logits).all())

    def test_export_parity_quantization_and_actual_float32_input_contract(self):
        set_seed(1);model=build_eegnet(8).eval()
        X=np.random.default_rng(2).normal(size=(8,8,500)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            fp32,int8=Path(temp)/"model.onnx",Path(temp)/"model_int8.onnx"
            deploy.export_onnx(model,fp32)
            reference=predict(model,X)
            actual=deploy.logits(deploy.session(fp32),X)
            np.testing.assert_allclose(actual,reference,atol=1e-5,rtol=1e-4)
            deploy.quantize_int8(fp32,int8,X)
            session=deploy.session(int8)
            self.assertEqual(session.get_inputs()[0].type,"tensor(float)")
            quantized=deploy.logits(session,X)
            self.assertEqual(quantized.shape,reference.shape)
            self.assertTrue(np.isfinite(quantized).all())
        cost=cost_summary(model)
        self.assertEqual(cost["input_bytes"]["int8"],X[:1].nbytes)
        self.assertEqual(cost["modeled_input_bytes"]["int8"]*4,X[:1].nbytes)


if __name__=="__main__":unittest.main()
