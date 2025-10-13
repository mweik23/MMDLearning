# src/yourpkg/models/predictors.py
import torch.nn as nn
import torch
import importlib
from typing import Dict, Any, Tuple

SRC_DIR = (Path(__file__).parents[1]).resolve()
import sys
sys.path.append(str(SRC_DIR))

from utils.model_utils import unwrap
from training.training_utils import split_st, merge_st

# ----------- Model Registry (update as needed) -----------
MODEL_REGISTRY = {
    "particlenet-lite": ".model_PNet:GroupedParticleNet",
    "particlenet": ".model_PNet:GroupedParticleNet",
    "lorentznet": ".model_LNet:LorentzNet",
}
#-----------------------------------------------------------
class BasePredictor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _load_model(model_name: str, **kwargs):
        spec = MODEL_REGISTRY[model_name.lower()]
        module, cls = spec.split(":")
        Mod = importlib.import_module(module, package=__package__)
        ModelCls = getattr(Mod, cls)
        return ModelCls(**kwargs)

class ParticleNetPredictor(BasePredictor):
    def __init__(self, 
                 model_name: str = 'ParticleNet', 
                 target_encoder_groups: Tuple[str]=(), 
                 encoder_layer: str = 'encoder',
                 **model_kwargs):
        super().__init__()
        self.model = self._load_model(model_name, **model_kwargs)
        self.groups = model_kwargs['groups']
        self.tap_keys = model_kwargs.get('tap_keys', ())
        if len(target_encoder_groups)>0:
            target_kwargs = model_kwargs.copy()
            target_kwargs['groups'] = target_encoder_groups
            self.joint = [group not in target_encoder_groups for group in self.groups]
            if self.joint[0]:
                target_kwargs['input_dims'], target_kwargs['seen_fc'] = self.model.stages[
                    self.groups[self.joint.index(False)-1]
                ]._infer_output_info()
            print('creating target model with args:', target_kwargs)
            self.target_model = self._load_model(model_name, **target_kwargs)
            self.pipeline = []
            curr_joint = False
            groups = []
            seen_encoder = False
            for joint, group in zip(self.joint, self.groups):
                if joint != curr_joint:
                    self.add_runner(groups, curr_joint, seen_encoder) # only adds if groups non-empty
                    seen_encoder = seen_encoder or (encoder_layer in groups)
                    groups = [group]
                    curr_joint = joint
                else:
                    groups.append(group)
            # for the groups left at the end of the loop
            self.add_runner(groups, curr_joint, seen_encoder)
        else:
            self.target_model = None
            self.pipeline = None

    def add_runner(self, groups, curr_joint, seen_encoder):
        if len(groups) > 0:
            if curr_joint:
                self.pipeline.append(JointRunner(tuple(groups), tap_keys=(k for k in self.tap_keys if k in groups), after_encoder=seen_encoder))   
            else:
                self.pipeline.append(SplitRunner(tuple(groups), tap_keys=(k for k in self.tap_keys if k in groups), after_encoder=seen_encoder))

    @property
    def stages(self):
        return self.model.stages
    
    @property
    def stages_encoder(self):
        if self.target_encoder is not None:
            return self.target_encoder.stages
        return None
    
    @staticmethod  
    def prepare_batch(batch, device, dtype):
        return {
            "points":   batch['points'].to(device, dtype),
            "features": batch['features'].to(device, dtype),
            "mask":     batch['label'].to(device, dtype),
            "is_signal": batch['is_signal'].to(device, dtype).long(),
            "is_target": batch['is_target'].to(device, dtype),
            "n_s": batch['n_s'] if 'n_s' in batch else None
        }
    
    def encode_target(self, p, **kw):
        if self.target_encoder is None:
            raise ValueError("Target encoder not defined.")
        for use_sec, group in zip(self.use_secondary, self.groups):
            if use_sec:
                p['features'] = self.target_encoder.stages[group](**p)
            else:
                p['features'] = self.model.stages[group](**p)
            if group == 'encoder':
               return p['features']

    def forward(self, x, x_sec=None, domains=('Source', 'Target'), target_preds=False, **kw):
        taps={}
        if self.pipeline is None:
            split_output = False
            if x is not None and x_sec is not None:
                split_output=True
                x = merge_st(x, x_sec, ns=x['n_s'] or len(x['is_signal']))
                x_sec = None
                assert domains==('Source', 'Target'), "If both x and x_sec are provided, domains must be ('Source', 'Target')"
            elif x_sec is not None:
                x = x_sec
                x_sec = None
                assert domains==('Target',), "If x is None and x_sec is not None, domains must be ('Target',)"
            if domains==('Source', 'Target') and not target_preds:
                split_output = True
            pred, tap_values = self.model(**x, tap_keys=self.tap_keys)
            
            for k, v in zip(self.tap_keys, tap_values):
                v_spl = split_st(v)
                taps[k] = {d: v_spl[0] if d == 'Source' else v_spl[1] for d in domains}
            if split_output:
                pred = pred[:x['n_s']]
                pred_sec = pred[x['n_s']:]
            elif domains==('Source',):
                pred_sec = None
            elif domains==('Target',):
                pred_sec = pred
                pred = None
            if not target_preds:
                pred_sec = None
        else:
            if x is not None and x_sec is not None:
                split_output=True
            for runner in self.pipeline:
                x, x_sec, taps = runner(x, x_sec=x_sec, owner=self, taps=taps, domains=domains, target_preds=target_preds)
            if domains==('Source',):
                pred = x['features']
                pred_sec = None
            elif domains==('Target',):
                pred_sec = x_sec['features']
                pred = None
            elif domains==('Source', 'Target'):
                if x_sec is not None and not split_output:
                    pred = merge_st(x, x_sec, ns=x['n_s'] or len(x['is_signal']))['features']
                    pred_sec = None
                elif x_sec is None and split_output:
                    (x, x_sec) = split_st(x, domains=domains)
                    pred = x['features']
                    pred_sec = x_sec['features']
                else:
                    pred = x['features'] if x is not None else None
                    pred_sec = x_sec['features'] if x_sec is not None else None
        return (pred, pred_sec), taps

class LorentzNetPredictor(BasePredictor):
    def __init__(self, model_name: str = 'LorentzNet', **model_kwargs):
        super().__init__()
        self.model = self._load_model(model_name, **model_kwargs)
    
    @staticmethod 
    def prepare_batch(batch, device, dtype):
        B, N, _ = batch['Pmu'].size()
        return {
            "scalars":   batch['nodes'].view(B*N, -1).to(device, dtype),
            "x":         batch['Pmu'].view(B*N, -1).to(device, dtype),
            "edges":     [e.to(device) for e in batch['edges']],
            "node_mask": batch['atom_mask'].view(B*N, -1).to(device),
            "edge_mask": batch['edge_mask'].reshape(B*N*N, -1).to(device),
            "n_nodes":   N,
            "label":     batch['is_signal'].to(device, dtype).long(),
            "is_source": batch['is_source'],
        }
    def forward(self, p, **kw):
        pred, mmd_in = self.model(
            scalars=p["scalars"], x=p["x"], edges=p["edges"],
            node_mask=p["node_mask"], edge_mask=p["edge_mask"], n_nodes=p["n_nodes"]
        )
        return pred, mmd_in
    
def make_predictor(model_name: str, **kwargs) -> BasePredictor:
    m = model_name.lower()
    if m in ("lorentznet"):
        return LorentzNetPredictor(**kwargs)
    if m in ("particlenet", "particlenet-lite"):
        return ParticleNetPredictor(**kwargs)
    raise ValueError(f"Unknown model: {model_name}")

class JointRunner(torch.nn.Module):
    def __init__(self, names: Tuple[str], tap_keys: Tuple[str]=(), after_encoder=False):
        super().__init__()
        self.names = names
        self.tap_keys = tap_keys
        self.after_encoder = after_encoder
    def forward(self, x_prim, x_sec=None, owner=None, taps=None, target_preds=False, domains=('Source', 'Target')):  # mask can be None for single-domain
        if self.after_encoder and not target_preds:
            if len(domains)==2 and x_sec is None:
                (x_prim, x_sec) = split_st(x_prim, domains=domains)
            x_sec = None
            domains = tuple(d for d in domains if d!='Target')
        assert len(domains)>0, "No domains left to process."
        assert x_prim is not None or x_sec is not None, "At least one of x_prim or x_sec must be provided."
        if x_sec is not None and x_prim is not None:
            x_prim = merge_st(x_prim, x_sec, ns=x_prim['n_s'])
            x_sec = None
        elif x_sec is not None:
            assert domains==('Target',), "If x_prim is None, domains must be ('Target',)"
            x_prim = x_sec
            x_sec = None
        if len(domains)==1:
            x_prim['n_s'] = len(x_prim['is_signal']) if domains[0]=='Source' else 0 
        for name in self.names:
            x_prim['features'] = unwrap(owner).model.stages[name](**x_prim)
            if name in self.tap_keys:
                if taps is None:
                    taps = {}
                x = split_st(x_prim, domains=domains)
                taps[name] = {d:x[0]['features'] if d=='Source' else x[1]['features'] for d in domains}
        if domains==('Target',):
            x_sec = x_prim
            x_prim = None
        return x_prim, x_sec, taps

class SplitRunner(torch.nn.Module):
    def __init__(self, names: Tuple[str], tap_keys: Tuple[str]=(), after_encoder=False):
        super().__init__()
        self.names = names
        self.tap_keys = tap_keys
        self.after_encoder = after_encoder
    def forward(self, x_prim=None, x_sec=None, owner=None, taps=None, target_preds=False, domains=('Source', 'Target')):  # mask can be None for single-domain
        if x_sec is None and len(domains)==2:
            (x_prim, x_sec) = split_st(x_prim, domains=domains)
        if self.after_encoder and not target_preds:
            x_sec = None
            domains = tuple(d for d in domains if d!='Target')
        for name in self.names:
            if x_prim is not None:
                x_prim['features'] = unwrap(owner).model.stages[name](**x_prim)
            if x_sec is not None:
                x_sec['features'] = unwrap(owner).target_model.stages[name](**x_sec)
            if name in self.tap_keys:
                if taps is None:
                    taps = {}
                taps[name] = {d: x_prim['features'] if d=='Source' else x_sec['features'] for d in domains}
        return x_prim, x_sec, taps