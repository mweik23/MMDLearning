# src/yourpkg/models/predictors.py
import torch
import torch.nn as nn
import importlib
from typing import Dict, Any, Tuple

# ----------- Model Registry (update as needed) -----------
MODEL_REGISTRY = {
    "particlenet-lite": "model_PNet:GroupedParticleNet",
    "particlenet": "model_PNet:GroupedParticleNet",
    "lorentznet": "model_LNet:LorentzNet",
}
#-----------------------------------------------------------
class BasePredictor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _load_model(model_name: str, **kwargs):
        spec = MODEL_REGISTRY[model_name.lower()]
        module, cls = spec.split(":")
        Mod = importlib.import_module(module)
        ModelCls = getattr(Mod, cls)
        return ModelCls(**kwargs)

class ParticleNetPredictor(BasePredictor):
    def __init__(self, model_name: str = 'ParticleNet', **model_kwargs):
        super().__init__()
        self.model = self._load_model(model_name, **model_kwargs)
    
    @staticmethod  
    def prepare_batch(batch, device, dtype):
        return {
            "points":   batch['points'].to(device, dtype),
            "features": batch['features'].to(device, dtype),
            "mask":     batch['label'].to(device, dtype),
            "is_signal": batch['is_signal'].to(device, dtype).long(),
            "is_target": batch['is_target'].to(device, dtype),
            "n_s": batch['n_s'].to(device, dtype)
        }
    def forward(self, p, **kw):
        pred, int_out = self.model(p["points"], p["features"], mask=p["mask"], **kw)
        return pred, *int_out

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
