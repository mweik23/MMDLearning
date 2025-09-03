# src/yourpkg/models/predictors.py
import torch
from typing import Dict, Any, Tuple

class Predictor:
    def prepare_batch(self, batch: Dict[str, Any], device, dtype) -> Dict[str, Any]:
        raise NotImplementedError
    def forward(self, ddp_model: torch.nn.Module, p: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class LorentzNetPredictor(Predictor):
    def prepare_batch(self, batch, device, dtype):
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
    def forward(self, ddp_model, p):
        pred, mmd_in = ddp_model(
            scalars=p["scalars"], x=p["x"], edges=p["edges"],
            node_mask=p["node_mask"], edge_mask=p["edge_mask"], n_nodes=p["n_nodes"]
        )
        return pred, mmd_in

class ParticleNetPredictor(Predictor):
    def prepare_batch(self, batch, device, dtype):
        return {
            "points":   batch['points'].to(device, dtype),
            "features": batch['features'].to(device, dtype),
            "mask":     batch['label'].to(device, dtype),
            "label":    batch['is_signal'].to(device, dtype).long(),
            "is_target": batch['is_target'].to(device, dtype),
            "n_s": batch['n_s'].to(device, dtype)
        }
    def forward(self, ddp_model, p):
        mmd_in, pred = ddp_model(p["points"], p["features"], mask=p["mask"])
        return pred, mmd_in

def make_predictor(model_name: str) -> Predictor:
    m = model_name.lower()
    if m in ("lorentznet", "LorentzNet", "Lorentznet"):
        return LorentzNetPredictor()
    if m in ("particlenet", "particlenet-lite", "ParticleNet", "ParticleNet-Lite", 
             "Particlenet", "Particlenet-Lite", "Particlenet-lite"):
        return ParticleNetPredictor()
    raise ValueError(f"Unknown model: {model_name}")
