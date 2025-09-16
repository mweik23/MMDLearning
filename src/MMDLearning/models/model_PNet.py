''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.
'''
from os import name
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, impl="auto"):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.impl = impl

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def _graph_feature(self, x, k, idx):
        if self.impl == "gpu":
            return get_graph_feature_v1(x, k, idx)
        if self.impl == "cpu":
            return get_graph_feature_v2(x, k, idx)
        # auto
        return get_graph_feature_v1(x, k, idx) if x.is_cuda else get_graph_feature_v2(x, k, idx)

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self._graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)

class GroupedMLP(nn.Module):
    """
    A multi-layer perceptron broken into named groups of layers.
    
    Args:
        input_dim (int): Size of the input feature vector.
        group_specs (OrderedDict[str, dict]): For each group name, a dict:
            {
              "layers": List[int],      # output dims of successive Linear layers
              "activation": nn.Module,  # activation class (default nn.ReLU)
              "dropout": float,         # dropout p after each activation
            }
        final_activation (nn.Module or None): activation after last group.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 group_specs: OrderedDict,
                 final_activation: nn.Module = None):
        super().__init__()
        self.groups = nn.ModuleDict()
        prev_dim = input_dim

        for name, spec in group_specs.items():
            last_lyr = len(spec['layers'])-1
            layer_dims = spec['layers']
            if name=='head':
                layer_dims.append(num_classes)
            layers = []
            act_cls = spec.get("activation", nn.ReLU)
            do_p    = spec.get("dropout", 0.0)
            for i, out_dim in enumerate(layer_dims):
                layers.append(nn.Linear(prev_dim, out_dim))
                if i<=last_lyr:
                    layers.append(act_cls())
                if do_p > 0:
                    layers.append(nn.Dropout(do_p))
                prev_dim = out_dim

            # wrap the list of layers in a Sequential
            self.groups[name] = nn.Sequential(*layers)

        # optional final activation (e.g. Softmax, Identity, etc.)
        self.final_act = final_activation() if final_activation else nn.Identity()

    def forward(self, x: torch.Tensor, latent_space: str = None) -> torch.Tensor:
        out = x
        group_outputs = {}
        for name, group in self.groups.items():
            out = group(out)
            group_outputs[name] = out
        out = self.final_act(out)
        res = {'out': out}  
        if latent_space is not None:
            if latent_space not in group_outputs:
                raise KeyError(f"Unknown group {latent_space}")
            res[latent_space] = group_outputs[latent_space]
        return res

    def get_group(self, name: str) -> nn.Sequential:
        """Return the Sequential module for a given group."""
        return self.groups[name]

    def parameters_by_group(self):
        """
        Yields (group_name, parameters) so you can build optimizer
        param_groups like:
        
            optim.Adam([
                {"params": p, "lr": lr_backbone},
                {"params": q, "lr": lr_head},
            ])
        """
        for name, module in self.groups.items():
            yield name, module.parameters()

class FullyConnected(nn.Module):

    def __init__(self,
                 input_dims=64,
                 num_classes=None,
                 fc_params=[(64, 0), (128, 0.1)],
                 for_segmentation=False,
                 for_inference=False,
                 out_lyrs = [-1]):

        self.for_inference = for_inference
        self.for_segmentation = for_segmentation

        super().__init__()
        
        fcs = nn.ModuleList()
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx==0:
                in_chn = input_dims
            else:
                in_chn = fc_params[idx-1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if num_classes is not None:
            if self.for_segmentation:
                fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
            else:
                fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        num_lyrs = len(fcs)
        out_lyrs = [(num_lyrs+lyr) % num_lyrs for lyr in out_lyrs]
        self.fc = nn.ModuleList([nn.Sequential(*fcs[:idx+1]) if i==0 else nn.Sequential(*fcs[out_lyrs[i-1]+1:idx+1]) for i, idx in enumerate(out_lyrs)])

    def forward(self, in_feat):

        output=[in_feat]
        for part in self.fc:
            output.append(part(output[-1]))
        if self.for_inference:
            output[-1] = torch.softmax(output[-1], dim=1)

        return output[1:]

def build_mlp(input_dim, hidden_dims, act=nn.ReLU, final_block=False):
    """hidden_dims: list of (channels, dropout) for each layer after the input layer
       final_block: if True, do not add activation/dropout after the last layer
    """
    layers = []
    for i, (channels, dropout) in enumerate(hidden_dims):
        layers += [nn.Linear(input_dim if i == 0 else hidden_dims[i-1][0], channels)]
        is_last = (i == len(hidden_dims) - 1) and final_block
        if not is_last:
            layers += [act()]
        if dropout>0 and (not is_last):
            layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)

class StageBlock(nn.Module):
    """
    A stage block consisting of multiple EdgeConv layers followed by an MLP.
    """
    def __init__(self,
                 input_dims,
                 conv_params=None,  
                 fc_params=None,
                 final_block=False,
                 num_classes=2,
                 aggregate=False,
                 init_block=False,
                 use_counts=True,
                 for_inference=False,
                 freeze_bn=False):
        super().__init__()
        self._freeze_bn = freeze_bn
        self.aggregate = aggregate
        self.use_counts = use_counts
        self.use_fts_bn = init_block
        self.init_block = init_block
        self.final_block = final_block
        self.input_dims = input_dims
        self.for_inference = for_inference
        
        conv_params = list(conv_params or [])
        fc_params = list(fc_params or [])
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)
        self.edge_convs = nn.ModuleList()
        
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels))
       
        if len(conv_params)>0:
            mlp_input_dim = conv_params[-1][1][-1]
        else:
            mlp_input_dim = input_dims
        if len(fc_params)>0:
            if self.final_block:
                fc_params.append((num_classes, 0.0))
            self.mlp = build_mlp(mlp_input_dim, fc_params, final_block=self.final_block)
        self.conv_params = conv_params
        self.fc_params = fc_params
    
    def _set_bn_eval(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def train(self, mode: bool = True):
        # call parent first so all children get toggled
        super().train(mode)
        # then force BN back to eval if requested
        if self._freeze_bn:
            self._set_bn_eval()
        return self
    
    def _infer_output_info(self):
        if len(self.fc_params)>0:
            is_fc = True
            return self.fc_params[-1][0], is_fc
        elif len(self.conv_params)>0:
            is_fc = False
            return self.conv_params[-1][1][-1], is_fc
        return self.input_dims, False
        
    def forward(self, x, points=None, mask=None):
        if len(self.conv_params)>0:
            if mask is None:
                mask = (x.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
            x = x * mask
            
            if self.init_block:
                points = points * mask
            else:
                points = x
            coord_shift = (mask == 0) * 1e9
            if self.use_counts:
                counts = mask.float().sum(dim=-1)
                counts = torch.max(counts, torch.ones_like(counts))  # >=1

            if self.use_fts_bn and self.init_block:
                x = self.bn_fts(x) * mask
                
            for idx, conv in enumerate(self.edge_convs):
                pts = (points if idx == 0 and self.init_block else x) + coord_shift
                x = conv(pts, x) * mask
   
        #TODO: make this more specific to this use case
        if self.aggregate:
            if len(self.conv_params)==0:
                if mask is None:
                    mask = (x.abs().sum(dim=1, keepdim=True) != 0) # (N, 1, P)
                if self.use_counts:
                    counts = mask.float().sum(dim=-1)
                    counts = torch.max(counts, torch.ones_like(counts))  # >=1

            if self.use_counts:
                x = x.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = x.mean(dim=-1)
                
        if len(self.fc_params)>0 and self.mlp is not None:
            x = self.mlp(x)
            if self.for_inference and self.final_block:
                x = torch.softmax(x, dim=1)
        return x

class GroupedParticleNet(nn.Module):

    def __init__(self,
                 input_dims=7,
                 num_classes=2,
                 cfg=None,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 **kwargs):
        super(GroupedParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts
        num_stages = len(cfg['group_order'])
            
        self.stages = nn.ModuleDict()
        cur_dim = input_dims
        seen_fc = False
        for i, name in enumerate(cfg['group_order']):
            conv_params = cfg['group_specs'][name]['conv_params']
            fc_params = cfg['group_specs'][name]['fc_params']
            freeze_bn = cfg['group_specs'][name]['freeze_bn']
            self.stages[name] = StageBlock(
                cur_dim,
                conv_params = conv_params,
                fc_params = fc_params,
                final_block = (i+1==num_stages),
                num_classes = num_classes,
                aggregate = (not seen_fc) and len(fc_params)>0,
                init_block = (i==0),
                for_inference = for_inference,
                freeze_bn = freeze_bn
            )
            cur_dim, is_fc = self.stages[name]._infer_output_info()
            if is_fc:
                seen_fc = True
        
    def param_groups(self):
        return {name: list(stage.parameters()) for name, stage in self.stages.items()}
    
    def forward(self, points, features, mask=None, intermediates=[]):
        x = features
        output = []
        for i, (name, stage) in enumerate(self.stages.items()):
            x = stage(x, points = (points if i==0 else None))
            if name in intermediates:
                output.append(x)
        return x, output
    
    
class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 out_lyrs=[-1],
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)
        self.for_segmentation=for_segmentation
        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.fc_block = FullyConnected(num_classes=num_classes,
                 fc_params=fc_params,
                 for_segmentation=self.for_segmentation,
                 for_inference=for_inference,
                 out_lyrs = out_lyrs)
        

    def forward(self, points, features, mask=None):
#         print('points:\n', points)
#         print('features:\n', features)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)
        
        output = self.fc_block(x)
        # print('output:\n', output)
        return output

class ParticleNetOld(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 num_latent=None,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 intermed_access=False,
                 **kwargs):
        super(ParticleNetOld, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation
        self.intermed = intermed_access
        fcs = nn.ModuleList()
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if num_latent==0:
            if self.for_segmentation:
                fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
            else:
                fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        else:
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(fc_params[-1][0], num_latent, kernel_size=1), nn.ReLU()))
                fcs.append(nn.Conv1d(num_latent, num_classes, kernel_size=1))
            else:
                fcs.append(nn.Sequential(nn.Linear(fc_params[-1][0], num_latent), nn.ReLU()))
                fcs.append(nn.Linear(num_latent, num_classes))
        if self.intermed:
            self.fc = fcs
        else:
            self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
#         print('points:\n', points)
#         print('features:\n', features)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)
        
        if self.intermed:
            x = [x]
            for layer in self.fc:
                x.append(layer(x[-1]))
            output = [x[-3], x[-1]]
        else:
            output = [None, self.fc(x)]
        if self.for_inference:
            output[1] = torch.softmax(output[1], dim=1)
        # print('output:\n', output)
        return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=2)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        return self.pn(points, features, mask)