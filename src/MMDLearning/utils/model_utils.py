def print_stage_param_summary(model, name='Source Model'):
    model_total = sum(p.numel() for p in model.parameters())
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'Stage':<15}{'Total Params':>15}{'Trainable':>15}")
    print("-" * 45)

    stage_total_sum = 0
    stage_trainable_sum = 0

    for name, stage in model.stages.items():
        total = sum(p.numel() for p in stage.parameters())
        trainable = sum(p.numel() for p in stage.parameters() if p.requires_grad)
        stage_total_sum += total
        stage_trainable_sum += trainable
        print(f"{name:<15}{total:>15,}{trainable:>15,}")

    print("-" * 45)
    print(f"{'Model total':<15}{model_total:>15,}{model_trainable:>15,}")
    print("-" * 45 + "\n")
    # Safety check
    assert stage_total_sum == model_total, f"Stage totals ({stage_total_sum}) do not match model total ({model_total})!"
    assert stage_trainable_sum == model_trainable, f"Stage trainables ({stage_trainable_sum}) do not match model trainables ({model_trainable})!"

    return model_total, model_trainable

def get_param_groups(model, model_config, peak_lr=1e-3, target_model_groups=()):
    param_groups = {}
    param_groups['main'] = [{
            "params": list(model.stages[name].parameters()),
            "lr": specs['optim_params'].get('lr', model_config['defaults']['lr'])*peak_lr,
            "weight_decay": specs['optim_params'].get('weight_decay', model_config['defaults']['weight_decay']),
            "name": name
        }
        for name, specs in model_config['group_specs'].items()
    ]
    if len(target_model_groups) > 0:
        param_groups['target_model'] = [{
                "params": list(model.target_model.stages[name].parameters()),
                "lr": model_config['group_specs'][name]['optim_params'].get('lr', model_config['defaults']['lr'])*peak_lr,
                "weight_decay": model_config['group_specs'][name]['optim_params'].get('weight_decay', model_config['defaults']['weight_decay']),
                "name": name
            }
            for name in target_model_groups
        ]
    return param_groups

def freeze_param_groups(param_groups, frozen_groups={'main': ()}):
    trainable_groups = {}
    for k, groups in param_groups.items():
        trainable_groups[k] = [g for g in groups if g['name'] not in frozen_groups.get(k, ())]
        for i, g in enumerate(groups):
            if g['name'] in frozen_groups.get(k, ()):
                g['lr'] = 0.0
                for p in g['params']:
                    p.requires_grad = False
    return param_groups, trainable_groups

def unwrap(m):
    return m.module if hasattr(m, "module") else m