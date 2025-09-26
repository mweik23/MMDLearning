def print_stage_param_summary(model):
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

    # Safety check
    assert stage_total_sum == model_total, f"Stage totals ({stage_total_sum}) do not match model total ({model_total})!"
    assert stage_trainable_sum == model_trainable, f"Stage trainables ({stage_trainable_sum}) do not match model trainables ({model_trainable})!"
