"""Automated functional test — exercises all code paths without GUI."""
import torch
import os
import csv
import tempfile
from parser_utils import parse_program, validate_dsl, DSL_PRESETS, create_modern_nn
from trainer import TrainingEngine
from network import ModernMLP

print("=" * 60)
print("NeuroDSL Infinity v4.0 — Full Functional Test")
print("=" * 60)
errors = []

# --------------------------------------------------
# TEST 1: All DSL Presets build and forward pass
# --------------------------------------------------
print("\n[TEST 1] Building and running all 8 DSL presets...")
for name, dsl in DSL_PRESETS.items():
    try:
        result = parse_program(dsl)
        assert result, f"Preset '{name}' parse failed"
        model = create_modern_nn(result)
        
        in_dim = result[0].get('in', result[0].get('dim'))
        x = torch.randn(4, in_dim)
        with torch.no_grad():
            y = model(x)
        print(f"  ✅ {name}: input={in_dim} → output={tuple(y.shape)}")
    except Exception as e:
        errors.append(f"Preset '{name}': {e}")
        print(f"  ❌ {name}: {e}")

# --------------------------------------------------
# TEST 2: DSL Validation catches dim mismatches
# --------------------------------------------------
print("\n[TEST 2] DSL Validation...")
try:
    issues, result = validate_dsl("[128, 64], [32, 10]")
    assert result is not None
    assert any("match" in msg.lower() for _, msg in issues), "Should detect dim mismatch"
    print(f"  ✅ Detects dimension mismatch ({len(issues)} warnings)")
except Exception as e:
    errors.append(f"Validation: {e}")
    print(f"  ❌ {e}")

try:
    issues, result = validate_dsl("invalid garbage text")
    assert result is None
    print(f"  ✅ Rejects invalid DSL")
except Exception as e:
    errors.append(f"Invalid DSL rejection: {e}")
    print(f"  ❌ {e}")

# --------------------------------------------------
# TEST 3: Training engine with all loss functions
# --------------------------------------------------
print("\n[TEST 3] Training engine with all loss functions...")
for loss_fn in ['MSE', 'Huber', 'MAE']:
    try:
        defs = [{'type': 'linear', 'in': 8, 'out': 16}, {'type': 'linear', 'in': 16, 'out': 4}]
        model = ModernMLP(defs)
        trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=1.0, warmup_steps=3)
        X, y = trainer.generate_dummy_data(8, 4, n_samples=50)
        
        losses = []
        for i in range(20):
            loss, lr, gn = trainer.train_step(X, y)
            losses.append(loss)
        
        print(f"  ✅ {loss_fn}: loss {losses[0]:.4f} → {losses[-1]:.4f} | LR: {lr:.6f} | ∇: {gn:.3f}")
    except Exception as e:
        errors.append(f"Training {loss_fn}: {e}")
        print(f"  ❌ {loss_fn}: {e}")

# --------------------------------------------------
# TEST 4: CrossEntropy with integer targets
# --------------------------------------------------
print("\n[TEST 4] CrossEntropy loss...")
try:
    defs = [{'type': 'linear', 'in': 8, 'out': 10}]
    model = ModernMLP(defs)
    trainer = TrainingEngine(model, loss_fn='CrossEntropy', grad_clip=1.0, warmup_steps=2)
    X = torch.randn(50, 8)
    y = torch.randint(0, 10, (50, 1)).float()
    
    for i in range(10):
        loss, lr, gn = trainer.train_step(X, y)
    print(f"  ✅ CrossEntropy: loss={loss:.4f}")
except Exception as e:
    errors.append(f"CrossEntropy: {e}")
    print(f"  ❌ {e}")

# --------------------------------------------------
# TEST 5: CSV data loading
# --------------------------------------------------
print("\n[TEST 5] CSV data loading...")
try:
    # Create a temp CSV
    csv_path = os.path.join(tempfile.gettempdir(), "test_neurodsl.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "f3", "target"])
        for _ in range(20):
            import random
            writer.writerow([random.random(), random.random(), random.random(), random.randint(0, 1)])
    
    defs = [{'type': 'linear', 'in': 3, 'out': 1}]
    model = ModernMLP(defs)
    trainer = TrainingEngine(model, loss_fn='MSE')
    X, y = trainer.load_csv_data(csv_path)
    assert X.shape == (20, 3), f"Expected (20, 3), got {X.shape}"
    assert y.shape == (20, 1), f"Expected (20, 1), got {y.shape}"
    
    loss, lr, gn = trainer.train_step(X, y)
    print(f"  ✅ CSV loaded: {X.shape[0]} samples, {X.shape[1]} features → loss={loss:.4f}")
    os.remove(csv_path)
except Exception as e:
    errors.append(f"CSV loading: {e}")
    print(f"  ❌ {e}")

# --------------------------------------------------
# TEST 6: TorchScript export
# --------------------------------------------------
print("\n[TEST 6] TorchScript export...")
try:
    defs = [{'type': 'linear', 'in': 4, 'out': 2}]
    model = ModernMLP(defs)
    trainer = TrainingEngine(model)
    ts_path = os.path.join(tempfile.gettempdir(), "test_model.pt")
    trainer.export_torchscript(ts_path, 4)
    assert os.path.exists(ts_path)
    
    # Verify it loads and runs
    loaded = torch.jit.load(ts_path)
    out = loaded(torch.randn(1, 4))
    print(f"  ✅ TorchScript exported and loaded: output shape={tuple(out.shape)}")
    os.remove(ts_path)
except Exception as e:
    errors.append(f"TorchScript: {e}")
    print(f"  ❌ {e}")

# --------------------------------------------------
# TEST 7: Model summary generation
# --------------------------------------------------
print("\n[TEST 7] Model summary...")
try:
    defs = parse_program("fractal: [128, 2], moe: [128, 4], residual: [128], dropout: [0.2], [128, 10]")
    model = ModernMLP(defs)
    summary, total = model.get_summary()
    print(f"  ✅ Summary: {len(summary)} layer entries, {total:,} total params")
    for entry in summary:
        print(f"     [{entry['index']:2}] {entry['type']:<20} params={entry['params']:>8,}")
except Exception as e:
    errors.append(f"Summary: {e}")
    print(f"  ❌ {e}")

# --------------------------------------------------
# TEST 8: Complex architecture end-to-end
# --------------------------------------------------
print("\n[TEST 8] Kitchen Sink architecture end-to-end...")
try:
    dsl = DSL_PRESETS["Kitchen Sink"]
    defs = parse_program(dsl)
    model = create_modern_nn(defs)
    
    in_dim = defs[0].get('in', defs[0].get('dim'))
    trainer = TrainingEngine(model, loss_fn='MSE', grad_clip=0.5, warmup_steps=5)
    X, y = trainer.generate_dummy_data(in_dim, 10, n_samples=32)
    
    for i in range(15):
        loss, lr, gn = trainer.train_step(X, y)
    
    # Inference
    model.eval()
    with torch.no_grad():
        pred = model(X[:1])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Kitchen Sink: {total_params:,} params | Final loss={loss:.4f} | Output shape={tuple(pred.shape)}")
except Exception as e:
    errors.append(f"Kitchen Sink: {e}")
    print(f"  ❌ {e}")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"❌ FAILED: {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
else:
    print("✅ ALL 8 TESTS PASSED — No bugs found!")
print("=" * 60)
