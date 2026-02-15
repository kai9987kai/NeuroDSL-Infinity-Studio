````markdown
# NeuroDSL Infinity Studio (v4.0)

A **GUI-first neural architecture sandbox** that lets you describe modern PyTorch networks in a compact **DSL (Domain Specific Language)**, compile them into a runnable model, **train** (synthetic or CSV), **run inference** (single or batch), **visualize layer stats**, and **export** (PTH / ONNX / TorchScript).

> Built around:
> - **NeuroDSL** (pyparsing-based parser + validator)
> - **ModernMLP** (a composable “MLP+” stack that supports attention, MoE, fractal blocks, residuals, conv1d, BiLSTM, dropout)
> - **FreeSimpleGUI** desktop app (“Infinity Studio”)

---

## What you get

### ✅ NeuroDSL → PyTorch compiler
Write an architecture like:

```text
[64, 128], fractal: [128, 2], gqa: [128, 8, 2], moe: [128, 8], dropout: [0.1], [128, 10]


…and the studio will:

* parse it
* validate dimension flow (warn if mismatched)
* build a PyTorch model
* show param counts + “memory estimate”
* let you train + infer + export

### ✅ GUI Studio (Dark UI)

Tabs include:

* **Training Studio**: epochs slider, loss choice, gradient clipping, LR warmup, live loss curve
* **Inference Lab**: single-vector inference + random input generator
* **Batch Inference**: run a whole CSV through the model and preview outputs
* **Architecture Viz**: text table of layers + params + trainable params
* **Neural Stream**: live log feed + build/train traces

### ✅ Training Engine (v4.0)

* Losses: **MSE**, **CrossEntropy**, **Huber**, **MAE**
* Optimizer: **AdamW** + weight decay
* Scheduler: **CosineAnnealingLR**
* **Linear warmup** (first N steps)
* **Gradient clipping**
* **AMP** (automatic mixed precision) when CUDA is available
* Data:

  * **Synthetic dummy data** generator for quick experiments
  * **CSV training** loader for real numeric datasets

### ✅ Export

* **Save/Load weights** (`.pth`)
* **Export ONNX** (`.onnx`) *(optional deps)*
* **Export TorchScript** (`.pt`) *(always available)*

---

## Repo layout

```text
.
├─ main.py           # FreeSimpleGUI app (build/train/infer/export)
├─ parser_utils.py   # DSL presets + parser + validator + model factory
├─ network.py        # ModernMLP + blocks (MoE/GQA/Transformer/Fractal/etc.)
├─ trainer.py        # TrainingEngine (multi-loss, warmup, clip, AMP, exports)
├─ verify.py         # Verification tests for parsing, blocks, training, exports
├─ LICENSE           # MIT
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
└─ SECURITY.md
```

---

## Install

### 1) Clone

```bash
git clone https://github.com/kai9987kai/NeuroDSL-Infinity-Studio.git
cd NeuroDSL-Infinity-Studio
```

### 2) Create a venv

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install -U pip
python -m pip install torch numpy pyparsing FreeSimpleGUI
```

#### Optional (for ONNX export)

```bash
python -m pip install onnx onnxscript
```

> Notes:
>
> * `network.py` uses `torch.nn.functional.scaled_dot_product_attention`, so **PyTorch 2.x** is recommended.
> * FreeSimpleGUI typically uses **tkinter** by default; on Linux you may need your distro’s tkinter package.

---

## Run the Studio

```bash
python main.py
```

You should see **“NeuroDSL Infinity Studio v4.0”**.

---

## NeuroDSL reference

NeuroDSL is a **comma-separated list of layer specs**.

### Core syntax

* **Linear layer**

  * `[in, out]`
* **Keyword layer**

  * `keyword: [args...]`

### Supported layer types

#### 1) Linear

```text
[128, 64]
```

Creates: `Linear(128→64) + RMSNorm(64) + SiLU`

#### 2) Self-attention

```text
attn: [dim]
```

Creates: `SOTAAttention(dim)` (single-token attention over features)

#### 3) Grouped Query Attention (GQA)

```text
gqa: [dim, heads, groups]
```

Defaults:

* `heads = 8`
* `groups = 2`

#### 4) Mixture of Experts (MoE)

```text
moe: [dim, experts]
```

Defaults:

* `experts = 8`
  Internals:
* top-k routing (`top_k=2`)
* mixture of `SwiGLU` and lightweight `FractalBlock` experts

#### 5) Transformer block

```text
trans: [dim]
```

Creates: RMSNorm + Attention + RMSNorm + SwiGLU MLP + LayerScale skips

#### 6) Fractal block

```text
fractal: [dim, depth]
```

Defaults:

* `depth = 2`
  Internals:
* recursive-ish “extreme depth” block using repeated SwiGLU + stochastic depth + layer scale

---

## v4.0 layers

#### 7) Dropout

```text
dropout: [0.2]
```

* Accepts `0–1` as probability
* If you pass `20`, it’s treated as `20%` → `0.2`

#### 8) Residual FFN block

```text
residual: [dim, expansion]
```

Defaults:

* `expansion = 4`
  Creates: pre-norm FFN + skip + layer scale + stochastic depth

#### 9) Conv1D block

```text
conv1d: [dim, kernel]
```

Defaults:

* `kernel = 3`
  Treats the feature dim as channels, adds a length-1 “time axis”, convs, then squeezes back.

#### 10) BiLSTM block

```text
lstm: [dim, layers]
```

Defaults:

* `layers = 1`
  Wraps features into a length-1 sequence, runs a **bidirectional LSTM**, then projects back.

---

## Dimension flow rules (important)

The validator checks that dimensions chain correctly:

* If you do `[128, 64]`, the next block expecting `dim` should be `64`
* Keywords like `fractal/moe/gqa/trans/attn/residual/conv1d/lstm` are treated as “**dim-preserving**” blocks and should match the previous output.

Example (valid):

```text
[32, 64], residual: [64], [64, 10]
```

Example (warns):

```text
[128, 64], [32, 10]
```

---

## Built-in presets

The GUI includes presets like:

* **Classifier (MLP)**
* **Deep Classifier**
* **AutoEncoder**
* **Transformer Block**
* **MoE Heavy**
* **Attention Pipeline**
* **Conv-LSTM Hybrid**
* **Kitchen Sink** (everything)

Select a preset from the dropdown to populate the DSL field.

---

## Using the GUI (workflow)

1. **Pick a preset** or type your own DSL
2. Click **VALIDATE** to see warnings/errors in *Neural Stream*
3. Click **INITIALIZE CORE** to compile the model
4. Choose training settings (loss, grad clip, warmup, epochs)
5. Click **START** (synthetic) or **TRAIN CSV** (real data)
6. Use **Inference Lab** or **Batch Inference**
7. Save/export from the sidebar

---

## CSV training format

### Training CSV (`TRAIN CSV`)

* Header row is allowed (skipped)
* Non-numeric rows are skipped
* Default: **last column is target** and all previous columns are features
* Target shape is loaded as `(N, 1)`

**Regression example**

```csv
x1,x2,x3,y
0.1,0.2,0.3,0.9
...
```

Use **MSE/Huber/MAE**.

**Classification notes (CrossEntropy)**

* Use `CrossEntropy` loss in the GUI.
* Your model output dimension must equal **num_classes** (final linear out = C).
* The trainer will convert `y` into class indices if possible.

  * Best practice: store the label as an integer in the last column.

---

## Batch inference CSV format

### Batch inference (`BATCH RUN`)

* Header row allowed
* Each row should contain **only input features**
* The studio will run all rows through the model and preview outputs.

---

## Exports

### Save / Load weights (PTH)

* Saves `state_dict()` to `.pth`
* Loads back onto the model device
* Has a small compatibility shim for models wrapped by compilation (`_orig_mod`)

### Export ONNX

* Uses `torch.onnx.export(..., opset_version=11)`
* If you get an ONNX-related error, install:

  ```bash
  pip install onnx onnxscript
  ```

### Export TorchScript

* Uses tracing (`torch.jit.trace`)
* Outputs `.pt`

---

## Verification tests

Run:

```bash
python verify.py
```

It covers:

* parsing + validation warnings
* “Infinity blocks” (Fractal, GQA, MoE)
* v4.0 layers (Dropout, Residual, Conv1D, LSTM)
* training engine sanity checks + TorchScript export
* presets parse cleanly
* model summary output

---

## Implementation notes (for devs)

### Where to add a new DSL keyword

1. `parser_utils.py`

   * define a `*_layer` parser
   * add it to the `layer = (...)` union in the right precedence order
2. `network.py`

   * implement the corresponding module/block
   * handle it inside `ModernMLP.__init__`
3. `verify.py`

   * add a small shape test + parser test

### Current tensor shape convention

The model operates primarily on **2D tensors**:

* `x.shape == (batch, dim)`

Several “sequence-like” blocks (attention, transformer, LSTM) currently operate on a **single-step pseudo-sequence** internally (length = 1). If you want true sequence modeling, the next step is to extend the DSL and model pipeline to support `(batch, seq, dim)` end-to-end.

---

## Contributing

PRs are welcome. If you add a new layer:

* update parser + network + tests
* ensure `verify.py` stays green

See `CODE_OF_CONDUCT.md` for community standards.
`SECURITY.md` exists — feel free to open an issue/PR to formalize reporting details.

---

## License

MIT — see `LICENSE`.

---

## Credits

* **PyTorch** for the training/runtime stack
* **pyparsing** for building the DSL parser
* **FreeSimpleGUI** (PySimpleGUI-style API) for the desktop UI

```

::contentReference[oaicite:0]{index=0}
```
