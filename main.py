import FreeSimpleGUI as sg
import torch
import threading
import time
import os
import csv
from parser_utils import parse_program, create_modern_nn, validate_dsl, DSL_PRESETS
from trainer import TrainingEngine

# --- Threading Wrappers ---

stop_training_flag = False

def build_thread(program, window):
    try:
        window.write_event_value("-STATUS-UPDATE-", "v4.0 Trace: Validating DSL Specification...")
        issues, layer_defs = validate_dsl(program)
        
        # Report warnings
        for severity, msg in issues:
            window.write_event_value("-STATUS-UPDATE-", f"[{severity}] {msg}")
        
        if layer_defs is None:
            raise Exception("DSL Syntax Error: Check brackets, commas, and layer keywords.")
            
        window.write_event_value("-STATUS-UPDATE-", "v4.0 Trace: Constructing Neural Architecture...")
        model = create_modern_nn(layer_defs)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        window.write_event_value("-BUILD-DONE-", (model, layer_defs, total_params, trainable))
    except Exception as e:
        import traceback
        err = f"Fault in Neural Construction:\n{e}\n{traceback.format_exc()}"
        window.write_event_value("-THREAD-ERROR-", err)

def train_thread(trainer, epochs, in_dim, out_dim, window):
    global stop_training_flag
    stop_training_flag = False
    try:
        X, y = trainer.generate_dummy_data(in_dim, out_dim)
        trainer.update_epochs(epochs)
        for epoch in range(epochs):
            if stop_training_flag:
                window.write_event_value("-STATUS-UPDATE-", "Training Manually Aborted.")
                break
            loss, lr, grad_norm = trainer.train_step(X, y)
            window.write_event_value("-TRAIN-PROGRESS-", (epoch, loss, lr, grad_norm))
            time.sleep(0.005)
        window.write_event_value("-TRAIN-DONE-", None)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Training Stability Fault: {e}")

def train_csv_thread(trainer, epochs, csv_path, window):
    global stop_training_flag
    stop_training_flag = False
    try:
        X, y = trainer.load_csv_data(csv_path)
        trainer.update_epochs(epochs)
        window.write_event_value("-STATUS-UPDATE-", f"Loaded CSV: {X.shape[0]} samples, {X.shape[1]} features")
        for epoch in range(epochs):
            if stop_training_flag:
                window.write_event_value("-STATUS-UPDATE-", "Training Manually Aborted.")
                break
            loss, lr, grad_norm = trainer.train_step(X, y)
            window.write_event_value("-TRAIN-PROGRESS-", (epoch, loss, lr, grad_norm))
            time.sleep(0.005)
        window.write_event_value("-TRAIN-DONE-", None)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"CSV Training Fault: {e}")

def inference_thread(model, input_vals, window):
    try:
        input_tensor = torch.tensor([input_vals], dtype=torch.float32)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.to(device))
        window.write_event_value("-INF-DONE-", (input_vals, output.cpu().numpy().flatten().tolist()))
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Inference Error: {e}")

def batch_inference_thread(model, csv_path, window):
    try:
        data = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if vals:
                        data.append(vals)
                except ValueError:
                    continue
        
        if not data:
            window.write_event_value("-THREAD-ERROR-", "No valid numeric data in CSV.")
            return
        
        X = torch.tensor(data, dtype=torch.float32)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            output = model(X.to(device))
        
        results = output.cpu().numpy().tolist()
        window.write_event_value("-BATCH-INF-DONE-", (len(data), results[:20], len(results)))
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Batch Inference Error: {e}")


# --- GUI Components ---

def create_gui():
    sg.theme("DarkGrey8") 
    
    TEXT_MAIN = "#E0E0E0"
    TEXT_ACCENT = "#00D4FF"
    TEXT_GREEN = "#00FF9D"
    TEXT_RED = "#FF3D71"
    TEXT_GOLD = "#FFD700"
    BG_SIDEBAR = "#1A1C1E"
    BG_WORKSPACE = "#0D0E10"
    BG_INPUT = "#1F2226"
    
    font_main = ("Segoe UI", 11)
    font_bold = ("Segoe UI", 11, "bold")
    font_mono = ("Cascadia Code", 10)
    font_title = ("Segoe UI", 22, "bold")
    font_small = ("Segoe UI", 9)
    
    preset_names = list(DSL_PRESETS.keys())
    loss_names = list(TrainingEngine.LOSS_FUNCTIONS.keys())

    # Sidebar: Model Properties & Persistence
    sidebar = [
        [sg.Text("ARCHITECTURE INFO", font=font_bold, text_color=TEXT_ACCENT, pad=(0, 10))],
        [sg.Frame("", [
            [sg.Text("Nodes:", font=font_main), sg.Text("0", key="-STAT-NODES-", text_color=TEXT_MAIN, font=font_bold)],
            [sg.Text("Params:", font=font_main), sg.Text("0", key="-STAT-PARAMS-", text_color=TEXT_MAIN, font=font_bold)],
            [sg.Text("Trainable:", font=font_main), sg.Text("0", key="-STAT-TRAINABLE-", text_color=TEXT_GREEN, font=font_bold)],
            [sg.Text("Backend:", font=font_main), sg.Text("CUDA" if torch.cuda.is_available() else "CPU", text_color=TEXT_GREEN, font=font_bold)],
        ], background_color=BG_SIDEBAR, border_width=0, pad=(0, 10))],
        [sg.HorizontalSeparator(color="#333333")],
        [sg.Text("TRAINING CONFIG", font=font_bold, text_color=TEXT_ACCENT, pad=(0, 8))],
        [sg.Text("Loss:", font=font_small), sg.Combo(loss_names, default_value="MSE", key="-LOSS-FN-", size=(14, 1), font=font_small, readonly=True)],
        [sg.Checkbox("Grad Clip", key="-GRAD-CLIP-", default=True, font=font_small, text_color=TEXT_MAIN),
         sg.Text("Max:", font=font_small), sg.InputText("1.0", key="-CLIP-VAL-", size=(5, 1), font=font_small, background_color=BG_INPUT, text_color="#FFFFFF")],
        [sg.Text("Warmup:", font=font_small), sg.Slider(range=(0, 50), default_value=10, orientation='h', size=(10, 15), key="-WARMUP-", text_color=TEXT_MAIN)],
        [sg.HorizontalSeparator(color="#333333")],
        [sg.Text("WEIGHT CONTROL", font=font_bold, text_color=TEXT_ACCENT, pad=(0, 8))],
        [sg.Button("💾 SAVE PTH", key="-SAVE-", disabled=True, size=(16, 1), font=font_bold, button_color=("#FFFFFF", "#2B6CB0"))],
        [sg.Button("📂 LOAD PTH", key="-LOAD-", disabled=True, size=(16, 1), font=font_bold, button_color=("#FFFFFF", "#553C9A"))],
        [sg.Button("🚀 EXPORT ONNX", key="-EXPORT-", disabled=True, size=(16, 1), font=font_bold, button_color=("#FFFFFF", "#6B46C1"))],
        [sg.Button("⚡ TORCHSCRIPT", key="-EXPORT-TS-", disabled=True, size=(16, 1), font=font_bold, button_color=("#FFFFFF", "#4A5568"))],
        [sg.VPush()],
        [sg.Text("SYSTEM HEARTBEAT", font=("Segoe UI", 9), text_color="#444444")],
        [sg.ProgressBar(100, orientation='h', size=(20, 10), key="-HEART-", bar_color=(TEXT_ACCENT, "#222222"))]
    ]

    # Workspace: DSL & Neural Stream
    workspace = [
        [sg.Text("NEURAL DSL SPECIFICATION", font=font_bold, text_color=TEXT_ACCENT),
         sg.Push(),
         sg.Text("Preset:", font=font_small, text_color=TEXT_MAIN),
         sg.Combo(preset_names, key="-PRESET-", size=(20, 1), font=font_small, readonly=True, enable_events=True)],
        [sg.InputText("fractal: [256], moe: [256, 8], [256, 1]", key="-PROGRAM-", size=(80, 1), font=font_mono, background_color=BG_INPUT, text_color="#FFFFFF", border_width=0)],
        [sg.Button("⚡ INITIALIZE CORE", key="-BUILD-", size=(25, 1), font=font_bold, button_color=("black", TEXT_ACCENT)),
         sg.Button("🔍 VALIDATE", key="-VALIDATE-", size=(12, 1), font=font_bold, button_color=("black", TEXT_GOLD))],
        [sg.HorizontalSeparator(color="#333333", pad=(0, 10))],
        [sg.TabGroup([
            [sg.Tab("  TRAINING STUDIO  ", [
                [sg.Text("Hyperparameter Tuning", font=font_bold)],
                [sg.Text("Epochs:", font=font_main), sg.Slider(range=(1, 2000), default_value=250, orientation='h', size=(25, 20), key="-EPOCHS-", text_color=TEXT_MAIN),
                 sg.Button("📁 TRAIN CSV", key="-TRAIN-CSV-", disabled=True, size=(12, 1), font=font_bold, button_color=("#FFFFFF", "#38A169"))],
                [sg.Button("▶ START", key="-TRAIN-", disabled=True, size=(12, 1), font=font_bold, button_color=("black", TEXT_GREEN)),
                 sg.Button("■ ABORT", key="-STOP-", disabled=True, size=(8, 1), font=font_bold, button_color=("#FFFFFF", TEXT_RED)),
                 sg.Text("", key="-LIVE-LOSS-", font=font_mono, text_color=TEXT_GREEN, size=(20, 1)),
                 sg.Text("", key="-LIVE-LR-", font=font_mono, text_color=TEXT_ACCENT, size=(18, 1)),
                 sg.Text("", key="-LIVE-GRAD-", font=font_mono, text_color=TEXT_GOLD, size=(18, 1))],
                [sg.Graph(canvas_size=(560, 180), graph_bottom_left=(0,0), graph_top_right=(100,1), background_color="#000000", key='-GRAPH-', pad=(0, 10))]
            ])],
            [sg.Tab("  INFERENCE LAB  ", [
                [sg.Text("Real-time Single Inference", font=font_bold)],
                [sg.Text("Input Vector (comma separated):", font=font_main)],
                [sg.InputText("0.5, 0.1, -0.2", key="-INPUT-", size=(52, 1), font=font_mono, background_color=BG_INPUT, text_color="#FFFFFF"),
                 sg.Button("🎲 RND", key="-GEN-INPUT-", size=(6, 1), font=font_bold, button_color=("#FFFFFF", "#4A5568"), tooltip="Generate random input vector")],
                [sg.Button("▶ RUN INFERENCE", key="-COMPUTE-", disabled=True, size=(20, 1), font=font_bold, button_color=("black", TEXT_ACCENT))],
                [sg.Multiline(size=(75, 5), key="-INF-OUTPUT-", font=font_mono, background_color="#000000", text_color=TEXT_ACCENT, border_width=0, no_scrollbar=True)],
                [sg.HorizontalSeparator(color="#333333")],
                [sg.Text("Batch Inference from CSV", font=font_bold)],
                [sg.InputText("", key="-BATCH-CSV-", size=(50, 1), font=font_small, background_color=BG_INPUT, text_color="#FFFFFF"),
                 sg.FileBrowse("Browse", file_types=(("CSV Files", "*.csv"),), font=font_small),
                 sg.Button("▶ BATCH RUN", key="-BATCH-INF-", disabled=True, size=(12, 1), font=font_bold, button_color=("black", "#38A169"))],
                [sg.Multiline(size=(75, 4), key="-BATCH-OUTPUT-", font=font_mono, background_color="#000000", text_color=TEXT_GREEN, border_width=0, no_scrollbar=True)]
            ])],
            [sg.Tab("  ARCHITECTURE VIZ  ", [
                [sg.Text("Model Layer Summary", font=font_bold, text_color=TEXT_ACCENT)],
                [sg.Multiline(size=(80, 14), key="-ARCH-VIZ-", font=font_mono, background_color="#000000", text_color=TEXT_MAIN, border_width=0, disabled=True)]
            ])],
            [sg.Tab("  NEURAL STREAM  ", [
                [sg.Multiline(size=(80, 14), key="-TRAIN-LOG-", font=font_mono, background_color="#000000", text_color=TEXT_GREEN, border_width=0)]
            ])]
        ], border_width=0, background_color=BG_WORKSPACE, title_color=TEXT_MAIN, selected_title_color=TEXT_ACCENT)]
    ]

    layout = [
        [sg.Column([[sg.Text("NEURODSL INFINITY v4.0", font=font_title, text_color="#FFFFFF", pad=(10, 8))]], background_color=BG_WORKSPACE, expand_x=True, element_justification='center')],
        [sg.Column(sidebar, background_color=BG_SIDEBAR, expand_y=True, pad=(0, 0), element_justification='center', vertical_alignment='top'),
         sg.Column(workspace, background_color=BG_WORKSPACE, expand_y=True, expand_x=True, pad=(10, 10))],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROG-', bar_color=(TEXT_ACCENT, '#1A1C1E'), visible=False)],
        [sg.StatusBar("System Ready. Select a preset or write DSL...", key="-STATUS-", font=font_main, background_color="#1A1C1E", text_color="#AAAAAA")]
    ]
    
    window = sg.Window("NeuroDSL Infinity Studio v4.0", layout, finalize=True, resizable=True, margins=(0,0))
    return window

def update_loss_graph(window, loss_history, max_epochs):
    graph = window['-GRAPH-']
    graph.erase()
    if not loss_history: return
    max_loss = max(loss_history) if loss_history else 1.0
    if max_loss <= 0:
        max_loss = 1.0
    graph.change_coordinates((0, 0), (max(max_epochs, 1), max_loss * 1.1))
    
    # Draw grid lines
    for frac in [0.25, 0.5, 0.75]:
        y = max_loss * frac
        try:
            graph.draw_line((0, y), (max_epochs, y), color='#1A1C1E', width=1)
        except:
            pass
    
    # Draw loss curve — use individual line segments for compatibility
    if len(loss_history) > 1:
        for i in range(len(loss_history) - 1):
            try:
                graph.draw_line(
                    (i, loss_history[i]), 
                    (i + 1, loss_history[i + 1]), 
                    color='#66FF66', width=2
                )
            except:
                pass

def update_arch_viz(window, model):
    """Generate a text-based architecture visualization."""
    summary, total = model.get_summary()
    lines = []
    lines.append(f"{'#':>3}  {'Layer Type':<22} {'Parameters':>12}  {'Trainable':>12}")
    lines.append("─" * 55)
    
    for entry in summary:
        lines.append(
            f"{entry['index']:>3}  {entry['type']:<22} {entry['params']:>12,}  {entry['trainable']:>12,}"
        )
    
    lines.append("─" * 55)
    lines.append(f"{'':>3}  {'TOTAL':<22} {total:>12,}")
    lines.append(f"\nMemory Estimate: ~{total * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    window["-ARCH-VIZ-"].update("\n".join(lines))

def main_loop(window):
    global stop_training_flag
    model = None
    trainer = None
    layer_defs = None
    loss_history = []
    pulse_val = 0
    pulse_dir = 1
    
    while True:
        event, values = window.read(timeout=100)
        
        if event == sg.WINDOW_CLOSED:
            break
            
        if event == "__TIMEOUT__":
            pulse_val += 5 * pulse_dir
            if pulse_val >= 100 or pulse_val <= 0: pulse_dir *= -1
            window["-HEART-"].update(pulse_val)

        # --- Preset Selection ---
        if event == "-PRESET-":
            preset_name = values["-PRESET-"]
            if preset_name in DSL_PRESETS:
                window["-PROGRAM-"].update(DSL_PRESETS[preset_name])
                window["-STATUS-"].update(f"Preset loaded: {preset_name}")
                window["-TRAIN-LOG-"].print(f"> Preset loaded: {preset_name}", text_color="#FFD700")

        # --- Validate DSL ---
        if event == "-VALIDATE-":
            program = values["-PROGRAM-"].strip()
            issues, result = validate_dsl(program)
            if result:
                window["-TRAIN-LOG-"].print(f"> DSL Valid: {len(result)} layers defined", text_color="#00FF9D")
                for sev, msg in issues:
                    color = "#FF3D71" if sev == "ERROR" else "#FFD700"
                    window["-TRAIN-LOG-"].print(f"  [{sev}] {msg}", text_color=color)
                window["-STATUS-"].update(f"Validated: {len(result)} layers, {len(issues)} warnings")
            else:
                window["-TRAIN-LOG-"].print("> DSL INVALID", text_color="#FF3D71")
                for sev, msg in issues:
                    window["-TRAIN-LOG-"].print(f"  [{sev}] {msg}", text_color="#FF3D71")
                window["-STATUS-"].update("Validation failed. Check Neural Stream.")

        # --- Build ---
        if event == "-BUILD-":
            program = values["-PROGRAM-"].strip()
            window["-STATUS-"].update("Initializing v4.0 Build Engine...")
            window["-BUILD-"].update("⏳ PROCESSING", disabled=True)
            window["-PROG-"].update(visible=True, current_count=10)
            threading.Thread(target=build_thread, args=(program, window), daemon=True).start()

        if event == "-BUILD-DONE-":
            model, layer_defs, t_params, t_trainable = values[event]
            
            # Create trainer with current UI settings
            loss_fn = values.get("-LOSS-FN-", "MSE")
            grad_clip = float(values.get("-CLIP-VAL-", "1.0")) if values.get("-GRAD-CLIP-") else 100.0
            warmup = int(values.get("-WARMUP-", 10))
            trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=grad_clip, warmup_steps=warmup)
            
            window["-STATUS-"].update("Infinity Core v4.0 Online.")
            window["-PROG-"].update(100, visible=False)
            window["-BUILD-"].update("⚡ INITIALIZE CORE", disabled=False)
            
            # Enable all action buttons
            for key in ["-TRAIN-", "-COMPUTE-", "-SAVE-", "-LOAD-", "-EXPORT-", "-EXPORT-TS-", "-TRAIN-CSV-", "-BATCH-INF-"]:
                window[key].update(disabled=False)
            
            # Update diagnostics
            window["-STAT-NODES-"].update(str(len(layer_defs)))
            window["-STAT-PARAMS-"].update(f"{t_params:,}")
            window["-STAT-TRAINABLE-"].update(f"{t_trainable:,}")
            window["-TRAIN-LOG-"].print(f"> Build Success. Params: {t_params:,} | Trainable: {t_trainable:,}", text_color="#00D4FF")
            
            # Update architecture visualization
            update_arch_viz(window, model)

        if event == "-STATUS-UPDATE-":
            window["-STATUS-"].update(values[event])
            window["-TRAIN-LOG-"].print(f"[{time.strftime('%H:%M:%S')}] {values[event]}")

        # --- Training ---
        if event == "-TRAIN-":
            epochs = int(values["-EPOCHS-"])
            in_dim = layer_defs[0].get('in', layer_defs[0].get('dim'))
            out_dim = layer_defs[-1].get('out', layer_defs[-1].get('dim'))
            loss_history = []
            
            # Update trainer config from UI
            loss_fn = values.get("-LOSS-FN-", "MSE")
            grad_clip = float(values.get("-CLIP-VAL-", "1.0")) if values.get("-GRAD-CLIP-") else 100.0
            warmup = int(values.get("-WARMUP-", 10))
            trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=grad_clip, warmup_steps=warmup)
            
            window["-TRAIN-"].update(disabled=True)
            window["-STOP-"].update(disabled=False)
            window["-STATUS-"].update(f"Training Active ({loss_fn} loss)...")
            window["-TRAIN-LOG-"].print(f"\n> Convergence Initiated (Epochs: {epochs}, Loss: {loss_fn})...", text_color="#FFD700")
            threading.Thread(target=train_thread, args=(trainer, epochs, in_dim, out_dim, window), daemon=True).start()

        if event == "-TRAIN-CSV-":
            csv_path = sg.popup_get_file("Select CSV Training Data", file_types=(("CSV Files", "*.csv"),))
            if csv_path:
                epochs = int(values["-EPOCHS-"])
                loss_history = []
                
                loss_fn = values.get("-LOSS-FN-", "MSE")
                grad_clip = float(values.get("-CLIP-VAL-", "1.0")) if values.get("-GRAD-CLIP-") else 100.0
                warmup = int(values.get("-WARMUP-", 10))
                trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=grad_clip, warmup_steps=warmup)
                
                window["-TRAIN-"].update(disabled=True)
                window["-STOP-"].update(disabled=False)
                window["-STATUS-"].update(f"Training from CSV ({loss_fn})...")
                window["-TRAIN-LOG-"].print(f"\n> CSV Training: {csv_path}", text_color="#FFD700")
                threading.Thread(target=train_csv_thread, args=(trainer, epochs, csv_path, window), daemon=True).start()

        if event == "-STOP-":
            stop_training_flag = True
            window["-STOP-"].update(disabled=True)
            window["-TRAIN-LOG-"].print("> Abort signal sent to compute engine.", text_color="#FF3D71")

        if event == "-TRAIN-PROGRESS-":
            epoch, loss, lr, grad_norm = values[event]
            loss_history.append(loss)
            
            # Live metrics display
            window["-LIVE-LOSS-"].update(f"Loss: {loss:.5f}")
            window["-LIVE-LR-"].update(f"LR: {lr:.6f}")
            window["-LIVE-GRAD-"].update(f"∇Norm: {grad_norm:.3f}")
            
            if epoch % 5 == 0:
                window["-TRAIN-LOG-"].print(f"E{epoch:4} | Loss: {loss:.5f} | LR: {lr:.6f} | ∇: {grad_norm:.3f}")
                update_loss_graph(window, loss_history, int(values["-EPOCHS-"]))

        if event == "-TRAIN-DONE-":
            window["-TRAIN-"].update(disabled=False)
            window["-STOP-"].update(disabled=True)
            window["-STATUS-"].update("Training Complete.")
            final_loss = loss_history[-1] if loss_history else 0
            window["-TRAIN-LOG-"].print(f"> Weights optimized. Final Loss: {final_loss:.6f}", text_color="#00FF9D")

        # --- Inference ---
        if event == "-GEN-INPUT-":
            if layer_defs:
                in_dim = layer_defs[0].get('in', layer_defs[0].get('dim'))
                # Generate random values between -1 and 1
                rnd_vals = [f"{torch.randn(1).item():.4f}" for _ in range(in_dim)]
                window["-INPUT-"].update(", ".join(rnd_vals))
                window["-STATUS-"].update(f"Generated {in_dim} random inputs.")
            else:
                window["-STATUS-"].update("Build model first!")

        if event == "-COMPUTE-":
            if model:
                try:
                    input_str = values["-INPUT-"]
                    input_vals = [float(x.strip()) for x in input_str.split(",") if x.strip()]
                    
                    expected_dim = layer_defs[0].get('in', layer_defs[0].get('dim'))
                    if len(input_vals) != expected_dim:
                        raise ValueError(f"Dim Mismatch: Engine expects {expected_dim}, got {len(input_vals)} inputs. Click '🎲 RND' to fix.")
                        
                    window["-STATUS-"].update("Inference Processing...")
                    window["-COMPUTE-"].update(disabled=True)
                    threading.Thread(target=inference_thread, args=(model, input_vals, window), daemon=True).start()
                except Exception as e:
                    window["-INF-OUTPUT-"].print(f"Engine Error: {e}", text_color="#FF3D71")

        if event == "-INF-DONE-":
            input_vals, output_list = values[event]
            window["-COMPUTE-"].update(disabled=False)
            window["-STATUS-"].update("Inference Ready.")
            # Format output nicely
            out_str = ", ".join([f"{v:.4f}" for v in output_list[:10]])
            suffix = f" ... (+{len(output_list) - 10} more)" if len(output_list) > 10 else ""
            window["-INF-OUTPUT-"].print(f"[Input]:  {input_vals}\n[Output]: [{out_str}{suffix}]\n---", text_color="#00D4FF")

        # --- Batch Inference ---
        if event == "-BATCH-INF-":
            if model:
                csv_path = values.get("-BATCH-CSV-", "")
                if csv_path and os.path.exists(csv_path):
                    window["-BATCH-INF-"].update(disabled=True)
                    window["-STATUS-"].update("Batch Inference Running...")
                    threading.Thread(target=batch_inference_thread, args=(model, csv_path, window), daemon=True).start()
                else:
                    window["-BATCH-OUTPUT-"].print("Error: Select a valid CSV file.", text_color="#FF3D71")

        if event == "-BATCH-INF-DONE-":
            n_samples, preview, total = values[event]
            window["-BATCH-INF-"].update(disabled=False)
            window["-STATUS-"].update(f"Batch Inference Done: {total} samples.")
            window["-BATCH-OUTPUT-"].update("")
            window["-BATCH-OUTPUT-"].print(f"Processed {total} samples. Showing first {min(5, len(preview))}:")
            for i, row in enumerate(preview[:5]):
                out_str = ", ".join([f"{v:.4f}" for v in row[:8]])
                window["-BATCH-OUTPUT-"].print(f"  [{i}]: [{out_str}]", text_color="#00FF9D")

        # --- Save/Load/Export ---
        if event == "-SAVE-":
            if model:
                path = sg.popup_get_file("Save Weights", save_as=True, default_extension=".pth", file_types=(("PyTorch Weights", "*.pth"),))
                if path:
                    try:
                        state = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
                        torch.save(state, path)
                        sg.popup(f"Weights saved to:\n{path}")
                    except Exception as e: sg.popup_error(f"Save Error: {e}")

        if event == "-LOAD-":
            if model:
                path = sg.popup_get_file("Load Weights", file_types=(("PyTorch Weights", "*.pth"),))
                if path:
                    try:
                        state = torch.load(path, map_location=next(model.parameters()).device)
                        if hasattr(model, '_orig_mod'): model._orig_mod.load_state_dict(state)
                        else: model.load_state_dict(state)
                        sg.popup("Weights restored successfully.")
                    except Exception as e: sg.popup_error(f"Load Error: {e}")

        if event == "-EXPORT-":
            if trainer:
                in_dim = layer_defs[0].get('in', layer_defs[0].get('dim'))
                path = sg.popup_get_file("Export ONNX Model", save_as=True, default_extension=".onnx", file_types=(("ONNX Model", "*.onnx"),))
                if path:
                    try:
                        trainer.export_onnx(path, in_dim)
                        sg.popup(f"ONNX Model exported to:\n{path}")
                    except Exception as e: sg.popup_error(f"Export Error:\n{e}")

        if event == "-EXPORT-TS-":
            if trainer:
                in_dim = layer_defs[0].get('in', layer_defs[0].get('dim'))
                path = sg.popup_get_file("Export TorchScript Model", save_as=True, default_extension=".pt", file_types=(("TorchScript", "*.pt"),))
                if path:
                    try:
                        trainer.export_torchscript(path, in_dim)
                        sg.popup(f"TorchScript Model exported to:\n{path}")
                    except Exception as e: sg.popup_error(f"TorchScript Export Error:\n{e}")

        # --- Error Handling ---
        if event == "-THREAD-ERROR-":
            err_msg = values[event]
            window["-TRAIN-LOG-"].print(f"\n[FATAL ENGINE ERROR]\n{err_msg}", text_color="#FF3D71")
            sg.popup_scrolled(f"System Fault Detected:\n\n{err_msg}", 
                             title="Neural Engine Error", 
                             background_color="#1A1C1E", 
                             text_color="#FF3D71",
                             size=(80, 20))
            window["-STATUS-"].update("System Fault. Check Logs.")
            window["-BUILD-"].update("⚡ INITIALIZE CORE", disabled=False)
            window["-TRAIN-"].update(disabled=False)
            window["-STOP-"].update(disabled=True)
            window["-PROG-"].update(visible=False)

    window.close()

if __name__ == "__main__":
    win = create_gui()
    try:
        main_loop(win)
    except Exception as e:
        import traceback
        sg.popup_error(f"GUI Lifecycle Crash:\n{e}\n{traceback.format_exc()}")
