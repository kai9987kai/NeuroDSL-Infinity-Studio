import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os

class TrainingEngine:
    """v4.0 Training Engine with multi-loss, gradient clipping, LR warmup, and CSV loading."""
    
    LOSS_FUNCTIONS = {
        'MSE': nn.MSELoss,
        'CrossEntropy': nn.CrossEntropyLoss,
        'Huber': nn.HuberLoss,
        'MAE': nn.L1Loss,
    }
    
    def __init__(self, model, loss_fn='MSE', max_epochs=250, grad_clip=1.0, warmup_steps=10):
        self.model = model
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.base_lr = 0.01
        
        # Loss function
        self.loss_name = loss_fn
        if loss_fn in self.LOSS_FUNCTIONS:
            self.criterion = self.LOSS_FUNCTIONS[loss_fn]()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer with AdamW
        self.optimizer = optim.AdamW(model.parameters(), lr=self.base_lr, weight_decay=0.1)
        
        # Scheduler with dynamic T_max
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, max_epochs)
        )
        
        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
    def update_epochs(self, epochs):
        """Dynamically update epoch count and recreate scheduler."""
        self.max_epochs = epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs)
        )
        self.step_count = 0
        
    def _warmup_lr(self):
        """Linear warmup for the first N steps."""
        if self.step_count < self.warmup_steps:
            warmup_factor = (self.step_count + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * warmup_factor
            return True
        return False
        
    def generate_dummy_data(self, input_dim, output_dim, n_samples=500):
        """Generate synthetic training data for quick experiments."""
        X = torch.randn(n_samples, input_dim)
        y = torch.sin(X[:, [0]]) * 0.5 + 0.5
        if output_dim > 1:
            y = torch.cat([y, torch.cos(X[:, [0]])], dim=1)
            if y.shape[1] < output_dim:
                # Fill remaining dims with linear combinations
                for j in range(y.shape[1], output_dim):
                    col_idx = j % input_dim
                    y = torch.cat([y, torch.tanh(X[:, [col_idx]])], dim=1)
                y = y[:, :output_dim]
            elif y.shape[1] > output_dim:
                y = y[:, :output_dim]
        return X, y
    
    def load_csv_data(self, csv_path, target_col=-1):
        """Load data from a CSV file. Last column is target by default.
        Returns: (X_tensor, y_tensor) or raises exception."""
        data = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                try:
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if vals:
                        data.append(vals)
                except ValueError:
                    continue  # skip non-numeric rows
        
        if not data:
            raise ValueError(f"No valid numeric data found in {csv_path}")
        
        arr = np.array(data, dtype=np.float32)
        if target_col == -1:
            X = torch.from_numpy(arr[:, :-1])
            y = torch.from_numpy(arr[:, -1:])
        else:
            X = torch.from_numpy(np.delete(arr, target_col, axis=1))
            y = torch.from_numpy(arr[:, target_col:target_col+1])
        
        return X, y

    def train_step(self, X, y):
        """Execute one training step. Returns (loss, lr, grad_norm)."""
        self.model.train()
        self.optimizer.zero_grad()
        
        device = next(self.model.parameters()).device
        X, y = X.to(device), y.to(device)
        
        # LR Warmup
        is_warmup = self._warmup_lr()
        
        # AMP Training
        use_amp = self.scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = self.model(X)
            
            # Handle CrossEntropy (needs long targets)
            if self.loss_name == 'CrossEntropy':
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1).long()
                else:
                    y = y.argmax(dim=1) if y.dim() > 1 else y.long()
            
            loss = self.criterion(outputs, y)
        
        # Backward pass
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        # Scheduler step (skip during warmup)
        if not is_warmup:
            self.scheduler.step()
        
        self.step_count += 1
        return loss.item(), self.optimizer.param_groups[0]['lr'], grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

    def export_onnx(self, path, input_dim):
        """Export model to ONNX format with graceful error handling."""
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, input_dim).to(device)
        
        try:
            torch.onnx.export(
                self.model, dummy_input, path, 
                export_params=True, 
                opset_version=11, 
                do_constant_folding=True,
                input_names=['input'], 
                output_names=['output']
            )
            return path
        except Exception as e:
            # Provide helpful fallback message
            error_msg = str(e)
            if 'onnxscript' in error_msg.lower() or 'onnx' in error_msg.lower():
                raise RuntimeError(
                    f"ONNX export failed. You may need to install onnx: "
                    f"pip install onnx onnxscript\n\nOriginal error: {e}"
                )
            raise
    
    def export_torchscript(self, path, input_dim):
        """Export model as TorchScript (always available, no extra deps)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, input_dim).to(device)
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(path)
        return path
