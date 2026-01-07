import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


class NNTrainer:
    def __init__(self, model, criterion, optimizer, device, exp_path, scheduler=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.exp_path = exp_path
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

    def _run_epoch(self, loader, is_train=True):
        if is_train: self.model.train()
        else: self.model.eval()

        total_loss, correct, total = 0, 0, 0
        with torch.set_grad_enabled(is_train):
            for batch in loader:
                ids, labels = batch['ids'].to(self.device), batch['label'].to(self.device)
                if is_train: self.optimizer.zero_grad()
                
                outputs = self.model(ids)
                loss = self.criterion(outputs, labels)
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    def fit(self, train_loader, val_loader, epochs=20, patience=5):
        """
        Training with Checkpointing and Early Stopping.
        patience: epochs to wait without improvement before stopping.
        """
        print(f"Starting training on {self.device}...")
        
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path, exist_ok=True)

        for epoch in range(epochs):
            t_loss, t_acc = self._run_epoch(train_loader, is_train=True)
            v_loss, v_acc = self._run_epoch(val_loader, is_train=False)
            
            self.history['train_loss'].append(t_loss)
            self.history['train_acc'].append(t_acc)
            self.history['val_loss'].append(v_loss)
            self.history['val_acc'].append(v_acc)

            
            if v_acc > self.best_val_acc:
                self.best_val_acc = v_acc
                self.early_stop_counter = 0 
                
                torch.save(self.model.state_dict(), os.path.join(self.exp_path, 'best_model.pth'))
                checkpoint_msg = " -> Best model saved"
            else:
                self.early_stop_counter += 1
                checkpoint_msg = ""

            if self.scheduler:
                self.scheduler.step(v_acc)
            
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {t_loss:.3f} | Val Acc: {v_acc:.3f}{checkpoint_msg}")

            
            if self.early_stop_counter >= patience:
                print(f"\n[Early Stopping] The model has not improved in {patience} epochs. Stopping...")
                
                self.model.load_state_dict(torch.load(os.path.join(self.exp_path, 'best_model.pth')))
                break

    def evaluate(self, loader):
        self.model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for batch in loader:
                ids, labels = batch['ids'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(ids)
                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        acc = np.mean(np.array(all_labels) == np.array(all_preds))
        return all_labels, all_preds, acc

    def save_results(self, y_true, y_pred, vocab):
        report = classification_report(y_true, y_pred)
        with open(os.path.join(self.exp_path, 'metrics.txt'), 'w') as f:
            f.write(report)

        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.exp_path, 'confusion_matrix.png'))
        plt.close()

        with open(os.path.join(self.exp_path, 'vocab.json'), 'w') as f:
            json.dump(vocab, f)
        
        print(f"Results saved in {self.exp_path}")

    def plot_learning_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Val')
        plt.title('Loss Evolution'); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train')
        plt.plot(epochs, self.history['val_acc'], label='Val')
        plt.title('Accuracy Evolution'); plt.legend()

        plt.savefig(os.path.join(self.exp_path, 'learning_curves.png'))
        plt.close()