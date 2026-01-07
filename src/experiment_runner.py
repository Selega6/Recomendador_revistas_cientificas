import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)


class ExperimentRunner:
    def __init__(self, base_path='results/experiments'):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def run(self, name, pipeline, X_train, X_test, y_train, y_test):
        """Runs a complete experiment and saves the results to disk."""
        print(f"\n>>> Starting Experiment: {name}")
        exp_path = os.path.join(self.base_path, name)
        os.makedirs(exp_path, exist_ok=True)

        
        pipeline.fit(X_train, y_train)

        model_filename = os.path.join(exp_path, f'{name}_model.joblib')
        joblib.dump(pipeline, model_filename)
        print(f"Model saved in: {model_filename}")
        
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        
        with open(os.path.join(exp_path, 'metrics.txt'), 'w') as f:
            f.write(f"Experiment: {name}\n")
            f.write("-" * 30 + "\n")
            f.write(report)

        
        self._save_confusion_matrix(y_test, y_pred, exp_path)

        
        test_summary = pd.DataFrame({'true': y_test, 'pred': y_pred})
        test_summary.to_csv(os.path.join(exp_path, 'predictions.csv'), index=False)

        print(f"Results saved in: {exp_path}")
        return report

    def _save_confusion_matrix(self, y_true, y_pred, path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Realidad')
        plt.xlabel('Predicción')
        plt.savefig(os.path.join(path, 'confusion_matrix.png'))
        plt.close()

    def _plot_roc_curve(self, y_test, y_score, classes, path):
        
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = len(classes)
        plt.figure(figsize=(10, 8))
        
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC {classes[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(path, 'roc_curve.png'))
        plt.close()