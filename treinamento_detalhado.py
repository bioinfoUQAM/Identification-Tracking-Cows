import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from fpdf import FPDF  # For generating PDF
import numpy as np

# ==============================
# Model
# ==============================
class CowClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        model_path = "/home/vonzin/scratch/SNA_2025_v2/efficientnet_b0_rwightman-7f5810bc.pth"
        self.model = models.efficientnet_b0(weights=None)

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }

# ==============================
# Data
# ==============================
def get_dataloaders(data_dir, batch_size=64, num_workers=4):
    def is_valid_image(path):
        valid_exts = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return path.lower().endswith(valid_exts) and '.ipynb_checkpoints' not in path

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tfms, is_valid_file=is_valid_image)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_tfms, is_valid_file=is_valid_image)

    test_path = os.path.join(data_dir, 'test')
    test_ds = None
    if os.path.exists(test_path):
        test_ds = datasets.ImageFolder(test_path, transform=val_test_tfms, is_valid_file=is_valid_image)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds else None

    return train_dl, val_dl, test_dl, len(train_ds.classes), train_ds.classes

# ==============================
# Callback for saving metrics
# ==============================
class MetricsLogger(pl.Callback):
    def __init__(self):
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.history.append({
            "epoch": trainer.current_epoch,
            "train_loss": metrics.get("train_loss", torch.tensor(0.)).item(),
            "val_loss": metrics.get("val_loss", torch.tensor(0.)).item(),
            "train_acc": metrics.get("train_acc", torch.tensor(0.)).item(),
            "val_acc": metrics.get("val_acc", torch.tensor(0.)).item()
        })

# ==============================
# Plots and analysis
# ==============================
def plot_metrics(log_dir, df_metrics):
    # Loss
    plt.figure()
    plt.plot(df_metrics['epoch'], df_metrics['train_loss'], label='Train Loss')
    plt.plot(df_metrics['epoch'], df_metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(df_metrics['epoch'], df_metrics['train_acc'], label='Train Acc')
    plt.plot(df_metrics['epoch'], df_metrics['val_acc'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Val Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'accuracy_curve.png'))
    plt.close()

def save_confusion_matrix(y_true, y_pred, classes, log_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))
    plt.close()

def generate_pdf_report(log_dir, class_report):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Final Report - Classification", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Overall Accuracy: {class_report['accuracy']*100:.2f}%", ln=True)
    pdf.ln(10)

    # Add plots
    for img in ["loss_curve.png", "accuracy_curve.png", "confusion_matrix.png"]:
        img_path = os.path.join(log_dir, img)
        if os.path.exists(img_path):
            pdf.image(img_path, w=170)
            pdf.ln(10)

    pdf.cell(0, 10, "CSV files saved in analise_model_Thomas/", ln=True)
    pdf.output(os.path.join(log_dir, "final_report.pdf"))

# ==============================
# Main
# ==============================
def main(args):
    log_dir = "analise_model_Thomas"
    os.makedirs(log_dir, exist_ok=True)

    train_dl, val_dl, test_dl, num_classes, class_names = get_dataloaders(args.data_dir, args.batch_size)
    model = CowClassifier(num_classes=num_classes, lr=args.lr)

    metrics_logger = MetricsLogger()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=16,
        accelerator="gpu",
        devices=1,
        callbacks=[
            metrics_logger,
            ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best_model"),
            EarlyStopping(monitor="val_acc", patience=5, mode="max")
        ],
        log_every_n_steps=10
    )

    # Training
    trainer.fit(model, train_dl, val_dl)

    # Save history
    df_metrics = pd.DataFrame(metrics_logger.history)
    df_metrics.to_csv(os.path.join(log_dir, "train_val_metrics.csv"), index=False)
    plot_metrics(log_dir, df_metrics)

    # Test evaluation
    if test_dl:
        print("\n=== TEST SET EVALUATION ===")
        preds, true_labels, confs = [], [], []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                pred_labels = torch.argmax(probs, dim=1)
                preds.extend(pred_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                confs.extend(torch.max(probs, dim=1)[0].cpu().numpy())

        # sklearn report
        report = classification_report(true_labels, preds, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(os.path.join(log_dir, 'classification_report.csv'))

        # Confusion matrix
        save_confusion_matrix(true_labels, preds, class_names, log_dir)

        # Save detailed predictions
        df_preds = pd.DataFrame({
            "true_label": [class_names[i] for i in true_labels],
            "pred_label": [class_names[i] for i in preds],
            "confidence": confs
        })
        df_preds.to_csv(os.path.join(log_dir, "test_predictions.csv"), index=False)

        # Generate PDF
        generate_pdf_report(log_dir, report)

        print("Full report saved in:", os.path.join(log_dir, "final_report.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="Dataset_Augmented7_split_TREINAR")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    main(args)
