import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# -------------------------------------------------------------
# 1. ASYMMETRIC LOSS (ASL) CHO MULTI-LABEL CLASSIFICATION
# -------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        x: input logits (batch_size, num_classes)
        y: targets (batch_size, num_classes) with 0 or 1
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

# -------------------------------------------------------------
# 2. DATASET CHEXPERT
# -------------------------------------------------------------
TASKS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other', 'Fracture', 'Support Devices'
]

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, sample_size=50000, uncertainty_policy='ones'):
        self.df = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        
        # Lấy subset 50,000 mẫu nếu cần (cho train)
        if sample_size and len(self.df) > sample_size:
            print(f"Sampling {sample_size} records from total {len(self.df)}...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
        self.labels = self.df[TASKS].copy()
        
        # Xử lý nhãn không chắc chắn (Uncertainty Handling)
        if uncertainty_policy == 'ones':
            # U-Ones: Coi nhãn -1 (uncertain) là 1 (positive)
            self.labels = self.labels.replace(-1.0, 1.0)
        elif uncertainty_policy == 'zeros':
            # U-Zeros: Coi nhãn -1 là 0 (negative)
            self.labels = self.labels.replace(-1.0, 0.0)
            
        # NaN coi là 0.0
        self.labels = self.labels.fillna(0.0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Đường dẫn dạng: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        # Trong ổ đĩa: E:/doan/archive/train/patient00001/study1/view1_frontal.jpg
        rel_path = self.df.iloc[idx]['Path'].replace('CheXpert-v1.0-small/', '')
        img_path = os.path.join(self.img_root, rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback nếu ảnh lỗi
            image = Image.new('RGB', (224, 224), color=0)
            
        if self.transform:
            image = self.transform(image)
            
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, target

# -------------------------------------------------------------
# 3. MÔ HÌNH DENSENET-121
# -------------------------------------------------------------
def get_densenet121(num_classes=14, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

# -------------------------------------------------------------
# 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ (TRAIN & EVALUATION)
# -------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.detach().cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    # Tính AUROC cho từng bệnh lý và Macro AUROC
    aurocs = []
    for i in range(len(TASKS)):
        try:
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aurocs.append(auc)
        except Exception:
            pass
            
    macro_auroc = np.mean(aurocs) if aurocs else 0.0
    
    # Tính F1-score với threshold 0.5
    bin_preds = (all_preds >= 0.5).astype(float)
    f1 = f1_score(all_targets, bin_preds, average='macro', zero_division=0)
    
    return val_loss, macro_auroc, f1

# -------------------------------------------------------------
# 5. CHƯƠNG TRÌNH CHÍNH (MAIN FUNCTION)
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DenseNet-121 with Asymmetric Loss on CheXpert")
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32 for RTX 4060 8GB)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--sample_size', type=int, default=50000, help='Number of train images (default: 50,000)')
    parser.add_argument('--output_dir', type=str, default='E:/doan/training_checkpoints', help='Checkpoint output dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Image Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    train_csv = 'E:/doan/archive/train.csv'
    valid_csv = 'E:/doan/archive/valid.csv'
    img_root = 'E:/doan/archive'

    print("Loading datasets...")
    train_dataset = CheXpertDataset(train_csv, img_root, transform=train_transform, sample_size=args.sample_size)
    valid_dataset = CheXpertDataset(valid_csv, img_root, transform=valid_transform, sample_size=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model, Loss, Optimizer, Scheduler
    model = get_densenet121(num_classes=14, pretrained=True).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # History Log File
    log_file = os.path.join(args.output_dir, 'training_log.csv')
    log_data = []

    best_auroc = 0.0
    print("\n" + "="*60)
    print(f"START TRAINING DENSENET-121 + ASL ({args.epochs} EPOCHS, {len(train_dataset)} SAMPLES)")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auroc, val_f1 = evaluate(model, valid_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")

        # Lưu lại log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auroc': val_auroc,
            'val_f1': val_f1,
            'time_sec': epoch_time,
            'lr': scheduler.get_last_lr()[0]
        }
        log_data.append(log_entry)
        pd.DataFrame(log_data).to_csv(log_file, index=False)

        # Lưu best checkpoint
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_model_path = os.path.join(args.output_dir, 'best_densenet121_asl.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_f1': val_f1
            }, best_model_path)
            print(f"  >>> Saved NEW BEST Checkpoint (AUROC: {best_auroc:.4f}) to {best_model_path}")

        # Định kỳ lưu mỗi 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))

    print("\n" + "="*60)
    print(f"TRAINING COMPLETED! Best Val AUROC: {best_auroc:.4f}")
    print(f"Logs and checkpoints saved in: {args.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()

