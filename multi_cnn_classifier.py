from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import nibabel as nib
from torchviz import make_dot
import json
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tools import plot_precision_recall_curve, plot_roc_curve
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

torch.manual_seed(42)

class BrainMRIDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.mip_paths = []
        self.labels = []
        self.transform = transform
        for _, series in data_dict.items():
            for file_path, label in series.items():
                sagittal_path = file_path[:-7] + '_resampled_sagital_mip.nii.gz'
                coronal_path = file_path[:-7] + '_resampled_coronal_mip.nii.gz'
                axial_path = file_path[:-7] + '_resampled_axial_mip.nii.gz'
                
                if os.path.exists(sagittal_path) and os.path.exists(coronal_path) and os.path.exists(axial_path):
                    sag_nif = nib.load(sagittal_path)
                    sag_data = sag_nif.get_fdata()
                    if not np.any(sag_data>0):
                        print(f"No brain mask found for {file_path}")
                        continue
                    self.mip_paths.append((sagittal_path, coronal_path, axial_path))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def set_train(self, train=True):
        self.train = train
    
    def __getitem__(self, idx):
        mip_paths = self.mip_paths[idx]
        imgs = []
        for mip_path in mip_paths:
            nifti_file = nib.load(mip_path)
            img = nifti_file.get_fdata().astype(np.float32)
            img = torch.FloatTensor(img)
            img = img.unsqueeze(0)
            img = (img - img.min()) / (img.max() - img.min())
            if self.transform and self.train:                
                img = self.transform(img)

            imgs.append(img)
        
        sagittal = imgs[0]
        coronal = imgs[1]
        axial = imgs[2]

        return sagittal, coronal, axial, self.labels[idx]



class HalvedEfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HalvedEfficientNet, self).__init__()
        efficientnet = efficientnet_b0(pretrained=True)
        # freeze the first 5 layers
        for param in efficientnet.features[:5].parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(efficientnet.features.children())[:6])
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.features(x)
        return self.dropout(x)

class MultiViewEfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(MultiViewEfficientNet, self).__init__()
        
        # Three halved EfficientNets for each view
        self.sagittal_net = HalvedEfficientNet(dropout_rate)
        self.coronal_net = HalvedEfficientNet(dropout_rate)
        self.axial_net = HalvedEfficientNet(dropout_rate)
        
        # Get the output channels of the halved EfficientNet
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output_channels = self.sagittal_net(dummy_input).shape[1]
        
        # Second half of EfficientNet
        efficientnet = efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet.features.children())[6:])
        
        # Adjust the first layer of the second half to accept concatenated input
        self.features[0][0] = nn.Conv2d(output_channels * 3, output_channels + 80, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(efficientnet.classifier[1].in_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sagittal, coronal, axial):
        # Repeat single channel input to create 3-channel input
        sagittal = sagittal.repeat(1, 3, 1, 1)
        coronal = coronal.repeat(1, 3, 1, 1)
        axial = axial.repeat(1, 3, 1, 1)
        
        # Process each view through its respective halved EfficientNet
        sagittal_features = self.sagittal_net(sagittal)
        coronal_features = self.coronal_net(coronal)
        axial_features = self.axial_net(axial)
        
        # Concatenate the features from all views
        combined_features = torch.cat((sagittal_features, coronal_features, axial_features), dim=1)
        
        # Process through the second half of EfficientNet
        features = self.features(combined_features)
        
        # Classification
        output = self.classifier(features)
        
        return output.squeeze()

def get_train_transform():
    return transforms.Compose([
        transforms.RandomAffine(degrees=40, translate=(0.5, 0.5)),
        transforms.RandomAutocontrast(),
        transforms.RandomResizedCrop(128, scale=(0.5, 1.5), ratio=(0.5, 1.5)),
        # random blur:
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  
        transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5),

    ])

def train_model(train_dataset, test_dataset, num_epochs, batch_size, learning_rate, weight_decay, device, res_dir):
    
    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store metrics for each fold
    val_precisions, val_recalls, val_aps = [], [], []
    val_fprs, val_tprs, val_aucs = [], [], []
    test_precisions, test_recalls, test_aps = [], [], []
    test_fprs, test_tprs, test_aucs = [], [], []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        
        # Split datasets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = MultiViewEfficientNet().to(device)
        criterion = BCELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        
        for epoch in range(num_epochs):
            model.train()
            train_dataset.set_train(train=True)
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for sagittal, coronal, axial, labels in train_loader:
                sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
                
                optimizer.zero_grad()
                outputs = model(sagittal, coronal, axial)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += ((outputs > 0.5) == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                train_dataset.set_train(train=False)
                model.eval()
                for sagittal, coronal, axial, labels in val_loader:
                    sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)

                    outputs = model(sagittal, coronal, axial)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_correct += ((outputs > 0.5) == labels).sum().item()
                    val_total += labels.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # calculate test loss and acc
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for sagittal, coronal, axial, labels in test_loader:
                    sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
                    outputs = model(sagittal, coronal, axial)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_correct += ((outputs > 0.5) == labels).sum().item()
                    test_total += labels.size(0)
            test_loss /= len(test_loader)
            test_acc = test_correct / test_total
              
            print(f"FOLD {fold}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Calculate final metrics for validation set
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for sagittal, coronal, axial, labels in val_loader:
                sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
                outputs = model(sagittal, coronal, axial)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        precision, recall, _ = precision_recall_curve(val_labels, val_preds)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_aps.append(auc(recall, precision))
        
        fpr, tpr, _ = roc_curve(val_labels, val_preds)
        val_fprs.append(fpr)
        val_tprs.append(tpr)
        val_aucs.append(auc(fpr, tpr))
        
        # Calculate final metrics for test set
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for sagittal, coronal, axial, labels in test_loader:
                sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
                outputs = model(sagittal, coronal, axial)
                test_preds.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        precision, recall, _ = precision_recall_curve(test_labels, test_preds)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_aps.append(auc(recall, precision))
        
        fpr, tpr, _ = roc_curve(test_labels, test_preds)
        test_fprs.append(fpr)
        test_tprs.append(tpr)
        test_aucs.append(auc(fpr, tpr))
        
    # Plot precision-recall and ROC curves
    plot_precision_recall_curve(val_precisions, val_recalls, val_aps, save_file_name=os.path.join(res_dir, 'validation_pr_curve.jpg'))
    plot_roc_curve(val_fprs, val_tprs, val_aucs, save_file_name=os.path.join(res_dir, 'validation_roc_curve.jpg'))
    plot_precision_recall_curve(test_precisions, test_recalls, test_aps, save_file_name=os.path.join(res_dir, 'test_pr_curve.jpg'))
    plot_roc_curve(test_fprs, test_tprs, test_aucs, save_file_name=os.path.join(res_dir, 'test_roc_curve.jpg'))

    # Train final model without cross validation
    print("Training final model...")
    model = MultiViewEfficientNet().to(device)
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_dataset.set_train(train=True)
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sagittal, coronal, axial, labels in train_loader:
            sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(sagittal, coronal, axial)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if loss < best_val_loss:
                best_val_loss = loss
                best_model_state = model.state_dict()
            
            train_loss += loss.item()
            train_correct += ((outputs > 0.5) == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # Save the final model
    torch.save(best_model_state, os.path.join(res_dir, 'final_model.pth'))
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for sagittal, coronal, axial, labels in test_loader:
            sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
            outputs = model(sagittal, coronal, axial)
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate and plot final metrics for test set
    precision, recall, _ = precision_recall_curve(test_labels, test_preds)
    test_ap = auc(recall, precision)
    plot_precision_recall_curve([precision], [recall], [test_ap], save_file_name=os.path.join(res_dir, 'final_test_pr_curve.jpg'))
    
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    test_auc = auc(fpr, tpr)
    plot_roc_curve([fpr], [tpr], [test_auc], save_file_name=os.path.join(res_dir, 'final_test_roc_curve.jpg'))
    
    print(f"Final Test AP: {test_ap:.4f}, Final Test AUC: {test_auc:.4f}")

def main():
    # Create results directory if it doesn't exist
    res_dir = 'multi_efficientnet_results'
    os.makedirs(res_dir, exist_ok=True)

    # Load data
    with open('alon_labels_filtered.json', 'r') as f:
        alon_labels = json.load(f)
    with open('oo_test_labels_filtered.json', 'r') as f:
        oo_test_labels = json.load(f)

    # Set up device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Training parameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 1e-4

    train_dataset = BrainMRIDataset(alon_labels, transform=None)
    test_dataset = BrainMRIDataset(oo_test_labels, transform=None)

    train_model(train_dataset, test_dataset, num_epochs, batch_size, learning_rate, weight_decay, device, res_dir)


if __name__ == '__main__':
    main()