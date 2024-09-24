import json
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import nibabel as nib
import matplotlib.pyplot as plt
from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b0, densenet121, inception_v3, vgg16, mobilenet_v2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, EfficientNet_B0_Weights, DenseNet121_Weights, Inception_V3_Weights, VGG16_Weights, MobileNet_V2_Weights
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from tools import load_nifti_data

# random seed
torch.manual_seed(42)

class BrainMRIDataset(Dataset):
    def __init__(self, data_dict, transform=None, is_train=False):
        self.sagittal_paths = []
        self.coronal_paths = []
        self.axial_paths = []
        self.labels = []
        self.transform = transform
        self.is_train = is_train 
        for _, series in data_dict.items():
            for file_path, label in series.items():
                sagittal_path = file_path[:-7] + '_resampled_sagital_mip.nii.gz'
                coronal_path = file_path[:-7] + '_resampled_coronal_mip.nii.gz'
                axial_path = file_path[:-7] + '_resampled_axial_mip.nii.gz'
                if os.path.exists(sagittal_path) and os.path.exists(coronal_path) and os.path.exists(axial_path):
                    self.sagittal_paths.append(sagittal_path)
                    self.coronal_paths.append(coronal_path)
                    self.axial_paths.append(axial_path)
                    self.labels.append(label)        


    def __len__(self):
        return len(self.labels) * 3

    def set_train_mode(self, is_train):
        self.is_train = is_train

    def __getitem__(self, idx):
        if idx % 3 == 0:
            img_path = self.sagittal_paths[idx//3]
        elif idx % 3 == 1:
            img_path = self.coronal_paths[idx//3]
        else:
            img_path = self.axial_paths[idx//3]
        nifti_file = nib.load(img_path)
        img = nifti_file.get_fdata().astype(np.float32)
        if self.transform:
            img = torch.FloatTensor(img).unsqueeze(0)
            img = self.transform(img)
            img = img.squeeze(0)
        
        img = (img - img.min()) / (img.max() - img.min())
        return img, np.float32(self.labels[idx // 3])
    

class ImageClassifier(nn.Module):
    def __init__(self, model_name='resnet34', dropout_rate=0.5, freeze_layers=7):
        super(ImageClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet34':
            self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            # Modify the first convolutional layer to accept 1 channel input
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.base_model.fc.in_features
        elif model_name == 'efficientnet':
            self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Modify the first convolutional layer to accept 1 channel input
            self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            num_ftrs = self.base_model.classifier[1].in_features
        elif model_name == 'densenet':
            self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            # Modify the first convolutional layer to accept 1 channel input
            self.base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.base_model.classifier.in_features
        elif model_name == 'inception':
            self.base_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            # Modify the first convolutional layer to accept 1 channel input
            self.base_model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
            num_ftrs = self.base_model.fc.in_features
        elif model_name == 'vgg':
            self.base_model = vgg16(weights=VGG16_Weights.DEFAULT)
            # Modify the first convolutional layer to accept 1 channel input
            self.base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            num_ftrs = self.base_model.classifier[6].in_features
        elif model_name == 'mobilenet':
            self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            # Modify the first convolutional layer to accept 1 channel input
            self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            num_ftrs = self.base_model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.layers_to_freeze = list(self.base_model.named_children())[:freeze_layers]
        self.freeze_layers()

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 1),
        )

        # Replace the original fc layer with Identity
        if model_name in ['resnet34', 'densenet', 'inception', 'xception']:
            self.base_model.fc = nn.Identity()
        elif model_name == 'efficientnet':
            self.base_model.classifier = nn.Identity()
        elif model_name == 'vgg':
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'mobilenet':
            self.base_model.classifier[1] = nn.Identity()

    def freeze_layers(self):
        for name, param in self.base_model.named_parameters():
            if any(name.startswith(layer_name) for layer_name, _ in self.layers_to_freeze):
                param.requires_grad = False

    def unfreeze_all_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.base_model(x)
        return self.fc(x)

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        with torch.no_grad():
            predictions = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
    return running_loss / len(train_loader), correct / total

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
    return running_loss / len(val_loader), correct / total

def plot_results(train_losses, train_accs, val_losses, val_accs, test_losses, test_accs, model_name):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(20, 6))  # Increase figure size
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.plot(epochs, test_losses, 'g-', label='Test Loss')
    plt.title(f'{model_name}\nTraining, Validation, and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.plot(epochs, test_accs, 'g-', label='Test Accuracy')
    plt.title(f'{model_name}\nTraining, Validation, and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accs, 'g-', label='Test Accuracy')
    plt.title(f'{model_name}\nTest Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'nn_results/{model_name}_training_results.png')
    plt.close()

def plot_precision_recall_curve(precisions, recalls, a_ps, save_file_name='precision_recall_curve.jpg'):
    plt.figure(figsize=(14, 10))
    
    # get min len of the precisions and recalls
    min_len = min([len(precision) for precision in precisions])
    # truncate the last precision and recall to the min_len
    precisions = [precision[:min_len] for precision in precisions]
    recalls = [recall[:min_len] for recall in recalls]
    for i, (precision, recall) in enumerate(zip(precisions, recalls)):
        ap = a_ps[i]
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label=f'Fold {i+1} (AP = {ap:.2f})')

    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_ap = np.mean(a_ps)
    std_ap = np.std(a_ps)
    plt.rcParams.update({'font.size': 16})
    plt.plot(mean_recall, mean_precision, color='b',
             label=f'Mean PR (AP = {mean_ap:.2f} ± {std_ap:.2f})',
             lw=2, alpha=.8)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_file_name)
    plt.close()

def plot_roc_curve(fprs, tprs, aucs, save_file_name='roc_curve.jpg'):
    plt.figure(figsize=(14, 10))
    
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        roc_auc = aucs[i]
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    
    plt.rcParams.update({'font.size': 16})
    # make the font bold
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
             lw=2, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_file_name)
    plt.close()

def train_and_evaluate(model_class, model_name, dataset, test_dataset, device, num_epochs, batch_size, learning_rate, weight_decay):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    val_precisions, val_recalls, val_aps = [], [], []
    test_precisions, test_recalls, test_aps = [], [], []
    val_fprs, val_tprs, val_aucs = [], [], []
    test_fprs, test_tprs, test_aucs = [], [], []

    val_misclassifications = {}
    test_misclassifications = {}

    # Define class names
    class_names = ['negative', 'positive']

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        
        # Reinitialize the model for each fold
        model = model_class('efficientnet', freeze_layers=0).to(device)
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Set the dataset mode before training
        dataset.set_train_mode(True)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion = nn.BCEWithLogitsLoss()

        fold_train_losses, fold_train_accs = [], []
        fold_val_losses, fold_val_accs = [], []
        fold_test_losses, fold_test_accs = [], []
        
        for epoch in range(num_epochs):
            model.freeze_layers()
            
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
            
            # Validate (set to False before validation)
            dataset.set_train_mode(False)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            
            # Test
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            
            # Set back to True for next training epoch
            dataset.set_train_mode(True)
            
            scheduler.step(val_loss)
            
            fold_train_losses.append(train_loss)
            fold_train_accs.append(train_acc)
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            fold_test_losses.append(test_loss)
            fold_test_accs.append(test_acc)
            
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        train_losses.append(fold_train_losses)
        train_accs.append(fold_train_accs)
        val_losses.append(fold_val_losses)
        val_accs.append(fold_val_accs)
        test_losses.append(fold_test_losses)
        test_accs.append(fold_test_accs)

        # Calculate precision-recall curve for validation set
        val_true, val_scores = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                outputs = torch.sigmoid(outputs)
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(outputs.cpu().numpy())
        
        val_true = np.array(val_true).flatten()
        val_scores = np.array(val_scores).flatten()
        val_precision, val_recall, _ = precision_recall_curve(val_true, val_scores)
        val_ap = average_precision_score(val_true, val_scores)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_aps.append(val_ap)

        # Calculate precision-recall curve for test set
        test_true, test_scores = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                outputs = torch.sigmoid(outputs)
                test_true.extend(labels.cpu().numpy())
                test_scores.extend(outputs.cpu().numpy())
        
        test_true = np.array(test_true).flatten()
        test_scores = np.array(test_scores).flatten()
        test_precision, test_recall, _ = precision_recall_curve(test_true, test_scores)
        test_ap = average_precision_score(test_true, test_scores)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_aps.append(test_ap)

        # Calculate ROC curve for validation set
        val_fpr, val_tpr, _ = roc_curve(val_true, val_scores)
        val_auc = auc(val_fpr, val_tpr)
        val_fprs.append(val_fpr)
        val_tprs.append(val_tpr)
        val_aucs.append(val_auc)

        # Calculate ROC curve for test set
        test_fpr, test_tpr, _ = roc_curve(test_true, test_scores)
        test_auc = auc(test_fpr, test_tpr)
        test_fprs.append(test_fpr)
        test_tprs.append(test_tpr)
        test_aucs.append(test_auc)

        # After training, evaluate on the validation set and track misclassifications
        # model.eval()
        # with torch.no_grad():
        #     for batch_idx, (inputs, labels, file_paths) in enumerate(val_loader):
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         predicted = torch.round(torch.sigmoid(outputs))
                
        #         # Check for misclassifications
        #         misclassified = predicted != labels
        #         for i, is_misclassified in enumerate(misclassified.flatten()):
        #             if is_misclassified:
        #                 idx = val_idx[batch_idx * batch_size + i]
        #                 if idx < len(dataset.file_paths):
        #                     true_label = labels[i].item()
        #                     pred_label = predicted[i].item()
        #                     if file_paths[i] not in val_misclassifications:
        #                         val_misclassifications[file_paths[i]] = []
        #                     val_misclassifications[file_paths[i]].append((fold, true_label, pred_label))

        # After training, evaluate on the test set and track misclassifications
        # model.eval()
        # with torch.no_grad():
        #     for batch_idx, (inputs, labels, file_paths) in enumerate(test_loader):
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         predicted = torch.round(torch.sigmoid(outputs))
                
        #         # Check for misclassifications
        #         misclassified = predicted != labels
        #         for i, is_misclassified in enumerate(misclassified.flatten()):
        #             if is_misclassified:
        #                 idx = batch_idx * batch_size + i
        #                 if idx < len(test_dataset.file_paths):
        #                     true_label = labels[i].item()
        #                     pred_label = predicted[i].item()
        #                     if file_paths[i] not in test_misclassifications:
        #                         test_misclassifications[file_paths[i]] = []
        #                     test_misclassifications[file_paths[i]].append((fold, true_label, pred_label))

    # Save the final model
    torch.save(model.state_dict(), f"{model_name}_final.pth")

    # Plot results
    # plot_results(train_losses, train_accs, val_losses, val_accs, test_losses, test_accs, model_name)
    
    # Plot precision-recall curves
    plot_precision_recall_curve(val_precisions, val_recalls, val_aps, f'nn_results/{model_name}_validation_pr_curve.png')
    plot_precision_recall_curve(test_precisions, test_recalls, test_aps, f'nn_results/{model_name}_test_pr_curve.png')

    # Plot ROC curves
    plot_roc_curve(val_fprs, val_tprs, val_aucs, f'nn_results/{model_name}_validation_roc_curve.png')
    plot_roc_curve(test_fprs, test_tprs, test_aucs, f'nn_results/{model_name}_test_roc_curve.png')

    # Analyze misclassifications
    # print("\nValidation Set Misclassifications:")
    # analyze_misclassifications(val_misclassifications, class_names)
    
    # print("\nTest Set Misclassifications:")
    # analyze_misclassifications(test_misclassifications, class_names)

def train_final_model(model, model_name, train_dataset, test_dataset, device, num_epochs, batch_size, learning_rate, weight_decay):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.freeze_layers()
        
        # Set to train mode
        train_dataset.set_train_mode(True)
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        
        # Set to eval mode for testing
        train_dataset.set_train_mode(False)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        scheduler.step(test_loss)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save the best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict()

    # Save the best model
    torch.save(best_model_state, f"{model_name}_best.pth")

    # Plot results
    # plot_results([train_losses], [train_accs], [test_losses], [test_accs], [test_losses], [test_accs], f"{model_name}_final")

    # Calculate and plot precision-recall curve for test set
    test_true, test_scores = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_true.extend(labels.cpu().numpy())
            test_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    
    test_precision, test_recall, _ = precision_recall_curve(test_true, test_scores)
    test_ap = average_precision_score(test_true, test_scores)
    
    plot_precision_recall_curve([test_precision], [test_recall], [test_ap], f'nn_results/{model_name}_final_test_pr_curve.png')

    # Calculate and plot ROC curve for test set
    test_fpr, test_tpr, _ = roc_curve(test_true, test_scores)
    test_auc = auc(test_fpr, test_tpr)
    
    plot_roc_curve([test_fpr], [test_tpr], [test_auc], f'nn_results/{model_name}_final_test_roc_curve.png')

def analyze_misclassifications(misclassifications, class_names):
    # Sort misclassifications by the number of folds they appear in
    sorted_misclassifications = sorted(
        misclassifications.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print("\nMost common misclassifications:")
    for file_path, misclassified_info in sorted_misclassifications[:10]:  # Show top 10
        print(f"File: {file_path}")
        print(f"Misclassified in {len(misclassified_info)} folds:")
        for fold, true_label, pred_label in misclassified_info:
            print(f"  Fold {fold}: True class: {class_names[true_label]}, Predicted class: {class_names[pred_label]}")
        print()

    # Calculate and print statistics
    total_misclassified = len(misclassifications)
    misclassified_in_all_folds = sum(1 for info in misclassifications.values() if len(info) == 5)
    
    print(f"Total misclassified scans: {total_misclassified}")
    print(f"Scans misclassified in all 5 folds: {misclassified_in_all_folds}")
    print(f"Percentage of consistently misclassified scans: {misclassified_in_all_folds / total_misclassified * 100:.2f}%")

def get_train_transform():
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.RandomAutocontrast(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        # random blur:
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),  
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),

    ])

def get_val_transform():
    return transforms.Compose([ 
    ])

def visualize_transforms(dataset, num_samples=5):
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    
    for i in range(num_samples):
        dataset.set_train_mode(False)
        # Get the original image without transforms
        img, label = dataset[i]

        img_np = img.numpy()[0]
        # Calculate statistics
        min_val = np.min(img_np)
        max_val = np.max(img_np)
        mean_val = np.mean(img_np)
        std_val = np.std(img_np)
        
        # Plot original image
        axes[0, i].imshow(img_np, cmap='gray')
        axes[0, i].set_title(f"Original (Label: {label})\nMin: {min_val:.2f}, Max: {max_val:.2f}\nMean: {mean_val:.2f}, Std: {std_val:.2f}")
        axes[0, i].axis('off')
        
        # Plot histogram of original image
        axes[1, i].hist(img_np.ravel(), bins=256, density=True)
        axes[1, i].set_title("Histogram")
        axes[1, i].set_xlabel("Pixel Intensity")
        axes[1, i].set_ylabel("Frequency")
        
        # Apply transforms and plot
        if dataset.transform:
            # Convert to tensor first
            dataset.set_train_mode(True)
            transformed_img, label, file_path = dataset[i]
            transformed_img_np = transformed_img.numpy()[0]
            # Calculate statistics for transformed image
            t_min_val = np.min(transformed_img_np)
            t_max_val = np.max(transformed_img_np)
            t_mean_val = np.mean(transformed_img_np)
            t_std_val = np.std(transformed_img_np)
            
            axes[2, i].imshow(transformed_img_np, cmap='gray')
            axes[2, i].set_title(f"Transformed\nMin: {t_min_val:.2f}, Max: {t_max_val:.2f}\nMean: {t_mean_val:.2f}, Std: {t_std_val:.2f}")
            axes[2, i].axis('off')
            
            # Plot histogram of transformed image
            axes[1, i].hist(transformed_img_np.ravel(), bins=256, density=True, alpha=0.7)
            axes[1, i].legend(['Original', 'Transformed'])

    plt.tight_layout()
    plt.savefig('nn_results/transform_examples_with_histograms_and_stats.png')
    plt.close()

def main():
    # Create results directory if it doesn't exist
    res_dir = 'nn_results_3_class_output'
    os.makedirs(res_dir, exist_ok=True)

    # Load data
    with open('alon_labels_filtered.json', 'r') as f:
        alon_labels = json.load(f)
    with open('oo_test_labels_filtered.json', 'r') as f:
        oo_test_labels = json.load(f)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training parameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0005
    weight_decay = 1e-4

    # Get transforms
    train_transform = get_train_transform()

    # Create datasets
    train_dataset = BrainMRIDataset(alon_labels, transform=train_transform)
    test_dataset = BrainMRIDataset(oo_test_labels, transform=None)

    

    train_and_evaluate(ImageClassifier, 'efficientnet', train_dataset, test_dataset, device, num_epochs, batch_size, learning_rate, weight_decay)

    # train_final_model(model, f"{model.model_name}_Final", train_dataset, test_dataset, device, num_epochs, batch_size, learning_rate, weight_decay)

    # Visualize transforms
    #visualize_transforms(train_dataset)

    # Add these lines to check class distribution
    train_labels = [label for series in alon_labels.values() for label in series.values()]
    test_labels = [label for series in oo_test_labels.values() for label in series.values()]
    
    print("Class distribution:")
    print(f"Train: {sum(train_labels)}/{len(train_labels)} positive")
    print(f"Test: {sum(test_labels)}/{len(test_labels)} positive")

if __name__ == "__main__":
    main()