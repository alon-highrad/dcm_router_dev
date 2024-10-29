from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import nibabel as nib
import json
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tools import plot_precision_recall_curve, plot_roc_curve
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import onnx
import onnxruntime
from dcm_router.classifier.mri_classification.brain_gd_classification import preprocess_series
from dcm_router.classifier.mri_classification.mri_brain_roi_classification import get_brain_segmentation

torch.manual_seed(42)

def save_preprocessed_dataset(path_to_json):
    # Load the JSON file created by create_labels.py
    with open(path_to_json, 'r') as f:
        dataset = json.load(f)
    
    # Create MIP images for every series specified in the JSON file
    for study, series in dataset.items():
        for file_path in series.keys():
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
                image_nifti = nib.load(file_path)
                image_data = image_nifti.get_fdata()
                brain_mask_nif = get_brain_segmentation(image_nifti, device="gpu")
                mask_data = brain_mask_nif.get_fdata()
                mip_list = preprocess_series(image_data, mask_data)
                
                for plane,mip in zip(('sagittal', 'coronal', 'axial'), mip_list):
                    res_path = file_path.replace('.nii.gz', '_' + plane + '_mip.nii.gz')
                    nib.save(nib.Nifti1Image(mip.numpy(), image_nifti.affine), res_path)
                    torch.save(mip, res_path.replace('.nii.gz', '.pt'))
            else:
                print(f"File not found: {file_path}")

class BrainMRIDataset(Dataset):
    def __init__(self, data_dict):
        self.mip_paths = []
        self.labels = []
        for _, series in data_dict.items():
            for file_path, label in series.items():
                sagittal_path = file_path.replace('.nii.gz', '_sagittal_mip.pt')
                coronal_path = file_path.replace('.nii.gz', '_coronal_mip.pt')
                axial_path = file_path.replace('.nii.gz', '_axial_mip.pt')
                if os.path.exists(sagittal_path) and os.path.exists(coronal_path) and os.path.exists(axial_path):
                    if not torch.any(torch.isnan(torch.load(sagittal_path))):
                        self.mip_paths.append((sagittal_path, coronal_path, axial_path))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mip_paths = self.mip_paths[idx]
        return torch.load(mip_paths[0]), torch.load(mip_paths[1]), torch.load(mip_paths[2]), self.labels[idx]


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
            dummy_input = torch.randn(1, 3, 128, 128)
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
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for sagittal, coronal, axial, labels in train_loader:
                sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(sagittal, coronal, axial)
                # for p in mip_paths[0]:
                #     print(p)
                # print(outputs)
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

def save_model_as_onnx(model_path, onnx_path):
    # Load the saved PyTorch model
    model = MultiViewEfficientNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input tensors
    dummy_sagittal = torch.randn(1, 1, 128, 128)
    dummy_coronal = torch.randn(1, 1, 128, 128)
    dummy_axial = torch.randn(1, 1, 128, 128)

    # Export the model to ONNX
    torch.onnx.export(model,
                      (dummy_sagittal, dummy_coronal, dummy_axial),
                      onnx_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['sagittal', 'coronal', 'axial'],
                      output_names=['output'],
                      dynamic_axes={'sagittal': {0: 'batch_size'},
                                    'coronal': {0: 'batch_size'},
                                    'axial': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    print(f"Model saved as ONNX: {onnx_path}")

    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checked successfully")

    # Test the ONNX model with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_path)

    ort_inputs = {
        'sagittal': dummy_sagittal.numpy(),
        'coronal': dummy_coronal.numpy(),
        'axial': dummy_axial.numpy()
    }
    ort_output = ort_session.run(None, ort_inputs)
    print("ONNX model inference test passed")

def main(overwrite_preprocessed_data=False):

    # Create preprocessed dataset
    if overwrite_preprocessed_data:
        save_preprocessed_dataset('hadassah_dataset.json')
        save_preprocessed_dataset('english_dataset.json')

    # Create results directory if it doesn't exist
    res_dir = 'multi_efficientnet_results'
    os.makedirs(res_dir, exist_ok=True)

    # # Load data
    with open('hadassah_dataset.json', 'r') as f:
        hadassah_dataset = json.load(f)
    with open('english_dataset.json', 'r') as f:
        english_dataset = json.load(f)

    # Set up device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Training parameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 1e-4

    train_dataset = BrainMRIDataset(hadassah_dataset)
    test_dataset = BrainMRIDataset(english_dataset)
    print(len(train_dataset))

    # # Train the model
    train_model(train_dataset, test_dataset, num_epochs, batch_size, learning_rate, weight_decay, device, res_dir)

    # Save the model as ONNX
    pytorch_model_path = os.path.join(res_dir, 'final_multiview_model.pth')
    onnx_model_path = os.path.join(res_dir, 'final_multiview_model.onnx')
    save_model_as_onnx(pytorch_model_path, onnx_model_path)

if __name__ == '__main__':
    main(overwrite_preprocessed_data=False)