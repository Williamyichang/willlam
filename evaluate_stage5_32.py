import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Use modified model
from models.hybrid_model_evaluate_stage5_32 import MyImprovedFourStageClassifier
from utils.visualization import plot_confusion_matrix, plot_tsne, generate_gradcam

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloaders(config):
    # Simple data loaders using ImageFolder.
    image_size = config["data"]["image_size"]
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root=config["data"]["train_dir"], transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=config["data"]["val_dir"], transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    return train_loader, val_loader

def evaluate():
    '''
    Evaluate the model on the validation set.
    :return: None
    '''
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # >>> CHANGED: Force the model size to "xtiny" so that the current configuration matches the checkpoint.
    config["model"]["size"] = "medium"
    print("Forced model size to 'medium' for evaluation.")
    # <<< END OF CHANGE

    # Instantiate the model with the full configuration dictionary.
    model = MyImprovedFourStageClassifier(config).to(device)
    
    # Load checkpoint that was saved using the "xtiny" configuration.
    checkpoint = torch.load("checkpoints/overall_training/best_checkpoint_medium_hw_32.pth", map_location=device)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    _, val_loader = get_dataloaders(config)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    cmatrix = confusion_matrix(all_labels, all_preds)
    class_names = [str(i) for i in range(config["model"]["num_classes"])]
    plot_confusion_matrix(cmatrix, class_names, save_path="confusion_matrix_medium_hw_32.png")

    # t-SNE visualization using CNN backbone features.
    features = []
    labels_list = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            feat = model.cnn(images)
            features.append(feat.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    features = np.concatenate(features, axis=0)
    plot_tsne(features, labels_list, save_path="tsne__medium_hw_32.png")

    # Generate Grad-CAM on a sample.
    sample_images, _ = next(iter(val_loader))
    generate_gradcam(model, sample_images[0:1], target_class=None, save_path="gradcam_medium_hw_32.png")

if __name__ == "__main__":
    evaluate()
