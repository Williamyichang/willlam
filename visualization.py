# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
import cv2

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_tsne(features, labels, save_path="tsne.png"):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE of Features")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def generate_gradcam(model, input_tensor, target_class, save_path="gradcam.png"):
    # Ensure the input tensor is on the same device as the model.
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Choose the target layer: "stage5.channel_attn"
    try:
        target_layer = model.stage5[0].channel_attn
    except AttributeError:
        available = [name for name, _ in model.named_modules()]
        raise AttributeError(f"Target layer 'model.stage5.channel_attn' not found. Available layers: {available}")

    # Register hooks: use register_full_backward_hook for full gradient info.
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass.
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    # Create a one-hot vector for the target class.
    one_hot = torch.zeros_like(output)
    one_hot[range(output.shape[0]), target_class] = 1

    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=True)

    with torch.no_grad():
        # Global-average pool the gradients and compute weighted sum of features.
        weights = gradients[0].mean(dim=[2, 3], keepdim=True)
        cam_map = (weights * activations[0]).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)
        cam_map = torch.nn.functional.interpolate(cam_map, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Detach and convert to NumPy.
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        cam_map = cam_map.squeeze().detach().cpu().numpy()

    # Create heatmap.
    import cv2
    import matplotlib.pyplot as plt
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Prepare the input image for overlay.
    input_img = input_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = std * input_img + mean
    input_img = np.clip(input_img, 0, 1)
    input_img = np.uint8(255 * input_img)

    # Overlay the heatmap on the input image.
    superimposed_img = cv2.addWeighted(input_img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.savefig(save_path)
    plt.close()
    print(f"Grad-CAM visualization saved to {save_path}")




# Compare this snippet from scripts/yaml_test.py:
# from train import load_config