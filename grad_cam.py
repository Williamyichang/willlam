import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

# Import the FastViT model variant from your uploaded file
from fastvit import fastvit_t12

# -------------------------
# 1. Load the Model and Checkpoint
# -------------------------
# Instantiate the model
model = fastvit_t12(pretrained=False)
model.eval()  # set the model to evaluation mode

# Specify the checkpoint path (.pt file)
checkpoint_path = 'path_to_checkpoint.pt'  # Replace with your checkpoint path
# Load the checkpoint (adjust map_location as needed)
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract the state dictionary (handle cases with 'state_dict' key)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Remove DataParallel "module." prefix if present
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

# Load the state dictionary into the model
model.load_state_dict(new_state_dict)
print("Checkpoint loaded successfully!")

# -------------------------
# 2. Define the Grad-CAM Helper Class
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        # Forward hook to capture the activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        # Backward hook to capture the gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        # If no class is specified, choose the one with highest score
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        # Backward pass for the target class score
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Global-average pool the gradients and compute weighted combination of activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam_map = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # Upsample the heatmap to match the input size
        grad_cam_map = F.interpolate(grad_cam_map, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        # Normalize the heatmap between 0 and 1
        grad_cam_map = (grad_cam_map - np.min(grad_cam_map)) / (np.max(grad_cam_map) - np.min(grad_cam_map) + 1e-8)
        return grad_cam_map, output

# -------------------------
# 3. Setup Grad-CAM on a Target Layer
# -------------------------
# For example, we use the last layer of the patch embedding as the target layer
target_layer = model.patch_embed[-1]
grad_cam = GradCAM(model, target_layer)

# -------------------------
# 4. Load and Preprocess an Input Image
# -------------------------
img_path = 'path_to_your_image.jpg'  # Replace with the path to your input image
img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# -------------------------
# 5. Generate and Visualize the Grad-CAM Heatmap
# -------------------------
cam, output = grad_cam(input_tensor)

# Visualize the Grad-CAM heatmap overlay on the original image
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()

# Clean up the hooks
grad_cam.remove_hooks()
