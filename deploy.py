# scripts/deploy.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import cv2
from torchvision import transforms
from models.hybrid_model import MyImprovedFourStageClassifier

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_image(image, config):
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["model"]["size"] = "small"
    model = MyImprovedFourStageClassifier(config).to(device)

    checkpoint = torch.load("checkpoints/overall_training/best_checkpoint_medium_hw_32.pth", map_location=device)

    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = preprocess_image(frame, config)
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            pred = outputs.argmax(dim=1).item()
        cv2.putText(frame, f"Prediction: {pred}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()