import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import timm
import os

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Looking for the model in the same folder as this script
MODEL_PATH = os.path.join(BASE_DIR, "models","medical_classifier_v1.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["High Risk", "Normal", "Medium Risk"]

# Global variable to store gradients from the backward hook
grad_map = None

# --- 2. MODEL LOADER ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Cannot find '{MODEL_PATH}'")
        return None
        
    print(f"🔄 Loading model on {DEVICE}...")
    # Using the exact architecture from your training
    model = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- 3. GRAD-CAM CORE ENGINE ---
def generate_gradcam(image_path):
    global grad_map
    model = load_model()
    if model is None: return

    # A. Prepare Image
    img_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    # B. Define Hooks
    features = []
    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        global grad_map
        grad_map = grad_output[0]

    # Target the last convolutional layer of EfficientNet-V2
    target_layer = model.conv_head
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)

    # C. Forward & Backward Pass
    output = model(input_tensor)
    idx = torch.argmax(output).item()
    
    model.zero_grad()
    output[0, idx].backward()

    # D. Calculate Heatmap
    # activations: [1, Channels, H, W] | gradients: [1, Channels, H, W]
    activations = features[0].detach()
    gradients = grad_map.detach()

    # Pool gradients to get channel importance weights
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the channels
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the weighted channels to get 2D heatmap
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0) # ReLU
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap) # Normalize 0 to 1

    # E. Visual Processing with OpenCV
    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv = cv2.resize(img_cv, (224, 224))
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    combined = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # F. Display Results
    print(f"✅ Prediction: {CLASS_NAMES[idx]}")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_cv)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combined)
    plt.title(f"Heatmap: {CLASS_NAMES[idx]}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # G. Cleanup
    h_forward.remove()
    h_backward.remove()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Checks for both .png and .jpg automatically
    possible_files = ["test.png", "test.jpg", "test.jpeg"]
    img_to_test = None

    for f in possible_files:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            img_to_test = path
            break

    if img_to_test:
        print(f"📸 Testing with: {os.path.basename(img_to_test)}")
        generate_gradcam(img_to_test)
    else:
        print("❌ Error: No 'test.png' or 'test.jpg' found in the folder.")