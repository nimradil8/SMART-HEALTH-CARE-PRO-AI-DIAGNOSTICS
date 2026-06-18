import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
import os
import sys

# --- 1. CONFIGURATION ---
# Use absolute paths or ensure your terminal is in the correct directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "medical_classifier_v1.pth")
CLASS_NAMES = ["High Risk", "Normal", "Medium Risk"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL LOADER ---
def get_model():
    print(f"🔄 Loading model on {DEVICE}...")
    try:
        model = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_PATH}")
        sys.exit(1)

# --- 3. PREDICTION ENGINE ---
class MedicalPredictor:
    def __init__(self):
        self.model = get_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return f"❌ File not found: {image_path}"

        # Image processing
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        return {
            "filename": os.path.basename(image_path),
            "status": CLASS_NAMES[pred.item()],
            "confidence": f"{conf.item()*100:.2f}%"
        }

# --- 4. EXECUTION ---
if __name__ == "__main__":
    predictor = MedicalPredictor()
    
    # You can pass an image via command line or hardcode it here
    test_image = os.path.join(BASE_DIR, "test.png") 
    
    result = predictor.predict(test_image)
    
    if isinstance(result, dict):
        print("\n" + "="*30)
        print(f"🏥 MEDICAL ANALYSIS")
        print("="*30)
        print(f"File:       {result['filename']}")
        print(f"Diagnosis:  {result['status']}")
        print(f"Confidence: {result['confidence']}")
        print("="*30 + "\n")
    else:
        print(result)