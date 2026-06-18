import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import timm
import json
import os
import datetime
from torchvision import transforms
from fpdf import FPDF
import easyocr
import plotly.graph_objects as go
import requests
import io
import base64
import tempfile
import rppg
import streamlit.components.v1 as components
import time
# Hardcoded API Key
openrouter_key = "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"

# --- 1. PAGE CONFIG & PERSISTENT SESSION ENGINE ---
st.set_page_config(page_title="Smart Health Care | Pro AI", layout="wide")

def apply_global_luxury_theme():
    video_path = "app/static/290119.mp4" 
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
        
        [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
        [data-testid="stSidebar"], .main, .stApp, [data-testid="stMainViewContainer"] {{
            background-color: transparent !important;
            background: transparent !important;
        }}

        #bgVideo {{
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            object-fit: cover; z-index: -2; 
            filter: brightness(0.6) contrast(1.1);
        }}

        .video-overlay {{
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(2, 6, 23, 0.4); 
            z-index: -1;
        }}

        .hero-banner {{
            background: rgba(15, 23, 42, 0.5); backdrop-filter: blur(20px);
            padding: 3rem 2rem; margin: -80px -100px 40px -100px;
            text-align: center; 
            border-bottom: 2px solid #00C8D2; 
            box-shadow: 0 4px 15px rgba(0, 200, 210, 0.2);
        }}

        .hero-title {{ 
            font-family: 'Orbitron', sans-serif;
            color: #00C8D2 !important;
            text-shadow: 0 0 10px #00C8D2;
            font-size: 3.5rem !important; font-weight: 800;
        }}

        .neon-blue-header {{
            color: #00C8D2 !important;
            text-shadow: 0 0 8px #00C8D2;
            font-family: 'Orbitron', sans-serif;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
        }}

        .glass-card {{
            background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(15px);
            border-radius: 24px; padding: 25px; margin-bottom: 20px;
            color: #E2E8F0 !important;
            border: 1px solid #00C8D2;
            box-shadow: 0 0 8px rgba(0, 200, 210, 0.1);
        }}

        .triage-row {{
            background: rgba(0, 0, 0, 0.5); 
            border: 2px solid #00C8D2; 
            box-shadow: 0 0 10px rgba(0, 200, 210, 0.3);
            border-radius: 15px; padding: 15px; margin-bottom: 15px;
            display: flex; justify-content: space-between; align-items: center;
            color: #00C8D2 !important;
            font-family: 'Orbitron', sans-serif;
        }}

        .evidence-card {{
            background: rgba(0, 0, 0, 0.65); 
            border: 2px solid #00C8D2; 
            box-shadow: 0 0 15px rgba(0, 200, 210, 0.2);
            border-radius: 20px; 
            padding: 25px; 
            margin-bottom: 30px;
            color: #00C8D2 !important;
            font-family: 'Orbitron', sans-serif;
        }}

        .diag-hero {{ 
            font-family: 'Orbitron', sans-serif; 
            font-size: 2.5rem; 
            text-align: center; 
            color: #00C8D2; 
            text-shadow: 0 0 8px #00C8D2; 
        }}
        </style>
        <video autoplay muted loop playsinline id="bgVideo">
            <source src="{video_path}" type="video/mp4">
        </video>
        <div class="video-overlay"></div>
    """, unsafe_allow_html=True)

# --- 2. BACKEND ENGINE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "medical_classifier_v1.pth")
PERSISTENT_DATA_PATH = os.path.join(BASE_DIR, "persistent_data.json")
FEEDBACK_DATA_PATH = os.path.join(BASE_DIR, "feedback_data.json")
CLASS_NAMES = ["High Risk", "Normal", "Medium Risk"]
SEVERITY_MAP = {"High Risk": 2, "Medium Risk": 1, "Normal": 0}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_persistent_data():
    """Load scan history and counters from disk."""
    default = {
        "total_scans": 0,
        "high_risk_count": 0,
        "high_risk_details": [],
        "scan_history": [],
        "global_emergency_queue": [],
        "last_scan_result": None
    }
    if os.path.exists(PERSISTENT_DATA_PATH):
        try:
            with open(PERSISTENT_DATA_PATH, "r") as f:
                data = json.load(f)
            for key in default:
                if key not in data:
                    data[key] = default[key]
            return data
        except Exception:
            return default
    return default

def save_persistent_data(data):
    """Save scan history and counters to disk."""
    try:
        with open(PERSISTENT_DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except TypeError as e:
        # Silently skip non-serializable data rather than crashing
        pass
    except Exception as e:
        st.error(f"Failed to save persistent data: {e}")

def sync_session_to_disk():
    """Sync current session state counters to persistent storage."""
    # Clean emergency queue - remove any entries with non-serializable objects
    clean_queue = []
    for item in st.session_state.global_emergency_queue:
        if isinstance(item, dict):
            clean_item = {k: v for k, v in item.items() if k not in ["orig", "heatmap", "pixels"]}
            clean_queue.append(clean_item)

    # Clean last_scan_result
    last_scan = st.session_state.last_scan_result
    if isinstance(last_scan, dict):
        last_scan = {k: v for k, v in last_scan.items() if k not in ["orig", "heatmap", "pixels"]}

    # Clean high_risk_details
    clean_details = []
    for item in st.session_state.high_risk_details:
        if isinstance(item, dict):
            clean_item = {k: v for k, v in item.items() if k not in ["orig", "heatmap", "pixels"]}
            clean_details.append(clean_item)

    save_persistent_data({
        "total_scans": st.session_state.total_scans,
        "high_risk_count": st.session_state.high_risk_count,
        "high_risk_details": clean_details,
        "scan_history": st.session_state.scan_history,
        "global_emergency_queue": clean_queue,
        "last_scan_result": last_scan
    })



# --- FEEDBACK SYSTEM ---
def load_feedback_data():
    """Load all user feedback from disk."""
    if os.path.exists(FEEDBACK_DATA_PATH):
        try:
            with open(FEEDBACK_DATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_feedback_data(feedback_list):
    """Save feedback list to disk."""
    try:
        with open(FEEDBACK_DATA_PATH, "w") as f:
            json.dump(feedback_list, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

def render_stars(rating, max_stars=5):
    """Render star rating as HTML string."""
    full_star = "★"
    empty_star = "☆"
    filled = int(rating)
    return full_star * filled + empty_star * (max_stars - filled)

def submit_feedback(name, email, overall, ui_rating, accuracy_rating, speed_rating, review, suggestions):
    """Submit new feedback entry to persistent storage."""
    feedback_list = load_feedback_data()
    entry = {
        "id": len(feedback_list) + 1,
        "name": name,
        "email": email,
        "overall_rating": overall,
        "ui_rating": ui_rating,
        "accuracy_rating": accuracy_rating,
        "speed_rating": speed_rating,
        "review": review,
        "suggestions": suggestions,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    feedback_list.insert(0, entry)
    save_feedback_data(feedback_list)
    return True

def render_feedback_stats():
    """Render feedback statistics and charts."""
    feedback_list = load_feedback_data()
    if not feedback_list:
        st.info("No feedback submitted yet. Be the first to share your experience!")
        return

    total = len(feedback_list)
    avg_overall = sum(f["overall_rating"] for f in feedback_list) / total
    avg_ui = sum(f["ui_rating"] for f in feedback_list) / total
    avg_accuracy = sum(f["accuracy_rating"] for f in feedback_list) / total
    avg_speed = sum(f["speed_rating"] for f in feedback_list) / total

    st.markdown('<h3 class="neon-blue-header">📊 Feedback Analytics</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews", total)
    c2.metric("Overall", f"{avg_overall:.1f}/5")
    c3.metric("UI / UX", f"{avg_ui:.1f}/10")
    c4.metric("Accuracy", f"{avg_accuracy:.1f}/10")
    c5.metric("Speed", f"{avg_speed:.1f}/10")

    st.markdown('<h4 style="color:#00C8D2;font-family:Orbitron;margin-top:15px;">⭐ Overall Rating Distribution</h4>', unsafe_allow_html=True)
    dist = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for f in feedback_list:
        r = int(f["overall_rating"])
        if r in dist:
            dist[r] += 1

    for star, count in dist.items():
        pct = (count / total * 100) if total > 0 else 0
        bar_html = f'''<div style="background:rgba(0,200,210,0.1);border-radius:4px;height:22px;width:100%;margin:5px 0;overflow:hidden;">
            <div style="background:linear-gradient(90deg, #00C8D2, #0891b2);height:22px;border-radius:4px;width:{pct}%;display:flex;align-items:center;padding-left:10px;color:#020617;font-size:0.8rem;font-weight:bold;white-space:nowrap;">
                {star}★ {count} review{"s" if count != 1 else ""} ({pct:.0f}%)
            </div>
        </div>'''
        st.markdown(bar_html, unsafe_allow_html=True)

def render_feedback_list(limit=50):
    """Render list of feedback entries in styled cards."""
    feedback_list = load_feedback_data()[:limit]
    if not feedback_list:
        return

    st.markdown('<h3 class="neon-blue-header">📝 User Reviews & Suggestions</h3>', unsafe_allow_html=True)
    for fb in feedback_list:
        stars = render_stars(fb["overall_rating"])

        if fb.get("suggestions"):
            suggestion_block = (
                '<div style="color:#E2E8F0;font-size:0.9rem;line-height:1.5;border-top:1px solid rgba(0,200,210,0.2);padding-top:10px;margin-top:10px;">'
                '<span style="color:#00C8D2;font-weight:600;">💡 Suggestions for Future Implementation:</span><br>'
                + fb["suggestions"] +
                '</div>'
            )
        else:
            suggestion_block = ""

        card_html = (
            '<div class="glass-card" style="border-left:4px solid #00C8D2;margin-bottom:18px;">'
            '  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">'
            '    <span style="color:#00C8D2;font-family:Orbitron;font-weight:bold;font-size:1.05rem;">' + fb["name"] + '</span>'
            '    <span style="color:#64748B;font-size:0.75rem;">' + fb["timestamp"] + '</span>'
            '  </div>'
            '  <div style="color:#FBBF24;font-size:1.3rem;margin-bottom:8px;letter-spacing:3px;">' + stars + '</div>'
            '  <div style="color:#94A3B8;font-size:0.8rem;margin-bottom:10px;">📧 ' + fb["email"] + '</div>'
            '  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
            '    <div style="background:rgba(0,200,210,0.08);border:1px solid rgba(0,200,210,0.15);border-radius:8px;padding:8px 10px;text-align:center;">'
            '      <div style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">UI / UX</div>'
            '      <div style="color:#00C8D2;font-weight:bold;font-size:1.1rem;">' + str(fb["ui_rating"]) + '/10</div>'
            '    </div>'
            '    <div style="background:rgba(0,200,210,0.08);border:1px solid rgba(0,200,210,0.15);border-radius:8px;padding:8px 10px;text-align:center;">'
            '      <div style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Accuracy</div>'
            '      <div style="color:#00C8D2;font-weight:bold;font-size:1.1rem;">' + str(fb["accuracy_rating"]) + '/10</div>'
            '    </div>'
            '    <div style="background:rgba(0,200,210,0.08);border:1px solid rgba(0,200,210,0.15);border-radius:8px;padding:8px 10px;text-align:center;">'
            '      <div style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Speed</div>'
            '      <div style="color:#00C8D2;font-weight:bold;font-size:1.1rem;">' + str(fb["speed_rating"]) + '/10</div>'
            '    </div>'
            '  </div>'
            '  <div style="color:#E2E8F0;font-size:0.92rem;line-height:1.6;">'
            '    <span style="color:#00C8D2;font-weight:600;">Review:</span> ' + fb["review"] +
            '  </div>'
            + suggestion_block +
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

def _init_session_state():
    """Initialize session state with persistent data."""
    data = load_persistent_data()
    if 'total_scans' not in st.session_state:
        st.session_state.total_scans = data["total_scans"]
    if 'high_risk_count' not in st.session_state:
        st.session_state.high_risk_count = data["high_risk_count"]
    if 'high_risk_details' not in st.session_state:
        st.session_state.high_risk_details = data["high_risk_details"]
    if 'last_scan_result' not in st.session_state:
        st.session_state.last_scan_result = data.get("last_scan_result", None)
    if 'global_emergency_queue' not in st.session_state:
        st.session_state.global_emergency_queue = data.get("global_emergency_queue", [])
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_dashboard' not in st.session_state:
        st.session_state.show_dashboard = False
    if 'scan_history' not in st.session_state:
        st.session_state.scan_history = data["scan_history"]

# Run initialization on module load
_init_session_state()


def load_model():
    model = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=3)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def get_ocr_reader(): 
    return easyocr.Reader(['en'])

def anonymize_image(img_pil):
    reader = get_ocr_reader()
    img_np = np.array(img_pil)
    results = reader.readtext(img_np)
    redacted_img = img_np.copy()
    for (bbox, text, prob) in results:
        top_left, bottom_right = tuple(map(int, bbox[0])), tuple(map(int, bbox[2]))
        cv2.rectangle(redacted_img, top_left, bottom_right, (0, 0, 0), -1)
    return Image.fromarray(redacted_img), len(results)

def measure_lesion_area(heatmap, threshold=0.6):
    gray_heatmap = np.uint8(255 * heatmap)
    _, thresh = cv2.threshold(gray_heatmap, int(255 * threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, None
    largest_contour = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    pixel_to_mm_ratio = 0.26 
    area_mm2 = area_px * (pixel_to_mm_ratio ** 2)
    diameter_mm = max(w, h) * pixel_to_mm_ratio
    return round(area_mm2, 2), round(diameter_mm, 2), (x, y, w, h)

def get_layered_gradcam(model, img_pil, layer_key):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    layer_map = {
        "Early": model.blocks[0], 
        "Middle": model.blocks[3], 
        "Mid-Deep": model.blocks[5], 
        "Deep": model.conv_head
    }
    target_layer = layer_map[layer_key]
    features, grads = [], []
    def f_hook(m, i, o): features.append(o)
    def b_hook(m, gi, go): grads.append(go[0])
    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)
    output = model(input_tensor)
    idx = torch.argmax(output).item()
    model.zero_grad()
    output[0, idx].backward()
    weights = torch.mean(grads[0], dim=[0, 2, 3])
    cam = torch.zeros(features[0].shape[2:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(weights): cam += w * features[0][0, i, :, :]
    cam = np.maximum(cam.detach().cpu().numpy(), 0)
    if cam.max() != 0: cam /= cam.max()
    cam = cv2.resize(cam, (224, 224))
    h1.remove(); h2.remove()
    return cam, idx, torch.max(F.softmax(output, dim=1)).item()

# --- GEMINI OPENROUTER INTEGRATION ---
def call_gemini_openrouter(api_key, label, img_pil):
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-4-maverick", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are an expert AI medical analyst. This scan was classified as '{label}'. Analyze the image and provide a highly structured clinical report. Include exactly these 4 sections:\n1. Disease Summary\n2. Precautions\n3. Medication Suggestions (general)\n4. Suggestions/Next Steps\nAvoid emojis and use clear text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            payload["model"] = "google/gemini-2.0-flash-001"
            res_fallback = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if res_fallback.status_code == 200:
                return res_fallback.json()['choices'][0]['message']['content']
            return f"API Error: {res_fallback.status_code} - {res_fallback.text}"
    except Exception as e:
        return f"Failed to connect to OpenRouter API: {str(e)}"

# --- AESTHETIC PDF GENERATOR ---
class PDF_Report(FPDF):
    def __init__(self):
        super().__init__()
        self.logo_path = os.path.join(BASE_DIR, "app/static/logo.png") 

    def add_watermark(self):
        if os.path.exists(self.logo_path):
            self.set_alpha(0.1)
            self.image(self.logo_path, x=45, y=100, w=120)
            self.set_alpha(1)

    def draw_page_border(self):
        self.set_line_width(0.5)
        self.set_draw_color(0, 160, 210)
        self.rect(5, 5, 200, 287)

def create_advanced_pdf(img1, img2, label, filename, conf_txt, gemini_text):
    dept = "Neurology / Radiology" 
    if any(x in filename.lower() for x in ["lung", "chest", "xray", "x-ray"]):
        dept = "Pulmonology / Radiology"
    elif any(x in filename.lower() for x in ["skin", "lesion", "derm", "mole"]):
        dept = "Dermatology"

    pdf = PDF_Report()
    pdf.add_page()
    pdf.draw_page_border()
    pdf.add_watermark()

    pdf.set_fill_color(220, 20, 60)
    pdf.rect(10, 10, 12, 4, 'F') 
    pdf.rect(14, 6, 4, 12, 'F')  
    
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(15, 65, 105) 
    pdf.set_x(30)
    pdf.cell(0, 10, "SMART HEALTH CARE | CLINICAL REPORT", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_x(30)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)

    pdf.set_fill_color(0, 160, 210) 
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, " CASE IDENTIFICATION & DIAGNOSTICS", ln=True, fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    pdf.ln(2)
    data = [
        ["File Reference", filename[:40]],
        ["Primary Diagnosis", label],
        ["AI Confidence", f"{conf_txt}%"],
        ["Department", dept],
        ["Clinical Status", "Processed & Verified"]
    ]
    for row in data:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(50, 8, row[0], border='B')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 8, row[1], border='B', ln=True)
    
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(15, 65, 105)
    pdf.cell(95, 10, "Original Scan Source", align='C')
    pdf.cell(95, 10, "Neural Attention Map", align='C', ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
        
        img1_res = img1.resize((800, 800))
        img1_res.save(tmp1.name, format="JPEG")
        
        if isinstance(img2, np.ndarray):
            img2_pil = Image.fromarray(img2).resize((800, 800))
            img2_pil.save(tmp2.name, format="JPEG")
        else:
            img2_res = img2.resize((800, 800))
            img2_res.save(tmp2.name, format="JPEG")

        pdf.set_draw_color(0, 160, 210)
        pdf.set_line_width(1)
        pdf.rect(15, 85, 80, 80)
        pdf.rect(115, 85, 80, 80)
        
        pdf.image(tmp1.name, x=15, y=85, w=80, h=80)
        pdf.image(tmp2.name, x=115, y=85, w=80, h=80)

    pdf.set_y(175)
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(15, 65, 105)
    pdf.cell(0, 10, " DETAILED CLINICAL ANALYSIS (AI ASSISTED)", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    clean_text = gemini_text.replace('**', '').replace('*', '-')
    clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, txt=clean_text)

    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "This report is generated by Smart Health Care Pro AI. Valid for clinical triage purposes.", ln=True, align='C')
    pdf.cell(0, 5, "St. Elizabeth Hospital | 3506 Union Street, Seattle, WA 98161", ln=True, align='C')

    return pdf.output(dest='S').encode('latin-1')


# --- STARTUP / LANDING PAGE ---
def show_landing_page():
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="stBottom"] {display: none !important;}
        footer {visibility: hidden;}
        .block-container {padding-top: 0 !important; max-width: 100% !important;}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        body { margin: 0; overflow: hidden; background: #020617; font-family: 'Orbitron', sans-serif; }
        #canvas-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }

        #loader {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: #020617; z-index: 100;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            transition: opacity 1s ease-out;
        }
        .loader-hidden { opacity: 0; pointer-events: none; }

        .pulse-ring {
            width: 90px; height: 90px; border-radius: 50%;
            border: 3px solid #00C8D2;
            box-shadow: 0 0 30px rgba(0,200,210,0.4), inset 0 0 20px rgba(0,200,210,0.2);
            animation: pulse 2s infinite ease-in-out;
            display: flex; align-items: center; justify-content: center;
            margin-bottom: 35px;
        }
        .pulse-ring::after {
            content: '✚'; color: #00C8D2; font-size: 32px; font-weight: bold;
        }
        @keyframes pulse {
            0% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(0,200,210,0.7); }
            50% { transform: scale(1); box-shadow: 0 0 0 25px rgba(0,200,210,0); }
            100% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(0,200,210,0); }
        }

        .loader-text {
            color: #00C8D2; font-size: 13px; letter-spacing: 4px;
            margin-bottom: 25px; text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
            animation: flicker 3s infinite;
        }
        @keyframes flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .progress-container {
            width: 280px; height: 2px; background: rgba(0,200,210,0.15);
            border-radius: 2px; overflow: hidden; position: relative;
        }
        .progress-bar {
            width: 0%; height: 100%; background: #00C8D2;
            box-shadow: 0 0 15px #00C8D2, 0 0 30px rgba(0,200,210,0.3);
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .progress-text {
            color: #475569; font-size: 11px; margin-top: 12px;
            font-family: 'Inter', sans-serif; letter-spacing: 1px;
        }

        .ecg-container {
            position: absolute; bottom: 80px; left: 50%; transform: translateX(-50%);
            width: 360px; height: 50px; opacity: 0.25;
        }
        svg.ecg-line path {
            stroke: #00C8D2; stroke-width: 1.5; fill: none;
            stroke-dasharray: 1200; stroke-dashoffset: 1200;
            animation: drawECG 2.5s linear infinite;
        }
        @keyframes drawECG {
            to { stroke-dashoffset: 0; }
        }

        #content {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            z-index: 10; pointer-events: none;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        }
        .content-visible { pointer-events: auto; }

        .hero-title {
            font-size: 3.8rem; color: #00C8D2; 
            text-shadow: 0 0 40px rgba(0,200,210,0.6), 0 0 80px rgba(0,200,210,0.2);
            margin-bottom: 12px; letter-spacing: 6px;
            font-family: 'Orbitron', sans-serif; font-weight: 700;
            opacity: 0; transform: translateY(25px);
            transition: all 1.2s cubic-bezier(0.22, 1, 0.36, 1) 0.3s;
        }
        .hero-subtitle {
            font-size: 1.05rem; color: #94A3B8; letter-spacing: 3px;
            margin-bottom: 35px; font-family: 'Inter', sans-serif; font-weight: 300;
            opacity: 0; transform: translateY(25px);
            transition: all 1.2s cubic-bezier(0.22, 1, 0.36, 1) 0.5s;
        }
        .hero-badge {
            background: rgba(0,200,210,0.08); border: 1px solid rgba(0,200,210,0.25);
            color: #00C8D2; padding: 10px 28px; border-radius: 25px;
            font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
            opacity: 0; transform: translateY(25px);
            transition: all 1.2s cubic-bezier(0.22, 1, 0.36, 1) 0.7s;
            box-shadow: 0 0 20px rgba(0,200,210,0.1);
        }
        .show-content .hero-title,
        .show-content .hero-subtitle,
        .show-content .hero-badge {
            opacity: 1; transform: translateY(0);
        }

        @media (max-width: 768px) {
            .hero-title { font-size: 2.2rem; letter-spacing: 3px; }
            .hero-subtitle { font-size: 0.85rem; }
            .ecg-container { width: 250px; }
        }
    </style>
    </head>
    <body>

    <div id="loader">
        <div class="pulse-ring"></div>
        <div class="loader-text">Initializing Neural Networks</div>
        <div class="progress-container"><div class="progress-bar" id="progressBar"></div></div>
        <div class="progress-text" id="progressText">0%</div>
        <div class="ecg-container">
            <svg class="ecg-line" viewBox="0 0 400 60" preserveAspectRatio="none">
                <path d="M0,30 L30,30 L40,30 L50,10 L60,50 L70,30 L80,30 L110,30 L120,30 L130,10 L140,50 L150,30 L180,30 L190,30 L200,10 L210,50 L220,30 L250,30 L260,30 L270,10 L280,50 L290,30 L320,30 L330,30 L340,10 L350,50 L360,30 L400,30" />
            </svg>
        </div>
    </div>

    <div id="canvas-container"></div>

    <div id="content">
        <div class="hero-title">SMART HEALTH CARE</div>
        <div class="hero-subtitle">Next-Generation Clinical Command Center</div>
        <div class="hero-badge">Pro AI Diagnostics v2.0</div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    // --- LOADING SEQUENCE ---
    let progress = 0;
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const loader = document.getElementById('loader');
    const content = document.getElementById('content');

    function updateProgress() {
        const increment = 5 + Math.random() * 20;
        progress = Math.min(progress + increment, 100);
        progressBar.style.width = progress + '%';
        progressText.innerText = Math.floor(progress) + '%';
        if(progress < 100) {
            setTimeout(updateProgress, 150 + Math.random() * 250);
        } else {
            setTimeout(() => {
                loader.classList.add('loader-hidden');
                content.classList.add('content-visible');
                content.classList.add('show-content');
            }, 600);
        }
    }
    setTimeout(updateProgress, 400);

    // --- THREE.JS 3D SCENE ---
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x020617, 0.04);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 9);

    const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x00C8D2, 0.35);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x00C8D2, 1.2, 60);
    pointLight1.position.set(6, 6, 6);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x66E5FF, 0.6, 60);
    pointLight2.position.set(-6, -4, 6);
    scene.add(pointLight2);

    const pointLight3 = new THREE.PointLight(0x004444, 0.8, 40);
    pointLight3.position.set(0, -6, 4);
    scene.add(pointLight3);

    // --- DNA DOUBLE HELIX ---
    const helixGroup = new THREE.Group();
    const particleCount = 400;
    const strand1Geo = new THREE.BufferGeometry();
    const strand2Geo = new THREE.BufferGeometry();
    const s1Pos = new Float32Array(particleCount * 3);
    const s2Pos = new Float32Array(particleCount * 3);
    const s1Col = new Float32Array(particleCount * 3);
    const s2Col = new Float32Array(particleCount * 3);

    for(let i = 0; i < particleCount; i++) {
        const t = (i / particleCount) * Math.PI * 8;
        const y = (i / particleCount) * 12 - 6;
        const r = 2.0;

        s1Pos[i*3] = Math.cos(t) * r;
        s1Pos[i*3+1] = y;
        s1Pos[i*3+2] = Math.sin(t) * r;

        const brightness1 = 0.7 + Math.random() * 0.3;
        s1Col[i*3] = 0.0;
        s1Col[i*3+1] = 0.78 * brightness1;
        s1Col[i*3+2] = 0.82 * brightness1;

        s2Pos[i*3] = Math.cos(t + Math.PI) * r;
        s2Pos[i*3+1] = y;
        s2Pos[i*3+2] = Math.sin(t + Math.PI) * r;

        const brightness2 = 0.6 + Math.random() * 0.4;
        s2Col[i*3] = 0.1 * brightness2;
        s2Col[i*3+1] = 0.85 * brightness2;
        s2Col[i*3+2] = 1.0 * brightness2;
    }

    strand1Geo.setAttribute('position', new THREE.BufferAttribute(s1Pos, 3));
    strand1Geo.setAttribute('color', new THREE.BufferAttribute(s1Col, 3));
    strand2Geo.setAttribute('position', new THREE.BufferAttribute(s2Pos, 3));
    strand2Geo.setAttribute('color', new THREE.BufferAttribute(s2Col, 3));

    const particleMat = new THREE.PointsMaterial({
        size: 0.07,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
        depthWrite: false
    });

    const strand1 = new THREE.Points(strand1Geo, particleMat);
    const strand2 = new THREE.Points(strand2Geo, particleMat.clone());
    helixGroup.add(strand1);
    helixGroup.add(strand2);

    // DNA rungs
    const lineGeo = new THREE.BufferGeometry();
    const linePos = [];
    for(let i = 0; i < particleCount; i+=2) {
        linePos.push(s1Pos[i*3], s1Pos[i*3+1], s1Pos[i*3+2]);
        linePos.push(s2Pos[i*3], s2Pos[i*3+1], s2Pos[i*3+2]);
    }
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
    const lineMat = new THREE.LineBasicMaterial({
        color: 0x00C8D2, 
        transparent: true, 
        opacity: 0.12,
        blending: THREE.AdditiveBlending
    });
    const rungs = new THREE.LineSegments(lineGeo, lineMat);
    helixGroup.add(rungs);
    scene.add(helixGroup);

    // --- FLOATING MEDICAL CROSSES ---
    const crossGroup = new THREE.Group();
    const crossMat = new THREE.MeshBasicMaterial({
        color: 0x00C8D2, 
        transparent: true, 
        opacity: 0.2,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    for(let i = 0; i < 40; i++) {
        const cGroup = new THREE.Group();
        const vBar = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.35, 0.05), crossMat);
        const hBar = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.05, 0.05), crossMat);
        cGroup.add(vBar);
        cGroup.add(hBar);

        cGroup.position.set(
            (Math.random()-0.5) * 18,
            (Math.random()-0.5) * 18,
            (Math.random()-0.5) * 12 - 3
        );
        cGroup.rotation.set(Math.random()*Math.PI, Math.random()*Math.PI, Math.random()*Math.PI);
        cGroup.userData = {
            rotSpeed: {x: (Math.random()-0.5)*0.015, y: (Math.random()-0.5)*0.015, z: (Math.random()-0.5)*0.015},
            floatSpeed: 0.004 + Math.random()*0.008,
            floatOffset: Math.random() * Math.PI * 2,
            originalY: cGroup.position.y
        };
        crossGroup.add(cGroup);
    }
    scene.add(crossGroup);

    // --- CENTRAL CORE ---
    const coreGeo = new THREE.IcosahedronGeometry(0.35, 3);
    const coreMat = new THREE.MeshPhongMaterial({
        color: 0x00C8D2,
        emissive: 0x003333,
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0.75,
        shininess: 120,
        flatShading: false,
        blending: THREE.AdditiveBlending
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    scene.add(core);

    // Wireframe overlay
    const wireGeo = new THREE.IcosahedronGeometry(0.5, 2);
    const wireMat = new THREE.MeshBasicMaterial({
        color: 0x00C8D2, wireframe: true, transparent: true, opacity: 0.08,
        blending: THREE.AdditiveBlending
    });
    const wireMesh = new THREE.Mesh(wireGeo, wireMat);
    scene.add(wireMesh);

    // Outer rings
    const ringGeo = new THREE.TorusGeometry(0.9, 0.015, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({
        color: 0x00C8D2, transparent: true, opacity: 0.25,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide
    });
    const ring1 = new THREE.Mesh(ringGeo, ringMat);
    const ring2 = new THREE.Mesh(ringGeo, ringMat.clone());
    const ring3 = new THREE.Mesh(ringGeo, ringMat.clone());
    ring2.rotation.x = Math.PI/2.5;
    ring3.rotation.y = Math.PI/2.5;
    scene.add(ring1);
    scene.add(ring2);
    scene.add(ring3);

    // --- ANIMATION ---
    let mouseX = 0, mouseY = 0;
    let targetMouseX = 0, targetMouseY = 0;

    document.addEventListener('mousemove', (e) => {
        targetMouseX = (e.clientX / window.innerWidth) * 2 - 1;
        targetMouseY = (e.clientY / window.innerHeight) * 2 - 1;
    });

    function animate() {
        requestAnimationFrame(animate);
        const time = Date.now() * 0.001;

        mouseX += (targetMouseX - mouseX) * 0.05;
        mouseY += (targetMouseY - mouseY) * 0.05;

        helixGroup.rotation.y += 0.0025;
        helixGroup.rotation.x = Math.sin(time * 0.4) * 0.08;
        helixGroup.position.y = Math.sin(time * 0.6) * 0.2;

        crossGroup.children.forEach(cross => {
            cross.rotation.x += cross.userData.rotSpeed.x;
            cross.rotation.y += cross.userData.rotSpeed.y;
            cross.rotation.z += cross.userData.rotSpeed.z;
            cross.position.y = cross.userData.originalY + Math.sin(time * 0.8 + cross.userData.floatOffset) * 0.4;
        });
        crossGroup.rotation.y -= 0.0008;

        const pulse = 1 + Math.sin(time * 1.8) * 0.12;
        core.scale.set(pulse, pulse, pulse);
        core.rotation.y += 0.008;
        core.rotation.x = Math.sin(time * 0.3) * 0.1;

        wireMesh.rotation.y -= 0.005;
        wireMesh.rotation.x += 0.003;
        wireMesh.scale.set(1/pulse, 1/pulse, 1/pulse);

        ring1.rotation.x += 0.004;
        ring1.rotation.y += 0.003;
        ring2.rotation.x += 0.003;
        ring2.rotation.z += 0.004;
        ring3.rotation.y += 0.005;
        ring3.rotation.z += 0.002;

        camera.position.x += (mouseX * 0.8 - camera.position.x) * 0.04;
        camera.position.y += (-mouseY * 0.6 - camera.position.y) * 0.04;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    </script>
    </body>
    </html>
    """

    components.html(html_code, height=680, scrolling=False)

    # Streamlit native button below the 3D scene
    st.markdown("<div style='margin-top: -80px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 ENTER COMMAND CENTER", use_container_width=True, type="primary"):
            st.session_state.show_dashboard = True
            st.rerun()

# --- DASHBOARD GUARD ---
if not st.session_state.show_dashboard:
    show_landing_page()
    st.stop()


# --- SCAN HISTORY DISPLAY COMPONENT ---
def save_scan_to_history(name, label, conf, filename, img=None):
    """Persist scan result to dashboard history and sync to disk."""
    entry = {
        "id": len(st.session_state.scan_history) + 1,
        "name": name,
        "label": label,
        "conf": round(float(conf), 2),
        "sev": SEVERITY_MAP[label],
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename
    }
    st.session_state.scan_history.insert(0, entry)
    st.session_state.scan_history = st.session_state.scan_history[:100]

    # Sync all counters to disk
    sync_session_to_disk()

def render_scan_history(limit=20):
    """Render the scan history table/cards for the dashboard."""
    history = st.session_state.scan_history[:limit]

    if not history:
        st.info("No scans in history yet. Upload scans in Single Diagnostic or Batch Triage.")
        return

    st.markdown('<h4 style="color:#00C8D2;font-family:Orbitron;margin-top:10px;">📋 FULL SCAN HISTORY</h4>', unsafe_allow_html=True)

    # Table header
    st.markdown("""
    <div style="display:grid;grid-template-columns:0.5fr 2fr 1.2fr 1fr 1.5fr;background:rgba(0,200,210,0.1);border-radius:8px;padding:10px 15px;margin-bottom:8px;font-family:Orbitron;font-size:0.8rem;color:#00C8D2;">
        <div>#</div>
        <div>Filename</div>
        <div>Diagnosis</div>
        <div>Confidence</div>
        <div>Time</div>
    </div>
    """, unsafe_allow_html=True)

    for item in history:
        color = {"High Risk": "#F43F5E", "Medium Risk": "#FB923C", "Normal": "#00C8D2"}.get(item["label"], "#00C8D2")
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:0.5fr 2fr 1.2fr 1fr 1.5fr;background:rgba(15,23,42,0.6);border-left:3px solid {color};border-radius:0 8px 8px 0;padding:10px 15px;margin-bottom:6px;font-size:0.82rem;color:#E2E8F0;align-items:center;">
            <div style="color:#475569;">{item['id']}</div>
            <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{item['filename'][:35]}</div>
            <div style="color:{color};font-weight:600;">{item['label']}</div>
            <div>{item['conf']}%</div>
            <div style="color:#64748B;font-size:0.75rem;">{item['time']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- 3. UI LAYOUT ---
apply_global_luxury_theme()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.markdown('<h3 style="color:#00C8D2; font-family:Orbitron;">🧭 NAVIGATION</h3>', unsafe_allow_html=True)
    page = st.radio("Select View", ["🏠 Dashboard", "🔍 Single Diagnostic", "🫀 Vitals (rPPG)", "🚑 Batch Triage", "🤖 Clinical AI Agent", "💬 Feedback"])
    st.divider()
    privacy_mode = st.toggle("HIPAA Shield", value=False)
    layer_choice = st.select_slider("Neural Depth", options=["Early", "Middle", "Mid-Deep", "Deep"], value="Deep")
    st.divider()
    # API Key is now hardcoded at the top

st.markdown('<div class="hero-banner"><p style="color:#E2E8F0;">Next-Generation Clinical Command Center</p><p class="hero-title">Smart Health Care</p></div>', unsafe_allow_html=True)

if page == "🏠 Dashboard":
    st.markdown('<h2 class="neon-blue-header">📊 Live Analytics</h2>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    sh_status, sh_color = ("Encrypted", "#00C8D2") if privacy_mode else ("Bypass", "#F43F5E")

    with c1: st.markdown(f'<div class="glass-card"><p>HIPAA Shield</p><h2 style="color:{sh_color};">{sh_status}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="glass-card"><p>Total Scans Done</p><h2 style="color:#00C8D2;">{st.session_state.total_scans}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="glass-card"><p>High Risk Cases</p><h2 style="color:#F43F5E;">{st.session_state.high_risk_count}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="glass-card"><p>AI Status</p><h2 style="color:#00C8D2;">Ready</h2></div>', unsafe_allow_html=True)

    # Community Feedback Summary
    feedback_list = load_feedback_data()
    if feedback_list:
        avg = sum(f["overall_rating"] for f in feedback_list) / len(feedback_list)
        stars = "★" * int(avg) + "☆" * (5 - int(avg))
        st.markdown(f'''
        <div style="background:rgba(0,200,210,0.06);border:1px solid rgba(0,200,210,0.25);border-radius:14px;padding:14px 24px;margin:18px 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div style="color:#00C8D2;font-family:Orbitron;font-size:0.95rem;letter-spacing:1px;">⭐ Community Rating</div>
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="color:#FBBF24;font-size:1.4rem;letter-spacing:4px;">{stars}</span>
                <span style="color:#E2E8F0;font-size:1rem;font-weight:600;">{avg:.1f}/5</span>
                <span style="color:#64748B;font-size:0.8rem;">({len(feedback_list)} review{"s" if len(feedback_list) != 1 else ""})</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<h3 class="neon-blue-header">📡 Last Scan Monitor</h3>', unsafe_allow_html=True)
        if st.session_state.last_scan_result:
            ls = st.session_state.last_scan_result
            sev_color = {"High Risk": "#F43F5E", "Medium Risk": "#FB923C", "Normal": "#00C8D2"}.get(ls["label"], "#00C8D2")
            st.markdown(f'''
            <div class="glass-card" style="border-left:4px solid {sev_color};">
                <p style="color:#94A3B8;font-size:0.85rem;">{ls["time"]}</p>
                <p style="color:#94A3B8;font-size:0.8rem;overflow:hidden;text-overflow:ellipsis;">{ls["name"]}</p>
                <h3 style="color:{sev_color};font-family:Orbitron;margin:8px 0;">{ls["label"]}</h3>
                <p style="color:#00C8D2;font-size:1.1rem;">Score: <b>{ls["conf"]}%</b></p>
            </div>
            ''', unsafe_allow_html=True)
        else: st.info("No scans processed.")

    with col2:
        st.markdown('<h3 class="neon-blue-header">🚑 Emergency Queue (Global)</h3>', unsafe_allow_html=True)
        if st.session_state.global_emergency_queue:
            sorted_q = sorted(st.session_state.global_emergency_queue, key=lambda x: x['sev'], reverse=True)
            for i, item in enumerate(sorted_q[:8]):
                sev_color = {"High Risk": "#F43F5E", "Medium Risk": "#FB923C", "Normal": "#00C8D2"}.get(item['label'], "#00C8D2")

                # Card with done button
                card_col, btn_col = st.columns([4, 1])
                with card_col:
                    st.markdown(f'''
                    <div style="background:rgba(15,23,42,0.85);border:1px solid {sev_color};border-radius:15px;padding:12px 15px;margin-bottom:10px;">
                        <div style="color:{sev_color};font-family:Orbitron;font-weight:bold;font-size:0.95rem;">{item["label"]}</div>
                        <div style="color:#94A3B8;font-size:0.75rem;margin-top:2px;">{item["name"]} · {item["conf"]}%</div>
                        <div style="color:#475569;font-size:0.7rem;">{item.get("time", "")}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with btn_col:
                    if st.button("✓ Done", key=f"done_{i}_{item['name'][:20]}"):
                        st.session_state.global_emergency_queue.pop(i)
                        sync_session_to_disk()
                        st.rerun()
        else: st.info("Queue is empty.")

    st.divider()
    st.markdown('<h2 class="neon-blue-header">📚 Scan History & Archives</h2>', unsafe_allow_html=True)
    render_scan_history(limit=30)
elif page == "🔍 Single Diagnostic":
    file = st.sidebar.file_uploader("Upload Scan", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        if privacy_mode: img, _ = anonymize_image(img)
        model = load_model()

        heatmap, idx, conf = get_layered_gradcam(model, img, layer_choice)
        label = CLASS_NAMES[idx]
        conf_txt = f"{conf*100:.2f}"
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        img_np = np.array(img.resize((224, 224)))
        h_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB), 0.4, 0)

        st.session_state.total_scans += 1
        data = {"name": file.name, "label": label, "conf": conf_txt, "sev": SEVERITY_MAP[label], "time": now_ts}
        st.session_state.last_scan_result = data

        # Add to emergency queue for all non-Normal scans
        if label != "Normal":
            st.session_state.global_emergency_queue.append(data)

        if label == "High Risk":
            st.session_state.high_risk_count += 1
            st.session_state.high_risk_details.append(data)

        # Save to persistent history and disk
        save_scan_to_history(file.name, label, conf_txt, file.name)

        st.markdown(f"""
            <div class="glass-card">
                <p class="diag-hero">{label}</p>
                <p style="text-align:center; font-size:1.5rem; color:#00C8D2;">
                    AI CONFIDENCE SCORE: <b>{conf_txt}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<h3 class="neon-blue-header">🔬 Clinical Visualization</h3>', unsafe_allow_html=True)
        col_img, col_heat = st.columns(2)
        with col_img:
            st.image(img, caption="Original Scan Source", use_container_width=True)
        with col_heat:
            area, diameter, bbox = measure_lesion_area(heatmap)
            if bbox and label != "Normal":
                bx, by, bw, bh = bbox
                cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (0, 200, 210), 2)
                cv2.putText(overlay, f"{diameter}mm", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 210), 2)
            st.image(overlay, caption=f"Neural Attention Heatmap ({layer_choice})", use_container_width=True)

        if area > 0 and label != "Normal":
            st.markdown('<h3 class="neon-blue-header">📐 Automated Lesion Measurement</h3>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated Area", f"{area} mm²", "Abnormal")
            m2.metric("Max Diameter", f"{diameter} mm", "Axis")
            m3.metric("Status", "Geometric Lock", delta_color="off")

        st.markdown('<h3 class="neon-blue-header">📊 3D Spectral & Volumetric Analysis</h3>', unsafe_allow_html=True)
        with st.expander("🚀 View Advanced Interactive 3D Diagnostics", expanded=True):
            img_gray = img.convert('L').resize((120, 120))
            z_data = np.array(img_gray)
            fig_surface = go.Figure(data=[go.Surface(z=z_data, colorscale='Viridis')])
            fig_surface.update_layout(scene=dict(zaxis_title="Intensity", xaxis_title="W", yaxis_title="H"), height=600, margin=dict(l=0, r=0, b=0, t=0))
            fig_violin = go.Figure()
            fig_violin.add_trace(go.Violin(y=z_data.flatten(), name=label, box_visible=True, meanline_visible=True, fillcolor='#00C8D2', line_color='white'))
            fig_violin.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            v_col1, v_col2 = st.columns([2, 1])
            with v_col1: st.plotly_chart(fig_surface, use_container_width=True)
            with v_col2: st.plotly_chart(fig_violin, use_container_width=True)

        report_state_key = f"pdf_bytes_{file.name}"
        if st.button("🧠 Generate AI Clinical PDF "):
            with st.spinner("Gemini AI is analyzing the scan and generating the report..."):
                gemini_analysis = call_gemini_openrouter(openrouter_key, label, img)
                st.session_state[report_state_key] = create_advanced_pdf(img, overlay, label, file.name, conf_txt, gemini_analysis)

        if report_state_key in st.session_state:
            st.download_button("📥 Download Generated PDF", st.session_state[report_state_key], f"Detailed_Report_{file.name}.pdf", mime="application/pdf")
elif page == "🚑 Batch Triage":
    files = st.sidebar.file_uploader("Upload Batch", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if files:
        import plotly.express as px
        model = load_model()
        batch_data = []
        for f in files:
            img = Image.open(f).convert('RGB')
            if privacy_mode: img, _ = anonymize_image(img)
            heatmap, idx, conf = get_layered_gradcam(model, img, layer_choice)
            label = CLASS_NAMES[idx]
            conf_txt = f"{conf*100:.2f}"
            now_ts = datetime.datetime.now().strftime("%H:%M:%S")
            img_np = np.array(img.resize((224, 224)))
            h_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB), 0.4, 0)
            pixel_intensity = np.array(img.convert('L').resize((40,40))).flatten().tolist()
            scan_info = {"name": f.name, "label": label, "conf": float(conf_txt), "sev": SEVERITY_MAP[label], "orig": img, "heatmap": overlay, "time": now_ts, "pixels": pixel_intensity}
            st.session_state.total_scans += 1
            batch_data.append(scan_info)
        
        # Add non-Normal scans to emergency queue (strip image objects for JSON)
        for scan_info in batch_data:
            if scan_info["label"] != "Normal":
                queue_entry = {
                    "name": scan_info["name"],
                    "label": scan_info["label"],
                    "conf": scan_info["conf"],
                    "sev": scan_info["sev"],
                    "time": scan_info["time"]
                }
                st.session_state.global_emergency_queue.append(queue_entry)

        # Save all batch scans to persistent history and disk
        for scan_info in batch_data:
            save_scan_to_history(scan_info["name"], scan_info["label"], scan_info["conf"], scan_info["name"])
        st.markdown('<h2 class="neon-blue-header">📋 Triage Results</h2>', unsafe_allow_html=True)
        sorted_batch = sorted(batch_data, key=lambda x: (x['sev'], x['conf']), reverse=True)
        for r in sorted_batch:
            b_style = "row-high" if r['sev'] == 2 else "row-med" if r['sev'] == 1 else "row-norm"
            st.markdown(f'<div class="triage-row {b_style}"><div>{r["label"]} | {r["name"]}</div><div><b>{r["conf"]}%</b></div></div>', unsafe_allow_html=True)

        st.markdown('<h2 class="neon-blue-header">📊 Batch Analytics & Comparison</h2>', unsafe_allow_html=True)
        all_labels = [r['label'] for r in batch_data]
        all_confs = [r['conf'] for r in batch_data]
        all_names = [r['name'] for r in batch_data]
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            fig_bar = px.bar(x=all_labels, y=all_confs, color=all_labels, title="Confidence by Category", color_discrete_map={"High Risk": "#F43F5E", "Medium Risk": "#FB923C", "Normal": "#00C8D2"})
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_v_comp = go.Figure()
            for r in batch_data: fig_v_comp.add_trace(go.Violin(y=r['pixels'], name=r['name'][:8], box_visible=True, meanline_visible=True))
            fig_v_comp.update_layout(title="Volumetric Density Comparison", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig_v_comp, use_container_width=True)
        with col_meta2:
            fig_pie = px.pie(names=all_labels, title="Batch Risk Ratio", color=all_labels, color_discrete_map={"High Risk": "#F43F5E", "Medium Risk": "#FB923C", "Normal": "#00C8D2"}, hole=0.4)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig_pie, use_container_width=True)
            fig_line = px.line(x=all_names, y=all_confs, title="Batch Confidence Trend", markers=True)
            fig_line.update_traces(line_color='#00C8D2')
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()
        st.markdown('<h2 class="neon-blue-header">🧪 Visual Evidences</h2>', unsafe_allow_html=True)
        for r in sorted_batch:
            with st.container():
                st.markdown(f'<div class="evidence-card"><h4 style="color:#00C8D2;">CASE ID: {r["name"]}</h4><p>STATUS: <b>{r["label"]}</b> | CONFIDENCE: <b>{r["conf"]}%</b></p></div>', unsafe_allow_html=True)
                ec1, ec2 = st.columns(2)
                ec1.image(r['orig'], use_container_width=True)
                ec2.image(r['heatmap'], use_container_width=True)
                
                report_state_key = f"pdf_bytes_{r['name']}"
                if st.button(f"🧠 Generate AI Report: {r['name']}", key=f"gen_{r['name']}"):
                    with st.spinner(f"Gemini AI is analyzing {r['name']}..."):
                        gemini_analysis = call_gemini_openrouter(openrouter_key, r['label'], r['orig'])
                        st.session_state[report_state_key] = create_advanced_pdf(r['orig'], r['heatmap'], r['label'], r['name'], str(r['conf']), gemini_analysis)
                
                if report_state_key in st.session_state:
                    st.download_button(f"📥 Download {r['name']} PDF", st.session_state[report_state_key], f"Detailed_Report_{r['name']}.pdf", key=f"dl_btn_{r['name']}", mime="application/pdf")
# --- NEW SECTION: CLINICAL AI AGENT ---
elif page == "🫀 Vitals (rPPG)":
    st.markdown(
        "<div class='glass'><h3 style='margin:0 0 6px;color:#E6F7FA'>Contactless Vitals · rPPG + Respiratory</h3>"
        "<p style='opacity:0.75;margin:0'>Upload a short face video (5–15 s, well-lit, minimal motion). Heart rate is recovered from green-channel skin variations using FFT; respiration rate is recovered from facial motion bandpass.</p></div>",
        unsafe_allow_html=True,
    )

    video = st.file_uploader(
        "Upload face video (mp4 / mov / webm)",
        type=["mp4", "mov", "webm", "avi"], key="vitals_upload",
    )

    if not video:
        st.info("Awaiting video upload.")
        st.stop()

    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as tmp:
        tmp.write(video.getvalue())
        path = tmp.name

    with st.spinner("Analysing facial pulse signal (FFT)…"):
        try:
            vitals = rppg.analyse_video(path)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    c1, c2, c3, c4 = st.columns(4)
    hr_sub = "FFT peak in 0.7–3.5 Hz band" if vitals.heart_rate_bpm > 0 else "insufficient signal"
    c1.metric("Heart Rate", f"{vitals.heart_rate_bpm:.0f} BPM", hr_sub)
    c2.metric("Respiration", f"{vitals.respiration_rate_bpm:.0f} breaths/min")
    c3.metric("Signal Quality", f"{vitals.signal_quality*100:.0f}%")
    c4.metric("Samples", f"{vitals.samples} @ {vitals.fps:.1f} fps")

    if vitals.filtered_signal:
        st.subheader("Bandpass-filtered rPPG signal")
        t = np.arange(len(vitals.filtered_signal)) / vitals.fps
        sig_fig = go.Figure()
        sig_fig.add_trace(go.Scatter(x=t, y=vitals.filtered_signal, mode="lines"))
        st.plotly_chart(sig_fig, use_container_width=True)

        st.subheader("FFT spectrum")
        spec_fig = go.Figure()
        freqs_bpm = [f * 60 for f in vitals.frequencies]
        mask = [(f >= 30 and f <= 220) for f in freqs_bpm]
        f_x = [f for f, m in zip(freqs_bpm, mask) if m]
        f_y = [s for s, m in zip(vitals.spectrum, mask) if m]
        spec_fig.add_trace(go.Scatter(x=f_x, y=f_y, mode="lines"))
        st.plotly_chart(spec_fig, use_container_width=True)
    else:
        st.warning("Could not extract enough usable frames. Try a longer, well-lit clip.")

elif page == "🤖 Clinical AI Agent":
    st.markdown('<h2 class="neon-blue-header">🤖 Clinical AI Intelligence</h2>', unsafe_allow_html=True)
    
    # Chat UI Container
    chat_container = st.container()
    
    # ✅ KEEP EVERYTHING INSIDE THIS BLOCK
    with chat_container:
        for message in st.session_state.chat_history:
            role_class = "glass-card" if message["role"] == "assistant" else ""
            color = "#00C8D2" if message["role"] == "assistant" else "#E2E8F0"
            role_label = 'AI AGENT' if message['role'] == 'assistant' else 'DOCTOR'
            
            content = message['content'].strip()
            
            st.markdown(
                f'<div class="{role_class}" style="padding:15px; margin-bottom:10px; border-left:5px solid {color}; color:{color};">'
                f'<b>{role_label}:</b><br>{content}</div>',
                unsafe_allow_html=True
            )

        # Input Area
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        chat_file = st.file_uploader("📎 Attach Scan for Analysis", type=["jpg", "png", "jpeg"], key="chat_upload")
        user_input = st.chat_input("Ask about scans, symptoms, or medical research...")
        st.markdown('</div>', unsafe_allow_html=True)

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Analyzing Clinical Context..."):
                if chat_file:
                    chat_img = Image.open(chat_file).convert('RGB')
                    response = call_gemini_openrouter(openrouter_key, "Scan Uploaded by Dr.", chat_img)
                else:
                    headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
                    payload = {
                        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                        "messages": [{"role": "user", "content": user_input}]
                    }
                    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                    response = res.json()['choices'][0]['message']['content'] if res.status_code == 200 else "Error"

                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

# --- NEW SECTION: FEEDBACK & REVIEWS ---
elif page == "💬 Feedback":
    st.markdown('<h2 class="neon-blue-header">💬 Feedback & Reviews</h2>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📝 Submit Feedback", "📊 View All Feedback"])

    with tab1:
        st.markdown("""
        <style>
        div[data-testid="stForm"] button[kind="primary"] {
            background: linear-gradient(135deg, #00C8D2, #0891b2) !important;
            border: none !important;
            color: #020617 !important;
            font-family: 'Orbitron', sans-serif !important;
            letter-spacing: 1.5px !important;
            font-weight: 700 !important;
            box-shadow: 0 0 15px rgba(0,200,210,0.3) !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stForm"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0891b2, #00C8D2) !important;
            box-shadow: 0 0 25px rgba(0,200,210,0.5) !important;
            transform: translateY(-1px) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#00C8D2;font-family:Orbitron;margin-bottom:5px;">Share Your Experience</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94A3B8;font-size:0.85rem;margin-bottom:20px;">Your feedback helps us improve clinical outcomes and user experience.</p>', unsafe_allow_html=True)

        with st.form("feedback_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                fb_name = st.text_input("Full Name *", placeholder="Dr. John Doe")
            with col2:
                fb_email = st.text_input("Email Address *", placeholder="doctor@hospital.com")

            st.markdown('<p style="color:#00C8D2;font-family:Orbitron;margin-top:18px;margin-bottom:8px;">⭐ Overall Experience</p>', unsafe_allow_html=True)
            fb_overall = st.select_slider(
                "Rate your overall experience",
                options=[1, 2, 3, 4, 5],
                value=4,
                format_func=lambda x: ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Very Good", "⭐⭐⭐⭐⭐ Excellent"][x-1]
            )

            st.markdown('<p style="color:#00C8D2;font-family:Orbitron;margin-top:18px;margin-bottom:8px;">📊 Detailed Ratings</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                fb_ui = st.slider("UI / UX Design", 1, 10, 8, help="Visual design, ease of navigation, responsiveness")
            with c2:
                fb_accuracy = st.slider("Diagnostic Accuracy", 1, 10, 8, help="Reliability of AI predictions and reports")
            with c3:
                fb_speed = st.slider("Speed / Performance", 1, 10, 8, help="Loading times, scan processing speed")

            fb_review = st.text_area(
                "Written Review",
                placeholder="Describe your experience using the app. How has it impacted your clinical workflow? What did you like or dislike?",
                height=110
            )
            fb_suggestions = st.text_area(
                "Suggestions for Future Implementation",
                placeholder="What new features, scan types, integrations, or improvements would you like to see in upcoming versions?",
                height=90
            )

            submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True, type="primary")
            if submitted:
                if not fb_name.strip() or not fb_email.strip():
                    st.error("⚠️ Please enter both your name and email before submitting.")
                elif "@" not in fb_email or "." not in fb_email:
                    st.error("⚠️ Please enter a valid email address.")
                elif len(fb_review.strip()) < 10:
                    st.error("⚠️ Please write at least a brief review (10 characters minimum) to help us understand your experience.")
                else:
                    success = submit_feedback(fb_name.strip(), fb_email.strip(), fb_overall, fb_ui, fb_accuracy, fb_speed, fb_review.strip(), fb_suggestions.strip())
                    if success:
                        st.markdown('''
                        <div style="background:rgba(0,200,210,0.08);border:1px solid #00C8D2;border-radius:16px;padding:22px 28px;margin-top:18px;text-align:center;box-shadow:0 0 20px rgba(0,200,210,0.12);">
                            <div style="font-size:2.2rem;margin-bottom:10px;">🙏</div>
                            <div style="color:#00C8D2;font-family:Orbitron;font-size:1.25rem;font-weight:700;margin-bottom:8px;letter-spacing:1px;">THANK YOU FOR YOUR FEEDBACK</div>
                            <div style="color:#94A3B8;font-size:0.9rem;line-height:1.5;">Your insights have been securely recorded and will directly shape the future of Smart Health Care Pro AI.<br>We truly appreciate you taking the time to help us improve clinical outcomes.</div>
                        </div>
                        ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        render_feedback_stats()
        st.divider()
        render_feedback_list()