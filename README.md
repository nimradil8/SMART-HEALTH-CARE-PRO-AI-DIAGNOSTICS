# SMART-HEALTH-CARE-PRO-AI-DIAGNOSTICS
🧠 Smart Health Care Pro AI

An AI-powered clinical diagnostics platform combining computer vision, generative AI, and biomedical signal processing — built as a semester project for the Department of Computer Science (2026).


👩‍💻 Developers

Nimra Dil&
Chashman Aslam


📋 Table of Contents

About
Features
Methodology
System Modules
Results
Technologies Used
How to Run
Future Scope


📖 About
Smart Health Care Pro AI is a production-ready clinical intelligence system that automates medical image diagnosis and patient triage. It integrates deep learning, contactless vital monitoring, and AI-generated clinical reports into a unified healthcare platform.

✨ Features

🔬 AI-powered medical image classification (Normal / Medium Risk / High Risk)
🔥 Grad-CAM heatmap visualization for clinical explainability
📄 Auto-generated clinical PDF reports using LLMs
❤️ Contactless heart rate & respiration monitoring (rPPG)
🛡️ HIPAA-compliant patient data anonymization via OCR
🚑 Batch triage with severity-based emergency queue
🤖 Conversational AI clinical assistant


⚙️ Methodology
ModuleDescription🧠 Deep LearningEfficientNetV2 for 3-class medical image classification🔥 Grad-CAMNeural attention maps at 4 depths for explainability📄 AI ReportsOpenRouter API (Gemini & Llama) for clinical PDF generation❤️ rPPG VitalsFFT bandpass filtering on facial video for vitals🛡️ HIPAA ShieldEasyOCR-based automatic patient data redaction🚑 Batch TriageMulti-image upload with severity sorting & analytics

🖥️ System Modules

Live Dashboard — Real-time analytics, emergency queue, scan history
Single Diagnostic — Upload scan → Classification → Heatmap → Lesion measurement
Vitals Monitor — Contactless heart rate & respiration via facial video
Clinical AI Agent — Image-aware conversational medical assistant


📈 Results & Performance
MetricValueRisk Classes3Inference Time< 2 secondsNeural Network Depths4Classification CategoriesNormal, Medium Risk, High Risk

🛠️ Technologies Used

Python
EfficientNetV2 (Deep Learning)
Grad-CAM
OpenRouter API (Gemini / Llama)
EasyOCR
rPPG Signal Processing
FFT Bandpass Filtering
Three.js (Presentation UI)


▶️ How to Run
Presentation
Simply open the file in any browser:
bashSmart_Health_Care_3D_Presentation.html
Use Arrow Keys or ← → buttons to navigate slides.

Press F for fullscreen.

🔮 Future Scope

🌐 Cloud Deployment — Kubernetes-based scaling for multi-hospital networks
📱 Mobile App — Flutter companion for bedside diagnostics
🔗 EHR Integration — HL7/FHIR connectivity to hospital record systems



Department of Computer Science · 2026
