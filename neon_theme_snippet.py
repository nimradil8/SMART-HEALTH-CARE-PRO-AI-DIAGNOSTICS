def apply_global_luxury_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

        /* ── GLOBAL BACKGROUND ── */
        .stApp {
            background: #020B18 !important;
            font-family: 'Rajdhani', sans-serif !important;
        }

        /* Animated grid */
        .stApp::before {
            content: '';
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image:
                linear-gradient(rgba(0,245,255,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,245,255,0.04) 1px, transparent 1px);
            background-size: 40px 40px;
            animation: gridPan 20s linear infinite;
        }
        @keyframes gridPan { from{background-position:0 0} to{background-position:40px 40px} }

        /* Radial ambient glows */
        .stApp::after {
            content: '';
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background:
                radial-gradient(ellipse 60% 40% at 20% 20%, rgba(0,128,255,0.10) 0%, transparent 70%),
                radial-gradient(ellipse 50% 35% at 80% 75%, rgba(0,245,255,0.07) 0%, transparent 70%),
                radial-gradient(ellipse 40% 30% at 55% 45%, rgba(0,255,159,0.05) 0%, transparent 70%);
        }

        /* Scanlines */
        body::after {
            content: '';
            position: fixed; inset: 0; z-index: 1; pointer-events: none;
            background: repeating-linear-gradient(
                0deg, transparent, transparent 3px,
                rgba(0,0,0,0.07) 3px, rgba(0,0,0,0.07) 4px
            );
        }

        /* ── SIDEBAR ── */
        section[data-testid="stSidebar"] {
            background: rgba(0,8,22,0.97) !important;
            border-right: 1px solid rgba(0,245,255,0.2) !important;
            backdrop-filter: blur(20px);
        }
        section[data-testid="stSidebar"] * { color: #C8E8FF !important; }
        section[data-testid="stSidebar"] .stRadio label {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important; font-size: 0.95rem !important;
        }

        /* ── HERO BANNER ── */
        .hero-banner {
            background: linear-gradient(90deg, rgba(0,20,50,0.97) 0%, rgba(0,10,30,0.95) 100%);
            border-bottom: 1px solid rgba(0,245,255,0.2);
            padding: 3rem 2.5rem 2.5rem;
            margin: -80px -100px 40px -100px;
            text-align: center;
            position: relative; overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute; top: -60px; right: -60px;
            width: 350px; height: 350px; border-radius: 50%;
            background: radial-gradient(circle, rgba(0,245,255,0.08) 0%, transparent 70%);
            animation: heroPulse 5s ease-in-out infinite;
        }
        @keyframes heroPulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.2);opacity:0.5}}

        .hero-eyebrow {
            font-family: 'Share Tech Mono', monospace !important;
            font-size: 0.68rem !important; color: #00F5FF !important;
            letter-spacing: 5px; margin-bottom: 10px;
            text-transform: uppercase;
        }
        .hero-title {
            font-family: 'Orbitron', monospace !important;
            font-size: 3rem !important; font-weight: 900 !important;
            color: #C8E8FF !important; letter-spacing: -1px !important;
        }
        .hero-title span { color: #00F5FF; text-shadow: 0 0 20px rgba(0,245,255,0.6), 0 0 60px rgba(0,245,255,0.2); }

        /* ── GLASS CARDS ── */
        .glass-card {
            background: rgba(0,20,50,0.7) !important;
            border: 1px solid rgba(0,245,255,0.2) !important;
            border-radius: 20px !important;
            padding: 28px !important;
            margin-bottom: 20px;
            position: relative;
            transition: all 0.3s;
        }
        .glass-card::after {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
            border-radius: 20px 20px 0 0;
            background: linear-gradient(90deg, transparent, rgba(0,245,255,0.5), transparent);
        }

        /* ── DIAGNOSTICS RESULT ── */
        .diag-hero {
            font-family: 'Orbitron', monospace !important;
            font-size: 2.8rem !important; font-weight: 900 !important;
            letter-spacing: 2px !important; display: inline-block;
            padding: 8px 32px; border-radius: 50px;
            animation: badgePulse 2s ease-in-out infinite;
        }
        @keyframes badgePulse{0%,100%{transform:scale(1)}50%{transform:scale(1.02)}}

        .t-high {
            color: #FF2D6B !important; border: 2px solid #FF2D6B;
            background: rgba(255,45,107,0.08);
            box-shadow: 0 0 30px rgba(255,45,107,0.3), inset 0 0 20px rgba(255,45,107,0.05);
            text-shadow: 0 0 20px rgba(255,45,107,0.8);
        }
        .t-med {
            color: #FFB800 !important; border: 2px solid #FFB800;
            background: rgba(255,184,0,0.08);
            box-shadow: 0 0 30px rgba(255,184,0,0.3), inset 0 0 20px rgba(255,184,0,0.05);
            text-shadow: 0 0 20px rgba(255,184,0,0.8);
        }
        .t-norm {
            color: #00FF9F !important; border: 2px solid #00FF9F;
            background: rgba(0,255,159,0.08);
            box-shadow: 0 0 30px rgba(0,255,159,0.3), inset 0 0 20px rgba(0,255,159,0.05);
            text-shadow: 0 0 20px rgba(0,255,159,0.8);
        }

        /* ── TRIAGE ROWS ── */
        .triage-row {
            background: rgba(0,15,35,0.8) !important;
            border: 1px solid rgba(0,245,255,0.15) !important;
            border-left: 4px solid !important;
            border-radius: 12px !important;
            padding: 16px 20px !important; margin-bottom: 12px;
            display: flex; justify-content: space-between; align-items: center;
            transition: all 0.25s;
        }
        .triage-row:hover { transform: translateX(4px); background: rgba(0,30,60,0.9) !important; }
        .row-high { border-left-color: #FF2D6B !important; }
        .row-med  { border-left-color: #FFB800 !important; }
        .row-norm { border-left-color: #00FF9F !important; }

        /* ── METRIC WIDGETS ── */
        .metric-val {
            font-family: 'Orbitron', monospace !important;
            font-size: 1.8rem !important; font-weight: 700 !important;
            color: #00F5FF !important;
            text-shadow: 0 0 16px rgba(0,245,255,0.6) !important;
        }
        .metric-label {
            font-family: 'Share Tech Mono', monospace !important;
            font-size: 0.65rem !important; color: #4A7A9B !important;
            letter-spacing: 2px !important; text-transform: uppercase;
        }

        /* ── STREAMLIT OVERRIDES ── */
        h1, h2, h3 { font-family: 'Orbitron', monospace !important; color: #C8E8FF !important; letter-spacing: 1px !important; }
        p, li, span { color: #C8E8FF !important; }

        .stButton > button {
            font-family: 'Orbitron', monospace !important;
            font-size: 0.72rem !important; font-weight: 700 !important;
            letter-spacing: 1.5px !important;
            background: rgba(0,245,255,0.08) !important;
            color: #00F5FF !important;
            border: 1px solid #00F5FF !important;
            border-radius: 10px !important;
            transition: all 0.25s !important;
        }
        .stButton > button:hover {
            background: rgba(0,245,255,0.18) !important;
            box-shadow: 0 0 20px rgba(0,245,255,0.3) !important;
            transform: translateY(-1px);
        }

        .stDownloadButton > button {
            font-family: 'Orbitron', monospace !important;
            background: rgba(0,255,159,0.08) !important;
            color: #00FF9F !important;
            border: 1px solid #00FF9F !important;
            border-radius: 10px !important;
        }

        .stSelectSlider > div, .stSlider > div { color: #00F5FF !important; }
        .stToggle label { color: #C8E8FF !important; font-family: 'Rajdhani', sans-serif !important; }
        .stFileUploader {
            background: rgba(0,20,50,0.5) !important;
            border: 2px dashed rgba(0,245,255,0.3) !important;
            border-radius: 16px !important;
        }
        .stFileUploader:hover { border-color: #00F5FF !important; box-shadow: 0 0 20px rgba(0,245,255,0.1) !important; }

        .stExpander {
            background: rgba(0,15,35,0.8) !important;
            border: 1px solid rgba(0,245,255,0.2) !important;
            border-radius: 12px !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.2); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,245,255,0.4); }

        /* ── DASHBOARD welcome ── */
        h3 { color: #00F5FF !important; }
        </style>
    """, unsafe_allow_html=True)
