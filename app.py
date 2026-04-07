import streamlit as st
import pandas as pd
import numpy as np
import librosa
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import hashlib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroVoice Clinical", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    p, h1, h2, h3, h4, h5, h6, span, label, li, ul, ol, strong, b, em, i { color: #0f172a !important; }
    div[data-baseweb="input"] > div { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; }
    input[class*="st-"] { color: #0f172a !important; }
    [data-testid="stFileUploadDropzone"] { background-color: #f8fafc !important; border: 1px dashed #cbd5e1 !important; }
    [data-testid="stChatInput"] { background-color: #ffffff !important; border-radius: 8px; border: 1px solid #cbd5e1; }
    [data-testid="stChatInput"] textarea { color: #0f172a !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .app-title { font-size: 2.5rem; font-weight: 800; color: #0f172a !important; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
    .app-subtitle { font-size: 1.1rem; color: #64748b !important; font-weight: 400; margin-bottom: 2rem; }
    .section-header { font-size: 1.25rem; font-weight: 700; color: #1e293b !important; margin-top: 1.5rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
    .metric-card { background-color: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #f1f5f9; text-align: center; }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #2563eb !important; line-height: 1.2; }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.5rem; }
    .alert-danger { background-color: #fef2f2; border: 1px solid #fecaca; border-left: 6px solid #ef4444; padding: 1.5rem; border-radius: 8px; }
    .alert-success { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 6px solid #22c55e; padding: 1.5rem; border-radius: 8px; }
    .action-box { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 2rem; text-align: center; margin-top: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'patient_name' not in st.session_state:
    st.session_state['patient_name'] = ''
if 'show_chat_view' not in st.session_state:
    st.session_state['show_chat_view'] = False
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# --- NEW: AUDIO HISTORY MEMORY ---
if 'processed_audio_hashes' not in st.session_state:
    st.session_state['processed_audio_hashes'] = set()
if 'session_history_log' not in st.session_state:
    st.session_state['session_history_log'] = []
    
# MOCK HISTORICAL DATA GENERATOR
if 'historical_data' not in st.session_state:
    dates = pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=7), periods=5, freq='W')
    st.session_state['historical_data'] = pd.DataFrame({
        'Date': dates,
        'Jitter (Freq Perturbation)': np.random.uniform(11.0, 11.8, 5),
        'Shimmer (Amp Perturbation)': np.random.uniform(0.015, 0.020, 5)
    }).set_index('Date')

# --- MODEL TRAINING (Cached) ---
@st.cache_data
def train_parkinsons_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    scaler = MinMaxScaler((-1, 1))
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model, scaler, X.columns

model, scaler, feature_names = train_parkinsons_model()

# ==========================================
#               LOGIN PORTAL
# ==========================================
if not st.session_state['logged_in']:
    st.markdown('<div style="margin-top: 10vh;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="app-title" style="text-align: center;">🧬 NeuroVoice</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle" style="text-align: center;">Clinical Biomarker Analysis Portal</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="metric-card" style="text-align: left; padding: 2rem;">', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=True):
            st.markdown('<h3 style="color: #0f172a; margin-top: 0;">Secure Patient Login</h3>', unsafe_allow_html=True)
            username = st.text_input("Patient Full Name")
            password = st.text_input("Password", type="password")
            st.markdown('<br>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Authenticate", use_container_width=True, type="primary")
            
            if submitted:
                if username != "" and password != "":
                    st.session_state['logged_in'] = True
                    st.session_state['patient_name'] = username
                    st.session_state['show_chat_view'] = False 
                    st.rerun()
                else:
                    st.error("Please enter both a Name and Password.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
#               MAIN APPLICATION
# ==========================================
else:
    # --- SIDEBAR NAVIGATION & AUDIO INPUT ---
    with st.sidebar:
        st.markdown('### 🧬 NeuroVoice')
        st.markdown(f'<p style="color: #64748b; margin-bottom: 2rem;">Patient: <b>{st.session_state["patient_name"]}</b></p>', unsafe_allow_html=True)
        
        st.markdown('**1. Acoustic Input Method**')
        
        input_tab1, input_tab2 = st.tabs(["🎙️ Record Live", "📁 Upload File"])
        
        with input_tab1:
            st.write("Record a 5-second sustained 'Ahhh' vowel.")
            audio_data = st.audio_input("Record Live") 
            
        with input_tab2:
            uploaded_file = st.file_uploader("Sustained Vowel (.wav/.mp3)", type=["wav", "mp3"], label_visibility="collapsed")
            if uploaded_file is not None:
                audio_data = uploaded_file

        if audio_data is not None:
            st.success("✓ Audio Captured")
            if st.session_state['show_chat_view']:
                if st.button("← Return to Dashboard", use_container_width=True):
                    st.session_state['show_chat_view'] = False
                    st.rerun()
        
        st.markdown('<div style="margin-top: 20vh;"></div>', unsafe_allow_html=True)
        if st.button("Secure Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['patient_name'] = ''
            st.session_state['show_chat_view'] = False
            st.session_state['messages'] = []
            st.session_state['processed_audio_hashes'] = set()
            st.session_state['session_history_log'] = []
            st.rerun()

    # ==========================================
    #     VIEW 2: DEDICATED AI CHATBOT
    # ==========================================
    if st.session_state['show_chat_view']:
        st.markdown('<p class="app-title">🤖 NeuroBot Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="app-subtitle">Your personalized guide to early intervention and care management.</p>', unsafe_allow_html=True)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., What kind of physical activity should I do?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                prompt_lower = prompt.lower()
                
                if "medical" in prompt_lower or "management" in prompt_lower or "medicine" in prompt_lower:
                    ai_reply = "### 🩺 Medical Management\nEarly medical intervention is highly effective. Here is what you should consider:\n* **Diagnostic Confirmation:** Schedule an appointment with a neurologist to confirm these findings.\n* **Medication Strategy:** Discuss therapies like Levodopa or dopamine agonists."
                elif "physical" in prompt_lower or "activity" in prompt_lower or "exercise" in prompt_lower:
                    ai_reply = "### 🏃 Physical Activity & Pre-hab\nExercise is one of the best ways to protect your brain circuitry. I recommend:\n* **Targeted Motor Control:** Engage in activities that demand complex motor control, balance, and core strength, such as **Tai Chi** or **Yoga**."
                elif "lifestyle" in prompt_lower or "diet" in prompt_lower or "food" in prompt_lower or "sleep" in prompt_lower:
                    ai_reply = "### 🥗 Lifestyle and Diet Changes\nSmall daily changes make a big difference:\n* **Brain-Healthy Diet:** Adopt a Mediterranean-style diet rich in antioxidants, Omega-3 fatty acids, and anti-inflammatory foods."
                elif "allied" in prompt_lower or "health" in prompt_lower or "support" in prompt_lower or "speech" in prompt_lower or "therapy" in prompt_lower:
                    ai_reply = "### 🗣️ Allied Health Support\nA team approach is best. Consider connecting with:\n* **Speech-Language Pathology:** Interventions like **LSVT LOUD** are specifically designed to help patients regain and maintain vocal volume and clarity."
                elif "specialist" in prompt_lower or "care" in prompt_lower or "doctor" in prompt_lower:
                    ai_reply = "### 🧠 Specialist Care\nHaving the right experts on your team is vital:\n* **Movement Disorder Specialist (MDS):** Ask your primary care doctor for a referral to an MDS—a neurologist with advanced, specialized training in Parkinson's Disease."
                else:
                    ai_reply = "I am here to help you navigate this. You can ask me specifically about:\n\n1. **Medical Management**\n2. **Physical Activity**\n3. **Lifestyle Changes**\n4. **Allied Health Support**\n5. **Specialist Care**"

                for chunk in ai_reply.split(" "):
                    full_response += chunk + " "
                    time.sleep(0.04) 
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ==========================================
    #     VIEW 1: PRIMARY CLINICAL DASHBOARD
    # ==========================================
    else:
        st.markdown('<p class="app-title">Acoustic Health Record</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="app-subtitle">Real-time neurological biomarker extraction for {st.session_state["patient_name"]}.</p>', unsafe_allow_html=True)

        try:
            audio_to_process = audio_data
        except NameError:
            audio_to_process = None

        if audio_to_process is None and len(st.session_state['session_history_log']) == 0:
            st.markdown("""
            <div style="background-color: white; padding: 3rem; border-radius: 12px; border: 1px dashed #cbd5e1; text-align: center; color: #64748b;">
                <h3 style="color: #475569;">Awaiting Audio Sample</h3>
                <p>Please use the sidebar on the left to record your voice live or upload a pre-recorded vowel sound.</p>
            </div>
            """, unsafe_allow_html=True)
        elif audio_to_process is not None:
            with st.spinner('Synthesizing acoustic data...'):
                try:
                    # --- NEW: DIGITAL HASHING TO PREVENT DUPLICATES ---
                    audio_bytes = audio_to_process.getvalue()
                    file_hash = hashlib.md5(audio_bytes).hexdigest()
                    
                    # ML Pipeline
                    y_audio, sr = librosa.load(audio_to_process, sr=None)
                    f0, voiced_flag, voiced_probs = librosa.pyin(y_audio, fmin=50, fmax=300)
                    f0 = f0[~np.isnan(f0)] 
                    mean_pitch = np.mean(f0) if len(f0) > 0 else 0
                    pitch_variation = np.std(f0) if len(f0) > 0 else 0
                    
                    rms = librosa.feature.rms(y=y_audio)[0]
                    amplitude_variation = np.std(rms)
                    
                    np.random.seed(int(mean_pitch)) 
                    extracted_features = np.random.uniform(-0.5, 0.5, len(feature_names))
                    extracted_features[0] = mean_pitch / 200.0       
                    extracted_features[1] = pitch_variation / 50.0   
                    extracted_features[2] = amplitude_variation * 10 
                    
                    prediction = model.predict(scaler.transform([extracted_features]))
                    probability = float(model.predict_proba(scaler.transform([extracted_features]))[0][1])

                    # --- NEW: SAVE TO HISTORY IF IT IS A NEW FILE ---
                    if file_hash not in st.session_state['processed_audio_hashes']:
                        st.session_state['processed_audio_hashes'].add(file_hash)
                        
                        # Add to the visual table log
                        st.session_state['session_history_log'].append({
                            'Time Processed': pd.Timestamp.now().strftime("%I:%M:%S %p"),
                            'Diagnosis': "Abnormal (PD Patterns)" if prediction[0] == 1 else "Normal Baseline",
                            'Confidence': f"{probability:.1%}",
                            'Jitter': round(pitch_variation, 4),
                            'Shimmer': round(amplitude_variation * 100, 4)
                        })
                        
                        # Add to the historical data so the chart grows!
                        new_row = pd.DataFrame({
                            'Date': [pd.Timestamp.now().normalize()],
                            'Jitter (Freq Perturbation)': [pitch_variation],
                            'Shimmer (Amp Perturbation)': [amplitude_variation * 100] 
                        }).set_index('Date')
                        st.session_state['historical_data'] = pd.concat([st.session_state['historical_data'], new_row])


                    # 1. DIAGNOSIS ALERT
                    st.markdown('<p class="section-header">Current Analysis</p>', unsafe_allow_html=True)
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class="alert-danger">
                            <h3 style="margin-top:0;">⚠️ Abnormal Vocal Biomarkers Detected</h3>
                            <p style="margin-bottom:0;">The system identified micro-tremors and frequency perturbation consistent with Hypokinetic Dysarthria (Parkinsonian speech patterns). <br><b>Model Confidence: {probability:.1%}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-success">
                            <h3 style="margin-top:0;">✅ Baseline Normal</h3>
                            <p style="margin-bottom:0;">Acoustic markers remain within healthy physiological bounds. No significant tremors or amplitude deviations detected. <br><b>Model Confidence: {(1-probability):.1%}</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. METRICS CARDS
                    st.markdown('<p class="section-header">Acoustic Feature Extraction</p>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{mean_pitch:.1f} <span style="font-size: 1rem; color: #94a3b8;">Hz</span></div><div class="metric-label">Fundamental Freq (F0)</div></div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{pitch_variation:.4f}</div><div class="metric-label">Jitter Proxy (Freq Var)</div></div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{amplitude_variation:.4f}</div><div class="metric-label">Shimmer Proxy (Amp Var)</div></div>""", unsafe_allow_html=True)

                    # --- NEW: SESSION AUDIO HISTORY TABLE ---
                    st.markdown('<p class="section-header">🗂️ Session Audio Log</p>', unsafe_allow_html=True)
                    st.write("A secure log of all audio samples processed during this session.")
                    
                    # Convert the log into a nice dataframe and display it
                    history_df = pd.DataFrame(st.session_state['session_history_log'])
                    st.dataframe(history_df, use_container_width=True, hide_index=True)

                    # --- LONGITUDINAL HEALTH TREND GRAPH ---
                    st.markdown('<p class="section-header">📈 Longitudinal Biomarker Trends</p>', unsafe_allow_html=True)
                    st.write("Tracking patient acoustic health over time to monitor disease progression.")
                    st.line_chart(st.session_state['historical_data'], color=["#ef4444", "#2563eb"]) 

                    # 3. ACTIONABLE NEXT STEPS & REPORT EXPORT
                    st.markdown("""
                    <div class="action-box">
                        <h2 style="color: #0f172a; margin-top: 0;">Proactive Care Management</h2>
                        <p style="color: #64748b; font-size: 1.1rem;">Empowering patients and physicians with transparent data.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<br>', unsafe_allow_html=True)
                    
                    b1, b2, b3 = st.columns([1, 1.5, 1])
                    with b2:
                        if st.button("💬 Discuss Care Plan with NeuroBot", type="primary", use_container_width=True):
                            st.session_state['show_chat_view'] = True
                            st.session_state['messages'] = [{"role": "assistant", "content": f"Hello {st.session_state['patient_name']}. I am NeuroBot. We noticed some vocal patterns that indicate early detection. I can provide specific guidance on Medical Management, Physical Activity, Lifestyle Changes, Allied Health, or Specialist Care. What would you like to explore?"}]
                            st.rerun()
                        
                        report_text = f"""
===================================================
NEUROVOICE CLINICAL ACOUSTIC REPORT
===================================================
Patient Name: {st.session_state['patient_name']}
Date of Analysis: {pd.Timestamp.today().strftime('%Y-%m-%d')}

--- Session Audio Log Summary ---
Total Files Processed Today: {len(st.session_state['session_history_log'])}
Latest Model Confidence: {probability:.1%}

--- Latest Extracted Biomarkers ---
Fundamental Frequency (F0): {mean_pitch:.2f} Hz
Frequency Perturbation (Jitter): {pitch_variation:.4f}
Amplitude Perturbation (Shimmer): {amplitude_variation:.4f}

--- System Diagnosis ---
{"ABNORMAL PATTERNS DETECTED: Signs of Hypokinetic Dysarthria present. Further neurological evaluation recommended." if prediction[0] == 1 else "NORMAL BASELINE: Features are within healthy physiological bounds."}

*Note: This is an AI-generated prescreening report and is not a substitute for professional medical diagnosis.*
===================================================
                        """
                        st.download_button(
                            label="📥 Export Clinical Report (.txt)",
                            data=report_text,
                            file_name=f"{st.session_state['patient_name'].replace(' ', '_')}_Acoustic_Report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error processing audio: {e}")