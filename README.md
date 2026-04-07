# 🧬 NeuroVoice Clinical Portal

**Real-time acoustic biomarker analysis and patient care management for early neurodegenerative detection.**

## 💡 The Problem
Neurodegenerative diseases, such as Parkinson's Disease, are often diagnosed years after the onset of physiological changes. Early detection is critical for effective symptom management, but clinical voice analysis is often expensive, inaccessible, and requires specialized hardware. 

## 🚀 Our Solution
**NeuroVoice** is a secure, patient-facing web application that democratizes clinical-grade acoustic analysis. By simply recording a 5-second sustained vowel sound ("Ahhh") into any standard microphone, our machine learning pipeline extracts vocal micro-tremors and frequency perturbations to identify signs of Hypokinetic Dysarthria (Parkinsonian speech patterns).

## ✨ Key Features
* **🎙️ Live Audio Processing:** Patients can record audio directly in the browser or upload pre-recorded `.wav`/`.mp3` files.
* **🧠 ML Biomarker Extraction:** Extracts Fundamental Frequency (F0), Jitter (Frequency Perturbation), and Shimmer (Amplitude Perturbation) using advanced digital signal processing.
* **📈 Longitudinal Tracking:** Automatically logs session data and visualizes historical patient trends on a dynamic chart to monitor disease progression or stabilization over time.
* **🤖 NeuroBot AI Assistant:** A built-in conversational agent that guides patients through their results and provides actionable, multidisciplinary care plans (Diet, Exercise, Medical, Specialist Care).
* **📥 Clinical Report Export:** Instantly generates a formatted, timestamped `.txt` clinical report that patients can hand directly to their neurologist.
* **🔒 Secure Patient Portal:** Session-state authenticated dashboards customized to the individual patient.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **Audio Processing:** Librosa (Digital Signal Processing)
* **Machine Learning:** XGBoost Classifier, Scikit-Learn (MinMaxScaler)
* **Data Management:** Pandas, NumPy
* **UI/UX:** Custom CSS injections for a clean, premium clinical aesthetic.

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone [https://github.com/prajvalgowda/healthcare.git](https://github.com/prajvalgowda/healthcare.git)
