# 📄 AI Resume Reviewer Agent

A beginner-friendly, single-agent application built using **CrewAI**, **Streamlit**, and **Groq** (`openai/gpt-oss-120b`). This tool evaluates how well a candidate's resume matches a target job description without fabricating unmentioned qualifications.

---

## 📁 Repository Structure

```text
resume-analyzer/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml
```

---

## 🚀 Local Setup & Installation

### 1. Prerequisites
- Python **3.11** installed.
- A free API key from [Groq Console](https://console.groq.com/).

### 2. Installation
1. Clone or download this repository.
2. Open your terminal in the project directory.
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configure API Key
Create a `.streamlit/secrets.toml` file in the root directory:
```toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. **Push your code to GitHub:**
   - Initialize git and commit `app.py`, `requirements.txt`, and `README.md`.
   - **DO NOT** commit `.streamlit/secrets.toml`.

2. **Deploy on Streamlit:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in.
   - Click **New App**, select your GitHub repository and branch (`main`).
   - Set **Main file path** to `app.py`.
   - Ensure the Python version is set to **3.11**.

3. **Add Secrets on Streamlit Cloud:**
   - Expand **Advanced Settings...**
   - In the **Secrets** box, paste:
     ```toml
     GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
     ```
   - Click **Deploy**.