# AI Resume Reviewer

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Before starting, create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
# Optional provider model ID (no groq/ prefix):
GROQ_MODEL = "openai/gpt-oss-120b"
```

Alternatively set GROQ_API_KEY and optionally GROQ_MODEL as environment variables.
Never commit the secrets file. On Streamlit Community Cloud add these settings
through the app's Secrets settings. Put app.py, requirements.txt and packages.txt
in the repository root. packages.txt installs the OCR system dependencies there.

For local Ubuntu/Debian OCR support:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

On Windows or macOS install Tesseract and Poppler separately and make their
executables available on PATH. Default OCR language is English. Pasted text and
PDFs with readable text layers do not require these system programs.

This implementation intentionally replaces CrewAI/LiteLLM with Groq's official
SDK for the single review task. It sends only role and content in each message,
so the unsupported cache_breakpoint property is never introduced. The SDK model
ID is openai/gpt-oss-120b, not groq/openai/gpt-oss-120b.

PDFs are limited to 10 MB and 20 pages. Combined review input is limited to
40,000 characters; account-specific Groq token quotas may require shorter inputs.
OCR runs only for pages with no extracted text. Inspect the preview: sparse or
corrupted text layers may require manually pasting corrected text.

Reports persist during widget reruns and clear when the input or model changes.
Resume data is sent to Groq when Analyze is clicked. Match scores are qualitative
model estimates and should be checked against the quoted resume evidence.
