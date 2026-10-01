import os
import pytesseract
from pdf2image import convert_from_bytes
import streamlit as st
import litellm
from pypdf import PdfReader
from crewai import LLM, Agent, Task, Crew, Process
import litellm

# Monkey-patch LiteLLM completion to sanitize incoming arguments for Groq
_original_completion = litellm.completion

def sanitized_completion(*args, **kwargs):
    # Remove unsupported parameters if present
    kwargs.pop("cache_breakpoint", None)
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = sanitized_completion

# Force LiteLLM environment settings globally
os.environ["LITELLM_DROP_PARAMS"] = "True"
litellm.drop_params = True
litellm.modify_params = True

# Page Config
st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Review & Optimization Agent")
st.caption("Powered by CrewAI, Groq & OCR")

# API Key Handling
groq_api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🔑 GROQ API Key missing! Please add `GROQ_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


def extract_text_from_pdf(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    extracted_text = ""
    
    # 1. Standard text extraction
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        extracted_text = extracted_text.strip()
    except Exception as e:
        st.warning(f"Standard PDF reading issue: {str(e)}. Attempting OCR...")

    # 2. OCR Fallback
    if not extracted_text:
        try:
            with st.spinner("🔍 Scanned PDF detected. Running OCR..."):
                images = convert_from_bytes(file_bytes)
                ocr_text_list = []
                for image in images:
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        ocr_text_list.append(text.strip())
                extracted_text = "\n\n".join(ocr_text_list)
        except Exception as ocr_err:
            st.error(
                "OCR extraction failed. Ensure 'tesseract-ocr' and 'poppler-utils' "
                f"are installed. Error: {str(ocr_err)}"
            )
            return ""

    return extracted_text


def run_resume_review(resume_text: str, job_description: str) -> str:
    # Explicitly configure CrewAI LLM with drop_params
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
        drop_params=True,
        cache=False  # Disables prompt caching that injects 'cache_breakpoint'
    )

    resume_evaluator = Agent(
        role="Senior Executive Technical Recruiter & Resume Auditor",
        goal="Provide an accurate, honest evaluation of resume alignment without inventing unmentioned candidate qualifications.",
        backstory=(
            "You are a seasoned talent acquisition strategist with over 15 years of technical hiring experience. "
            "You are meticulous, strictly factual, and honest. You never assume or extrapolate skills that are "
            "not explicitly listed on the candidate's resume."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    review_task = Task(
        description=(
            "Analyze the candidate's resume against the target job description based EXCLUSIVELY on the provided text.\n"
            "CRITICAL CONSTRAINT: Do NOT invent, assume, or fabricate any experience or qualifications that are not explicitly in the resume.\n\n"
            "=== CANDIDATE RESUME ===\n{resume}\n\n"
            "=== TARGET JOB DESCRIPTION ===\n{job_description}"
        ),
        expected_output=(
            "A structured Markdown report including:\n"
            "1. **Overall Match Score** (Percentage 0-100% with brief justification)\n"
            "2. **Matching Core Competencies** (Strengths explicitly present in both)\n"
            "3. **Missing or Unmentioned Requirements** (Gaps found relative to the job description)\n"
            "4. **Actionable Recommendations** (Concrete advice on how to emphasize existing experience or address gaps without falsifying information)"
        ),
        agent=resume_evaluator
    )

    crew = Crew(agents=[resume_evaluator], tasks=[review_task], process=Process.sequential)
    return str(crew.kickoff(inputs={"resume": resume_text, "job_description": job_description}))


# Streamlit UI
col1, col2 = st.columns(2)
resume_content = ""

with col1:
    st.subheader("1. Candidate Resume")
    input_method = st.radio("Choose input method:", ["Upload PDF", "Paste Text"], horizontal=True)
    
    if input_method == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload PDF Resume", type=["pdf"])
        if uploaded_pdf is not None:
            resume_content = extract_text_from_pdf(uploaded_pdf)
            if resume_content:
                st.success("PDF text extracted successfully!")
                with st.expander("Preview Extracted Resume Text"):
                    st.text(resume_content[:1000] + "..." if len(resume_content) > 1000 else resume_content)
            else:
                st.error("Could not extract readable text even with OCR. Please try pasting text manually.")
    else:
        resume_content = st.text_area("Paste Resume Text here:", height=300)

with col2:
    st.subheader("2. Target Job Description")
    job_desc_content = st.text_area("Paste Job Description here:", height=350)

st.divider()

if st.button("🔍 Analyze Resume Alignment", type="primary", use_container_width=True):
    if not resume_content or not resume_content.strip():
        st.warning("⚠️ Please provide a valid resume.")
    elif not job_desc_content or not job_desc_content.strip():
        st.warning("⚠️ Please paste the target job description.")
    else:
        with st.spinner("🤖 Agent is analyzing your resume alignment..."):
            try:
                review_report = run_resume_review(resume_content, job_desc_content)
                st.subheader("📋 Audit Report")
                st.markdown(review_report)
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    st.error("⏳ Groq API rate limit reached. Please wait a minute before trying again.")
                else:
                    st.error(f"An unexpected error occurred during execution: {error_msg}")
