import os
import streamlit as st
from pypdf import PdfReader
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Review & Optimization Agent")
st.caption("Powered by CrewAI & Groq (openai/gpt-oss-120b)")

# ==============================================================================
# SECURE API KEY INITIALIZATION
# ==============================================================================
# Retrieve key safely from Streamlit secrets or local environment variables
groq_api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🔑 GROQ API Key missing! Please add `GROQ_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()

# Set environment variable for native LangChain/CrewAI detection
os.environ["GROQ_API_KEY"] = groq_api_key


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text safely from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {str(e)}")
        return ""


def run_resume_review(resume_text: str, job_description: str) -> str:
    """Configures CrewAI Agent with GPT OSS 120B and executes evaluation workflow."""
    
    # Initialize Groq LLM model
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
        temperature=0.2
    )

    # 1. Single Agent definition with strict factual backstory guarding against fabrication
    resume_evaluator = Agent(
        role="Senior Executive Technical Recruiter & Resume Auditor",
        goal="Accurately evaluate resume alignment against job descriptions and provide targeted, zero-hallucination feedback.",
        backstory=(
            "You are a seasoned talent acquisition strategist with over 15 years of experience evaluating candidate resumes. "
            "You are meticulous, strictly factual, and honest. You never fabricate skills, certifications, or experiences "
            "that are not explicitly present in the candidate's provided text. Your advice is concise, structured, and actionable."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # 2. Evaluation Task definition with structured output constraints
    review_task = Task(
        description=(
            "Carefully analyze the Candidate Resume against the Target Job Description below.\n\n"
            "=== CANDIDATE RESUME ===\n{resume}\n\n"
            "=== TARGET JOB DESCRIPTION ===\n{job_description}\n\n"
            "Your evaluation MUST adhere strictly to the following rules:\n"
            "1. DO NOT assume or invent skills/experiences not explicitly mentioned in the resume.\n"
            "2. Identify clear structural, technical, and keywords missing from the candidate's resume relative to the target role.\n"
            "3. Format the final output clearly using clean Markdown sections."
        ),
        expected_output=(
            "A structured report in Markdown format containing:\n"
            "## 🎯 Match Overview & Overall Score (0-100%)\n"
            "- Concise explanation of candidate fit\n\n"
            "## ✅ Key Strengths & Matching Qualifications\n"
            "- Bullet points highlighting explicit matches found in the resume\n\n"
            "## ⚠️ Missing Skills & Requirements Gaps\n"
            "- Critical requirements in the job description that are missing or weak in the resume\n\n"
            "## 💡 Actionable Improvement Recommendations\n"
            "- 3 to 5 specific, step-by-step revisions to optimize ATS readability and job alignment"
        ),
        agent=resume_evaluator
    )

    # 3. Instantiate and execute Crew
    crew = Crew(
        agents=[resume_evaluator],
        tasks=[review_task],
        process=Process.sequential
    )

    result = crew.kickoff(inputs={"resume": resume_text, "job_description": job_description})
    return str(result)


# ==============================================================================
# STREAMLIT USER INTERFACE
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    input_method = st.radio("Choose input method:", ["Upload PDF", "Paste Text"], horizontal=True)
    
    resume_content = ""
    if input_method == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload PDF Resume", type=["pdf"])
        if uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                resume_content = extract_text_from_pdf(uploaded_pdf)
                if resume_content:
                    st.success("PDF processed successfully!")
                    with st.expander("Preview Extracted Resume Text"):
                        st.text(resume_content[:1000] + "..." if len(resume_content) > 1000 else resume_content)
    else:
        resume_content = st.text_area("Paste Resume Text here:", height=300)

with col2:
    st.subheader("2. Target Job Description")
    job_desc_content = st.text_area("Paste Job Description here:", height=350)

st.divider()

# Analyze Trigger
if st.button("🔍 Analyze Resume Alignment", type="primary", use_container_width=True):
    # Validation checks
    if not resume_content.strip():
        st.warning("⚠️ Please provide a resume (either upload a valid PDF or paste text).")
    elif not job_desc_content.strip():
        st.warning("⚠️ Please paste the target job description.")
    else:
        with st.spinner("🤖 Agent is analyzing your resume with GPT OSS 120B..."):
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