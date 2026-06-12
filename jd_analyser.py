import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

# =====================================================
# AI ENGINEER SKILLS DATABASE
# =====================================================

SKILLS = {
    "Programming": [
        "python",
        "java",
        "sql"
    ],

    "Machine Learning": [
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "nlp"
    ],

    "Generative AI": [
        "llm",
        "large language model",
        "langchain",
        "rag",
        "retrieval augmented generation",
        "hugging face",
        "transformers",
        "prompt engineering"
    ],

    "Vector Databases": [
        "faiss",
        "chromadb",
        "pinecone"
    ],

    "Cloud & MLOps": [
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "mlops",
        "git",
        "cicd"
    ]
}

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_file):

    text = ""

    try:
        reader = PdfReader(pdf_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:

        st.error(f"Error reading PDF: {e}")

    return text.lower()

# =====================================================
# SKILL EXTRACTION
# =====================================================

def extract_skills(text):

    found = {}

    for category, skill_list in SKILLS.items():

        found[category] = []

        for skill in skill_list:

            if skill.lower() in text:
                found[category].append(skill)

    return found

# =====================================================
# FLATTEN SKILLS
# =====================================================

def flatten_skills(skill_dict):

    skills = set()

    for category in skill_dict:

        skills.update(skill_dict[category])

    return skills

# =====================================================
# EVALUATE CANDIDATE
# =====================================================

def evaluate_candidate(jd_text, resume_text):

    jd_skills = extract_skills(jd_text)

    resume_skills = extract_skills(resume_text)

    jd_set = flatten_skills(jd_skills)

    resume_set = flatten_skills(resume_skills)

    present = jd_set.intersection(resume_set)

    missing = jd_set - resume_set

    if len(jd_set) == 0:
        score = 0
    else:
        score = round(
            (len(present) / len(jd_set)) * 100,
            2
        )

    if score >= 75:
        verdict = "SHORTLISTED"

    elif score >= 50:
        verdict = "CONSIDER"

    else:
        verdict = "REJECTED"

    return score, verdict, present, missing

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening Agent",
    layout="wide"
)

st.title("🤖 AI Engineer Resume Screening Agent")

st.markdown(
    "Paste a Job Description and upload candidate resumes."
)

# =====================================================
# JD INPUT
# =====================================================

jd_text = st.text_area(
    "Paste Job Description",
    height=250
)

# =====================================================
# RESUME UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# =====================================================
# PROCESS BUTTON
# =====================================================

if st.button("Evaluate Candidates"):

    if not jd_text.strip():

        st.warning("Please paste a Job Description.")

        st.stop()

    if not uploaded_files:

        st.warning("Please upload at least one resume.")

        st.stop()

    results = []

    for pdf in uploaded_files:

        resume_text = extract_text_from_pdf(pdf)

        score, verdict, present, missing = evaluate_candidate(
            jd_text.lower(),
            resume_text
        )

        results.append({
            "Candidate": pdf.name,
            "Score (%)": score,
            "Verdict": verdict,
            "Present Skills": ", ".join(sorted(present)),
            "Missing Skills": ", ".join(sorted(missing))
        })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Score (%)",
        ascending=False
    )

    st.subheader("📊 Candidate Ranking")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("📋 Detailed Analysis")

    for _, row in df.iterrows():

        with st.expander(
            f"{row['Candidate']} | {row['Verdict']}"
        ):

            st.write(f"### Match Score: {row['Score (%)']}%")

            st.success(
                f"Present Skills:\n\n{row['Present Skills']}"
            )

            st.error(
                f"Missing Skills:\n\n{row['Missing Skills']}"
            )

            report = f"""
Candidate: {row['Candidate']}

Match Score: {row['Score (%)']}%

Verdict: {row['Verdict']}

Present Skills:
{row['Present Skills']}

Missing Skills:
{row['Missing Skills']}
"""

            st.download_button(
                label="Download Report",
                data=report,
                file_name=row["Candidate"].replace(
                    ".pdf",
                    "_report.txt"
                ),
                mime="text/plain"
            )
