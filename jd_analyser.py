import streamlit as st
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    box-shadow: 0 25px 80px rgba(79, 70, 229, 0.4);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
}
.hero h1 { color: white; font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero p { color: rgba(255,255,255,0.8); font-size: 1.15rem; margin-top: 12px; }

/* Stat Cards */
.stat-card {
    background: linear-gradient(145deg, #141428, #1a1a38);
    border: 1px solid rgba(79, 70, 229, 0.25);
    border-radius: 18px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.stat-num {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.stat-label { color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }

/* Winner */
.winner {
    background: linear-gradient(135deg, #d97706, #f59e0b, #fbbf24);
    border-radius: 18px;
    padding: 22px 30px;
    text-align: center;
    margin: 24px 0;
    box-shadow: 0 12px 40px rgba(245, 158, 11, 0.35);
}
.winner h2 { color: white; margin: 0; font-size: 1.6rem; font-weight: 800; text-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.winner p { color: rgba(255,255,255,0.9); margin: 6px 0 0; font-size: 1rem; }

/* Skill pills */
.skill-match {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.06));
    border: 1px solid rgba(16,185,129,0.45);
    border-radius: 12px;
    padding: 11px 16px;
    margin: 5px 0;
    color: #34d399;
    font-size: 0.9rem;
    font-weight: 500;
}
.skill-match-keyword {
    background: linear-gradient(135deg, rgba(16,185,129,0.3), rgba(16,185,129,0.1));
    border: 2px solid rgba(16,185,129,0.7);
    border-radius: 12px;
    padding: 11px 16px;
    margin: 5px 0;
    color: #6ee7b7;
    font-size: 0.9rem;
    font-weight: 700;
}
.skill-miss {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 12px;
    padding: 11px 16px;
    margin: 5px 0;
    color: #f87171;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Section title */
.sec-title {
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 28px 0 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(79,70,229,0.35);
    letter-spacing: -0.3px;
}

/* Resume card header */
.res-header {
    background: linear-gradient(145deg, #141428, #1a1a38);
    border: 1px solid rgba(79,70,229,0.25);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

/* Upload area */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(20,20,40,0.9) !important;
    border: 2px dashed rgba(79,70,229,0.5) !important;
    border-radius: 16px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    width: 100% !important;
    box-shadow: 0 6px 24px rgba(79,70,229,0.45) !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

h1,h2,h3,h4 { color: white !important; }
p, li, label { color: rgba(255,255,255,0.75) !important; }
.stSlider label { color: rgba(255,255,255,0.75) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI Model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────
# PDF text extract
# ─────────────────────────────────────────
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages).strip()

# ─────────────────────────────────────────
# HYBRID Skill Matching (Keyword + Semantic)
# Exact keyword match = strong match (green bold)
# Semantic > threshold = soft match (green)
# Below threshold = missing (red)
# ─────────────────────────────────────────
def analyze_resume(model, skills, resume_text, threshold):
    matched, missing = [], []
    resume_lower = resume_text.lower()

    # Split resume into sentences for semantic search
    sentences = [s.strip() for s in resume_text.replace("\n", ". ").split(".") if len(s.strip()) > 8]
    if not sentences:
        return matched, missing

    res_embeddings = model.encode(sentences, convert_to_tensor=True)

    for skill in skills:
        skill_lower = skill.lower()

        # 1. Exact keyword check first
        keyword_found = skill_lower in resume_lower

        # 2. Semantic similarity
        skill_emb = model.encode(skill, convert_to_tensor=True)
        scores = util.cos_sim(skill_emb, res_embeddings)[0]
        best_score = float(scores.max())
        best_sentence = sentences[int(scores.argmax())]

        if keyword_found:
            # Direct keyword match — most reliable
            matched.append({
                "skill": skill,
                "score": max(round(best_score * 100), 75),
                "reason": best_sentence[:120],
                "type": "keyword"   # exact match
            })
        elif best_score >= threshold:
            # Semantic match
            matched.append({
                "skill": skill,
                "score": round(best_score * 100),
                "reason": best_sentence[:120],
                "type": "semantic"
            })
        else:
            missing.append({
                "skill": skill,
                "score": round(best_score * 100)
            })

    return matched, missing

def score_color(pct):
    if pct >= 60: return "#10b981"
    elif pct >= 35: return "#f59e0b"
    else: return "#ef4444"

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    threshold = st.slider(
        "Semantic Match Sensitivity",
        0.30, 0.65, 0.50, 0.05,
        help="Higher = strict (only exact matches) | Lower = loose (more matches)"
    )
    st.caption("💡 Keyword matches are not affected by sensitivity")

    st.markdown("---")
    st.markdown("### 📋 Job Description Skills")
    st.caption("Enter one skill per line")

    jd_input = st.text_area("", value="""Python
Machine Learning
Deep Learning
Natural Language Processing
LangChain
FastAPI
Docker
AWS
Git
SQL
Data Analysis
Computer Vision
TensorFlow
REST APIs
Prompt Engineering""", height=340, label_visibility="collapsed")

    skills = [s.strip() for s in jd_input.strip().split("\n") if s.strip()]
    st.markdown(f"**{len(skills)} skills** loaded")

    st.markdown("---")
    st.markdown("### 🔍 Match Types")
    st.markdown('<div class="skill-match-keyword">🔑 Keyword Match — Exact</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-match">🧠 Semantic Match — AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-miss">❌ Missing Skill</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🤖 AI Resume Analyzer</h1>
    <p>Upload resumes • AI skill matching • Find the best candidate instantly</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────
st.markdown('<p class="sec-title">📤 Upload Resumes</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop PDF resumes here", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
if uploaded:
    st.success(f"✅ {len(uploaded)} resume(s) ready")

st.markdown("<br>", unsafe_allow_html=True)
go_btn = st.button("🚀 Analyze")

# ─────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────
if go_btn:
    if not uploaded:
        st.error("❌ Please upload resumes first!")
        st.stop()
    if len(skills) < 2:
        st.error("❌ Please enter skills in the sidebar!")
        st.stop()

    model = load_model()
    results = []
    bar = st.progress(0, "Analyzing...")

    for i, f in enumerate(uploaded):
        bar.progress((i+1)/len(uploaded), f"🔍 Analyzing: {f.name}")
        text = extract_text(f)
        if text:
            matched, missing = analyze_resume(model, skills, text, threshold)
            pct = round(len(matched)/len(skills)*100)
            results.append({
                "name": f.name.replace(".pdf",""),
                "short": f.name.replace(".pdf","")[:22],
                "matched": matched,
                "missing": missing,
                "score": len(matched),
                "total": len(skills),
                "pct": pct
            })

    bar.empty()

    if not results:
        st.error("❌ Could not extract text from the resumes!")
        st.stop()

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results_sorted[0]

    # ── STATS ──
    st.markdown("---")
    st.markdown('<p class="sec-title">📊 Overview</p>', unsafe_allow_html=True)

    cols = st.columns(len(results)+1)
    with cols[0]:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(results)}</div><div class="stat-label">Resumes Analyzed</div></div>', unsafe_allow_html=True)
    for i, r in enumerate(results_sorted):
        with cols[i+1]:
            c = score_color(r["pct"])
            st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:{c}">{r["score"]}/{r["total"]}</div><div class="stat-label">{r["short"]}</div></div>', unsafe_allow_html=True)

    # ── WINNER ──
    st.markdown(f"""
    <div class="winner">
        <h2>🏆 Best Candidate: {best['name']}</h2>
        <p>{best['score']}/{best['total']} skills matched &nbsp;|&nbsp; {best['pct']}% match score</p>
    </div>""", unsafe_allow_html=True)

    # ── BAR CHART ──
    st.markdown('<p class="sec-title">📈 Candidate Comparison</p>', unsafe_allow_html=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[r["short"] for r in results_sorted],
        y=[r["pct"] for r in results_sorted],
        marker=dict(
            color=[score_color(r["pct"]) for r in results_sorted],
            line=dict(width=0)
        ),
        text=[f"{r['pct']}%<br>({r['score']}/{r['total']})" for r in results_sorted],
        textposition="outside",
        textfont=dict(color="white", size=13, family="Inter"),
        width=0.5
    ))
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        yaxis=dict(range=[0,115], gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.5)", title="Match %"),
        xaxis=dict(color="rgba(255,255,255,0.7)"),
        showlegend=False, height=340,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="bar_comparison")

    # ── RADAR CHART ──
    if len(results) > 1:
        st.markdown('<p class="sec-title">🕸️ Skills Radar</p>', unsafe_allow_html=True)
        fig_radar = go.Figure()
        colors_radar = ["#818cf8","#34d399","#fb923c","#f472b6"]
        for idx, r in enumerate(results_sorted):
            skill_scores = []
            for skill in skills:
                m = next((x for x in r["matched"] if x["skill"]==skill), None)
                skill_scores.append(m["score"] if m else 0)
            fig_radar.add_trace(go.Scatterpolar(
                r=skill_scores + [skill_scores[0]],
                theta=skills + [skills[0]],
                fill='toself',
                name=r["short"],
                line_color=colors_radar[idx % len(colors_radar)],
                fillcolor=colors_radar[idx % len(colors_radar)].replace("#","rgba(").replace("818cf8","129,140,248,0.15)").replace("34d399","52,211,153,0.15)").replace("fb923c","251,146,60,0.15)").replace("f472b6","244,114,182,0.15)")
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(20,20,40,0.8)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.1)", color="rgba(255,255,255,0.4)"),
                angularaxis=dict(color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.1)")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            legend=dict(font=dict(color="white")),
            height=450,
            margin=dict(t=30,b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")

    # ── COMPARISON TABLE ──
    st.markdown('<p class="sec-title">📋 Skills Comparison Table</p>', unsafe_allow_html=True)
    table_data = {"Skill": skills}
    for r in results_sorted:
        col_vals = []
        for skill in skills:
            m = next((x for x in r["matched"] if x["skill"]==skill), None)
            if m:
                tag = "🔑" if m["type"]=="keyword" else "🧠"
                col_vals.append(f"{tag} {m['score']}%")
            else:
                col_vals.append("❌")
        table_data[r["short"]] = col_vals
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── INDIVIDUAL CARDS ──
    st.markdown('<p class="sec-title">📄 Detailed Analysis</p>', unsafe_allow_html=True)

    for idx, r in enumerate(results_sorted):
        medal = ["🥇","🥈","🥉"][idx] if idx < 3 else "📄"
        pct = r["pct"]
        c = score_color(pct)

        with st.expander(f"{medal} {r['name']}  —  {r['score']}/{r['total']} matched  ({pct}%)", expanded=(idx==0)):

            col1, col2 = st.columns([1, 2])

            with col1:
                fig_donut = go.Figure(go.Pie(
                    values=[r["score"], r["total"]-r["score"]],
                    labels=["Matched","Missing"],
                    hole=0.68,
                    marker_colors=[c, "rgba(255,255,255,0.07)"],
                    textinfo="none",
                    hoverinfo="label+value"
                ))
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(font=dict(color="white", size=11), orientation="h", y=-0.15),
                    height=230, margin=dict(t=10,b=10,l=10,r=10),
                    annotations=[dict(
                        text=f"<b>{pct}%</b>",
                        x=0.5, y=0.5,
                        font=dict(size=24, color=c, family="Inter"),
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig_donut, use_container_width=True, key=f"donut_{idx}_{r['name'][:10]}")

                # Mini stats
                keyword_count = sum(1 for m in r["matched"] if m["type"]=="keyword")
                semantic_count = sum(1 for m in r["matched"] if m["type"]=="semantic")
                st.markdown(f"🔑 Keyword matches: **{keyword_count}**")
                st.markdown(f"🧠 Semantic matches: **{semantic_count}**")
                st.markdown(f"❌ Missing: **{len(r['missing'])}**")

            with col2:
                st.markdown(f"**✅ Matched Skills ({len(r['matched'])}):**")
                for m in r["matched"]:
                    tag = "skill-match-keyword" if m["type"]=="keyword" else "skill-match"
                    icon = "🔑" if m["type"]=="keyword" else "🧠"
                    st.markdown(
                        f'<div class="{tag}">{icon} <b>{m["skill"]}</b> — {m["score"]}% &nbsp;<span style="opacity:0.65;font-size:0.82rem">→ {m["reason"][:80]}</span></div>',
                        unsafe_allow_html=True
                    )

            if r["missing"]:
                st.markdown(f"**❌ Missing Skills ({len(r['missing'])}):**")
                miss_cols = st.columns(3)
                for j, m in enumerate(r["missing"]):
                    with miss_cols[j%3]:
                        st.markdown(f'<div class="skill-miss">✘ {m["skill"]} ({m["score"]}%)</div>', unsafe_allow_html=True)

            # Download
            txt = f"RESUME ANALYSIS REPORT\n{'='*50}\n"
            txt += f"Candidate: {r['name']}\nMatch Score: {r['score']}/{r['total']} ({pct}%)\n\n"
            txt += "MATCHED SKILLS:\n"
            for m in r["matched"]:
                txt += f"  {'[KEYWORD]' if m['type']=='keyword' else '[SEMANTIC]'} {m['skill']} — {m['score']}%\n  → {m['reason']}\n\n"
            txt += "\nMISSING SKILLS:\n"
            for m in r["missing"]:
                txt += f"  ✘ {m['skill']} ({m['score']}% similarity)\n"

            st.download_button(
                f"⬇️ Download {r['name']} Report",
                txt,
                file_name=f"{r['name']}_analysis.txt",
                key=f"dl_{idx}_{r['name'][:10]}"
            )
