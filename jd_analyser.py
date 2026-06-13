import streamlit as st
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import pandas as pd
 
# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Resume Radar",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─────────────────────────────────────────
# Custom CSS — Cyber/Terminal-inspired aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
 
* { font-family: 'Space Grotesk', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }
 
:root {
    --neon: #00ffc8;
    --neon-dim: rgba(0,255,200,0.15);
    --pink: #ff2d75;
    --pink-dim: rgba(255,45,117,0.15);
    --amber: #ffb800;
    --bg-deep: #060818;
    --bg-panel: #0d1024;
    --bg-card: #11152b;
    --border: rgba(0,255,200,0.18);
}
 
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0,255,200,0.06) 0%, transparent 35%),
        radial-gradient(circle at 85% 90%, rgba(255,45,117,0.05) 0%, transparent 40%),
        var(--bg-deep);
    background-attachment: fixed;
    min-height: 100vh;
}
 
/* Subtle scanline texture */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(255,255,255,0.012) 0px,
        rgba(255,255,255,0.012) 1px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 9999;
}
 
/* ── HERO ── */
.hero {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 36px 40px;
    margin-bottom: 28px;
    background: linear-gradient(135deg, rgba(0,255,200,0.04), rgba(255,45,117,0.03));
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--neon), var(--pink));
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    color: var(--neon);
    font-size: 0.78rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.85;
}
.hero h1 {
    color: #fff;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -1px;
}
.hero h1 span { color: var(--neon); }
.hero p {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}
 
/* ── Section Title ── */
.sec-title {
    color: #fff;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 32px 0 16px;
    padding-left: 14px;
    border-left: 3px solid var(--neon);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}
 
/* ── Stat Cards ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
}
.stat-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 24px; height: 24px;
    border-top: 2px solid var(--neon);
    border-right: 2px solid var(--neon);
    opacity: 0.5;
}
.stat-num {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--neon);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.stat-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.72rem;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'JetBrains Mono', monospace;
}
 
/* ── Winner banner ── */
.winner {
    border: 1px solid rgba(255,184,0,0.4);
    background: linear-gradient(135deg, rgba(255,184,0,0.1), rgba(255,45,117,0.06));
    border-radius: 4px;
    padding: 24px 32px;
    margin: 24px 0;
    position: relative;
}
.winner::before {
    content: '◆ TOP MATCH';
    position: absolute;
    top: -11px; left: 24px;
    background: var(--bg-deep);
    color: var(--amber);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    padding: 2px 10px;
}
.winner h2 {
    color: #fff;
    margin: 6px 0 0;
    font-size: 1.5rem;
    font-weight: 700;
}
.winner p {
    color: rgba(255,255,255,0.6);
    margin: 6px 0 0;
    font-size: 0.92rem;
    font-family: 'JetBrains Mono', monospace;
}
 
/* ── Skill pills ── */
.skill-match {
    background: var(--neon-dim);
    border: 1px solid rgba(0,255,200,0.35);
    border-left: 3px solid var(--neon);
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    color: #7af0d4;
    font-size: 0.88rem;
    font-weight: 500;
}
.skill-match-keyword {
    background: rgba(0,255,200,0.1);
    border: 1px solid rgba(0,255,200,0.5);
    border-left: 3px solid var(--neon);
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    color: #00ffc8;
    font-size: 0.88rem;
    font-weight: 700;
}
.skill-miss {
    background: var(--pink-dim);
    border: 1px solid rgba(255,45,117,0.35);
    border-left: 3px solid var(--pink);
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    color: #ff7aa8;
    font-size: 0.88rem;
    font-weight: 500;
}
 
/* ── Upload area ── */
section[data-testid="stFileUploadDropzone"] {
    background: var(--bg-panel) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}
 
/* ── Button ── */
.stButton > button {
    background: var(--bg-deep) !important;
    color: var(--neon) !important;
    border: 1px solid var(--neon) !important;
    border-radius: 3px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--neon) !important;
    color: var(--bg-deep) !important;
    box-shadow: 0 0 24px rgba(0,255,200,0.5) !important;
}
 
/* download buttons */
.stDownloadButton > button {
    background: var(--bg-panel) !important;
    color: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stDownloadButton > button:hover {
    border-color: var(--neon) !important;
    color: var(--neon) !important;
}
 
/* expander */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-panel) !important;
}
 
h1,h2,h3,h4 { color: #fff !important; }
p, li, label { color: rgba(255,255,255,0.65) !important; }
.stSlider label { color: rgba(255,255,255,0.65) !important; }
 
/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
 
hr { border-color: rgba(255,255,255,0.08) !important; }
 
/* dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────
# PDF text extract
# ─────────────────────────────────────────
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages).strip()
 
# ─────────────────────────────────────────
# HYBRID Skill Matching (Keyword + Semantic)
# ─────────────────────────────────────────
def analyze_resume(skills, resume_text, threshold):
    matched, missing = [], []
    resume_lower = resume_text.lower()
 
    sentences = [
        s.strip()
        for s in resume_text.replace("\n", ". ").split(".")
        if len(s.strip()) > 8
    ]
 
    if not sentences:
        return matched, missing
 
    for skill in skills:
        skill_lower = skill.lower()
 
        if skill_lower in resume_lower:
            matched.append({
                "skill": skill,
                "score": 100,
                "reason": "Exact keyword found in resume",
                "type": "keyword"
            })
            continue
 
        docs = [skill] + sentences
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(docs)
        scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
 
        best_idx = scores.argmax()
        best_score = float(scores[best_idx])
 
        if best_score >= threshold:
            matched.append({
                "skill": skill,
                "score": round(best_score * 100),
                "reason": sentences[best_idx][:120],
                "type": "semantic"
            })
        else:
            missing.append({
                "skill": skill,
                "score": round(best_score * 100)
            })
 
    return matched, missing
 
def score_color(pct):
    if pct >= 60: return "#00ffc8"
    elif pct >= 35: return "#ffb800"
    else: return "#ff2d75"
 
# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ▸ CONFIG")
 
    threshold = st.slider(
        "Semantic Match Sensitivity",
        0.05, 0.50, 0.15, 0.05,
        help="Higher = strict (only exact matches) | Lower = loose (more matches)"
    )
    st.caption("Keyword matches bypass sensitivity")
 
    st.markdown("---")
    st.markdown("### ▸ TARGET SKILLS")
    st.caption("One skill per line")
 
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
    st.markdown(f"`{len(skills)} skills loaded`")
 
    st.markdown("---")
    st.markdown("### ▸ LEGEND")
    st.markdown('<div class="skill-match-keyword">⬢ KEYWORD — exact match</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-match">◈ SEMANTIC — AI inferred</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-miss">✕ MISSING</div>', unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">// candidate intelligence system</div>
    <h1>Resume <span>Radar</span></h1>
    <p>upload --pdf · scan --skills · rank --candidates</p>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────
st.markdown('<p class="sec-title">01 // Upload Resumes</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop PDF resumes here", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
if uploaded:
    st.success(f"{len(uploaded)} resume(s) loaded into queue")
 
st.markdown("<br>", unsafe_allow_html=True)
go_btn = st.button("▸ Run Analysis")
 
# ─────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────
if go_btn:
    if not uploaded:
        st.error("No resumes in queue — upload PDFs first")
        st.stop()
    if len(skills) < 2:
        st.error("Add target skills in the sidebar")
        st.stop()
 
    results = []
    bar = st.progress(0, "Scanning...")
 
    for i, f in enumerate(uploaded):
        bar.progress((i+1)/len(uploaded), f"Scanning: {f.name}")
        text = extract_text(f)
        if text:
            matched, missing = analyze_resume(skills, text, threshold)
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
        st.error("Could not extract text from the resumes")
        st.stop()
 
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results_sorted[0]
 
    # ── STATS ──
    st.markdown('<p class="sec-title">02 // Overview</p>', unsafe_allow_html=True)
 
    cols = st.columns(len(results)+1)
    with cols[0]:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(results)}</div><div class="stat-label">Resumes Scanned</div></div>', unsafe_allow_html=True)
    for i, r in enumerate(results_sorted):
        with cols[i+1]:
            c = score_color(r["pct"])
            st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:{c}">{r["score"]}/{r["total"]}</div><div class="stat-label">{r["short"]}</div></div>', unsafe_allow_html=True)
 
    # ── WINNER ──
    st.markdown(f"""
    <div class="winner">
        <h2>{best['name']}</h2>
        <p>{best['score']}/{best['total']} skills matched  ·  {best['pct']}% match score</p>
    </div>""", unsafe_allow_html=True)
 
    # ── BAR CHART ──
    st.markdown('<p class="sec-title">03 // Candidate Comparison</p>', unsafe_allow_html=True)
 
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[r["short"] for r in results_sorted],
        y=[r["pct"] for r in results_sorted],
        marker=dict(
            color=[score_color(r["pct"]) for r in results_sorted],
            line=dict(width=1, color="rgba(255,255,255,0.2)")
        ),
        text=[f"{r['pct']}%<br>({r['score']}/{r['total']})" for r in results_sorted],
        textposition="outside",
        textfont=dict(color="white", size=13, family="JetBrains Mono"),
        width=0.5
    ))
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="JetBrains Mono"),
        yaxis=dict(range=[0,115], gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.4)", title="Match %"),
        xaxis=dict(color="rgba(255,255,255,0.6)"),
        showlegend=False, height=340,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="bar_comparison")
 
    # ── RADAR CHART ──
    if len(results) > 1:
        st.markdown('<p class="sec-title">04 // Skills Radar</p>', unsafe_allow_html=True)
        fig_radar = go.Figure()
        colors_radar = ["#00ffc8","#ff2d75","#ffb800","#7c5cff"]
        fills_radar = ["rgba(0,255,200,0.12)","rgba(255,45,117,0.10)","rgba(255,184,0,0.10)","rgba(124,92,255,0.10)"]
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
                fillcolor=fills_radar[idx % len(fills_radar)]
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(13,16,36,0.7)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.08)", color="rgba(255,255,255,0.35)"),
                angularaxis=dict(color="rgba(255,255,255,0.55)", gridcolor="rgba(255,255,255,0.08)")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="JetBrains Mono"),
            legend=dict(font=dict(color="white")),
            height=450,
            margin=dict(t=30,b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
 
    # ── COMPARISON TABLE ──
    st.markdown('<p class="sec-title">05 // Skills Matrix</p>', unsafe_allow_html=True)
    table_data = {"Skill": skills}
    for r in results_sorted:
        col_vals = []
        for skill in skills:
            m = next((x for x in r["matched"] if x["skill"]==skill), None)
            if m:
                tag = "⬢" if m["type"]=="keyword" else "◈"
                col_vals.append(f"{tag} {m['score']}%")
            else:
                col_vals.append("✕")
        table_data[r["short"]] = col_vals
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
 
    # ── INDIVIDUAL CARDS ──
    st.markdown('<p class="sec-title">06 // Detailed Analysis</p>', unsafe_allow_html=True)
 
    for idx, r in enumerate(results_sorted):
        rank = f"#{idx+1:02d}"
        pct = r["pct"]
        c = score_color(pct)
 
        with st.expander(f"{rank}  {r['name']}  —  {r['score']}/{r['total']} matched  ({pct}%)", expanded=(idx==0)):
 
            col1, col2 = st.columns([1, 2])
 
            with col1:
                fig_donut = go.Figure(go.Pie(
                    values=[r["score"], r["total"]-r["score"]],
                    labels=["Matched","Missing"],
                    hole=0.68,
                    marker_colors=[c, "rgba(255,255,255,0.06)"],
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
                        font=dict(size=24, color=c, family="JetBrains Mono"),
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig_donut, use_container_width=True, key=f"donut_{idx}_{r['name'][:10]}")
 
                keyword_count = sum(1 for m in r["matched"] if m["type"]=="keyword")
                semantic_count = sum(1 for m in r["matched"] if m["type"]=="semantic")
                st.markdown(f"⬢ Keyword matches: **{keyword_count}**")
                st.markdown(f"◈ Semantic matches: **{semantic_count}**")
                st.markdown(f"✕ Missing: **{len(r['missing'])}**")
 
            with col2:
                st.markdown(f"**Matched Skills ({len(r['matched'])}):**")
                for m in r["matched"]:
                    tag = "skill-match-keyword" if m["type"]=="keyword" else "skill-match"
                    icon = "⬢" if m["type"]=="keyword" else "◈"
                    st.markdown(
                        f'<div class="{tag}">{icon} <b>{m["skill"]}</b> — {m["score"]}% &nbsp;<span style="opacity:0.6;font-size:0.82rem">→ {m["reason"][:80]}</span></div>',
                        unsafe_allow_html=True
                    )
 
            if r["missing"]:
                st.markdown(f"**Missing Skills ({len(r['missing'])}):**")
                miss_cols = st.columns(3)
                for j, m in enumerate(r["missing"]):
                    with miss_cols[j%3]:
                        st.markdown(f'<div class="skill-miss">✕ {m["skill"]} ({m["score"]}%)</div>', unsafe_allow_html=True)
 
            txt = f"RESUME ANALYSIS REPORT\n{'='*50}\n"
            txt += f"Candidate: {r['name']}\nMatch Score: {r['score']}/{r['total']} ({pct}%)\n\n"
            txt += "MATCHED SKILLS:\n"
            for m in r["matched"]:
                txt += f"  {'[KEYWORD]' if m['type']=='keyword' else '[SEMANTIC]'} {m['skill']} — {m['score']}%\n  → {m['reason']}\n\n"
            txt += "\nMISSING SKILLS:\n"
            for m in r["missing"]:
                txt += f"  ✕ {m['skill']} ({m['score']}% similarity)\n"
 
            st.download_button(
                f"⬇ Download {r['name']} Report",
                txt,
                file_name=f"{r['name']}_analysis.txt",
                key=f"dl_{idx}_{r['name'][:10]}"
            )
