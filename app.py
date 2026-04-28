import streamlit as st, pandas as pd, PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Skill Gap Recommendation Engine",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "step" not in st.session_state:
    st.session_state.step = 1
if "name" not in st.session_state:
    st.session_state.name = ""

# =====================================================
# DARK CHARCOAL + TEAL UI
# =====================================================
st.markdown("""
<style>

/* APP */
.stApp{
    background:#0F172A;
    color:#F8FAFC;
}

/* MAIN */
.block-container{
    max-width:1200px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#020617,#111827);
}
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* TITLES */
h1,h2,h3,h4{
    color:#F8FAFC !important;
    font-weight:800;
}

/* TEXT */
p,label,span{
    color:#CBD5E1 !important;
}

/* INPUT */
input{
    background:#111827 !important;
    color:#F8FAFC !important;
    border:1px solid #334155 !important;
    border-radius:14px !important;
}

/* SELECT */
div[data-baseweb="select"] > div{
    background:#111827 !important;
    color:#F8FAFC !important;
    border:1px solid #334155 !important;
    border-radius:14px !important;
}
div[data-baseweb="select"] span{
    color:#F8FAFC !important;
}

/* BUTTON */
.stButton > button{
    width:100%;
    border:none;
    border-radius:14px;
    padding:.9rem;
    font-size:18px;
    font-weight:700;
    color:white !important;
    background:linear-gradient(90deg,#1F2937,#14B8A6);
}

.stButton > button:hover{
    background:linear-gradient(90deg,#111827,#0F766E);
}

/* CARDS */
.card,
[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #334155;
    border-radius:18px;
    padding:1rem;
    box-shadow:0 8px 22px rgba(0,0,0,.25);
}

/* METRIC VALUES */
[data-testid="stMetricValue"]{
    color:#F8FAFC !important;
    font-size:34px !important;
    font-weight:800 !important;
}

/* METRIC LABEL */
[data-testid="metric-container"] label{
    color:#94A3B8 !important;
}

/* STEP BOX */
.stepbox{
    background:#111827;
    border:1px solid #334155;
    border-radius:14px;
    padding:12px;
    text-align:center;
    color:#14B8A6;
    font-weight:700;
    margin-bottom:18px;
}

/* PROGRESS */
.stProgress > div > div > div > div{
    background:linear-gradient(90deg,#1F2937,#14B8A6);
}

/* ALERT */
div[data-testid="stAlert"]{
    background:#111827;
    border:1px solid #334155;
    border-radius:14px;
    color:#F8FAFC !important;
}

/* HIGHLIGHT */
.gold{
    color:#14B8A6 !important;
    font-weight:700;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{
    background:#111827 !important;
    border:1px solid #334155 !important;
    border-radius:16px !important;
    padding:10px !important;
}

/* FILE UPLOADER TEXT */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p{
    color:#F8FAFC !important;
}

/* FILE BUTTON */
[data-testid="stFileUploader"] button{
    background:#14B8A6 !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    font-weight:700 !important;
}

/* FILE BUTTON HOVER */
[data-testid="stFileUploader"] button:hover{
    background:#0F766E !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA
# =====================================================
jobs = pd.read_csv("engineering_job_roles_max.csv")
courses = pd.read_csv("engineering_courses_max.csv")

# =====================================================
# PDF FUNCTION
# =====================================================
def pdf_text(file):
    if not file:
        return ""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += (page.extract_text() or "") + " "
    except:
        pass
    return text.lower()

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style='text-align:center;'>📈 Skill Gap Recommendation Engine</h1>
<p style='text-align:center;font-size:18px;'>
Premium AI-powered platform for career readiness & intelligent skill recommendations
</p>
""", unsafe_allow_html=True)

st.markdown(
f"<div class='stepbox'>Step {st.session_state.step} of 4</div>",
unsafe_allow_html=True
)

# =====================================================
# STEP 1
# =====================================================
if st.session_state.step == 1:

    st.subheader("👋 Welcome")

    name = st.text_input("Enter Your Full Name")

    if st.button("Continue"):

        if name.strip():
            st.session_state.name = name
            st.session_state.step = 2
            st.rerun()

# =====================================================
# STEP 2
# =====================================================
elif st.session_state.step == 2:

    st.subheader(f"🎯 Hello, {st.session_state.name}")

    domain = st.selectbox("Select Domain", jobs["Domain"].unique())

    roles = jobs[jobs["Domain"] == domain]["Job_Role"].unique()

    role = st.selectbox("Select Target Role", roles)

    skills = sorted(set(
        s.strip()
        for txt in jobs[jobs["Domain"] == domain]["Required_Skills"]
        for s in txt.split(",")
    ))

    selected = st.multiselect("Select Your Current Skills", skills)

    if st.button("Next"):

        st.session_state.domain = domain
        st.session_state.role = role
        st.session_state.skills = selected
        st.session_state.step = 3
        st.rerun()

# =====================================================
# STEP 3
# =====================================================
elif st.session_state.step == 3:

    st.subheader("📂 Upload Documents")

    resume = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

    certs = st.file_uploader(
        "Upload Certificates",
        type=["pdf","png","jpg","jpeg"],
        accept_multiple_files=True
    )

    c1,c2 = st.columns(2)

    with c1:
        if st.button("⬅ Back"):
            st.session_state.step = 2
            st.rerun()

    with c2:
        if st.button("Analyze Profile"):
            st.session_state.resume = resume
            st.session_state.step = 4
            st.rerun()

# =====================================================
# STEP 4
# =====================================================
else:

    st.subheader("📊 Analysis Dashboard")

    role = st.session_state.role
    skills = ",".join(st.session_state.skills).lower()
    resume = pdf_text(st.session_state.resume)

    req = jobs[jobs["Job_Role"] == role].iloc[0]["Required_Skills"].lower()
    req_list = [x.strip() for x in req.split(",")]

    combined = skills + " " + resume

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([combined, req])

    score = round(
        cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100,
        2
    )

    matched = [s for s in req_list if s in combined]
    missing = [s for s in req_list if s not in combined]

    # METRICS
    c1,c2,c3 = st.columns(3)

    c1.metric("Match Score", f"{score}%")
    c2.metric("Detected Skills", len(matched))
    c3.metric("Skill Gaps", len(missing))

    st.progress(int(score))

    # CARDS
    a,b = st.columns(2)

    with a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ✅ Detected Skills")
        st.write(", ".join(matched) if matched else "None")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ❌ Skill Gaps")
        st.write(", ".join(missing) if missing else "None")
        st.markdown("</div>", unsafe_allow_html=True)

    # COURSES
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "### <span class='gold'>⭐ Recommended Courses</span>",
        unsafe_allow_html=True
    )

    for skill in missing:
        r = courses[courses["Skill"].str.lower() == skill]
        if not r.empty:
            st.write("•", r.iloc[0]["Course_Name"])

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Start Again"):
        st.session_state.step = 1
        st.rerun()