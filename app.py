import streamlit as st
st.set_page_config(page_title="UNDERGRADE DOSSIER", page_icon=":page_facing_up:")

# ==============================
# Standard libs
# ==============================
import os
import io
import re
import uuid
import time
import random
import socket
import secrets
import datetime
import platform
import base64
import warnings
import streamlit as st
import base64

# ==============================
# Third-party libs
# ==============================
try:
    import bcrypt  # password hashing
except Exception:
    bcrypt = None

import pandas as pd
import pymysql
import plotly.express as px
from pdfminer.high_level import extract_text
import geocoder

# Optional: spaCy (we fallback if not available)
try:
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm")
    except Exception:
        NLP = None
except Exception:
    NLP = None

# Optional: NLTK for tokenization (handled softly)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==============================
# (Optional) Selenium imports (for job scraping)
# If you don't want LinkedIn scraping, you can remove this entire section
# ==============================
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

warnings.filterwarnings("ignore")

# ==============================
# Constants, templates, data
# ==============================
SKILLS_MASTER = [
    "Python","Java","C++","JavaScript","HTML","CSS","SQL",
    "Machine Learning","Data Analysis","React","Node.js",
    "Angular","Vue.js","Docker","Kubernetes","AWS","Azure",
    "GCP","Git","Agile","TensorFlow","Keras","Django","Flask",
    "MySQL","PostgreSQL","MongoDB","Pandas","NumPy","scikit-learn",
    "Kali Linux","Metasploit","SEO","BeautifulSoup","NLP","Gensim"
]

# Courses (sample lists)
ds_course = [
    "Coursera – IBM Data Science Professional Certificate",
    "Coursera – Machine Learning by Andrew Ng",
    "edX – Data Science MicroMasters",
    "Udacity – Data Analyst Nanodegree",
    "Kaggle – Intro to Machine Learning"
]
web_course = [
    "Udemy – The Complete 202x Web Development Bootcamp",
    "freeCodeCamp – Responsive Web Design",
    "Coursera – HTML, CSS, and JS for Web Developers",
    "Frontend Masters – Complete Intro to React",
    "Scrimba – Learn React for free"
]
android_course = [
    "Udacity – Developing Android Apps",
    "Coursera – Android App Development",
    "Google – Android Basics",
    "Udemy – Kotlin for Android",
    "Kodeco – Android with Kotlin"
]
ios_course = [
    "Stanford CS193p – Developing Apps for iOS",
    "Udemy – iOS & Swift - The Complete Bootcamp",
    "Hacking with Swift – 100 Days of Swift",
    "Coursera – iOS App Development",
    "Kodeco – iOS with Swift"
]
uiux_course = [
    "Coursera – Google UX Design Professional Certificate",
    "Interaction Design Foundation – UX Courses",
    "Udemy – UI/UX Design with Figma",
    "Skillshare – UX Fundamentals",
    "DesignLab – UX Academy (intro)"
]

FIELD_TO_COURSES = {
    "Data Science": ds_course,
    "Web Development": web_course,
    "Android Development": android_course,
    "iOS Development": ios_course,
    "UI/UX Design": uiux_course
}

# ==============================
# Utilities
# ==============================

def get_base64_image(image_path):
    """Convert local image to base64 for CSS embedding."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None
def generate_session_token():
    return secrets.token_hex(16)

def generate_unique_id():
    return str(uuid.uuid4())

def get_geolocation():
    try:
        g = geocoder.ip('me')
        return (g.latlng, g.city, g.state, g.country)
    except Exception:
        return (None, None, None, None)

def get_device_info():
    try:
        return {
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "hostname": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()}",
        }
    except Exception:
        return {
            "ip_address": "Unknown",
            "hostname": "Unknown",
            "os": "Unknown",
        }

def get_database_connection():
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'Gopi1431@@@@'),  # default as in your script
            database=os.getenv('DB_NAME', 'resume_analyzer'),
            autocommit=False
        )
        return connection
    except pymysql.MySQLError as e:
        st.error(f"Error connecting to the database: {e}")
        return None

def init_database():
    connection = get_database_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                # students table for login/registration
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS students (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        registration_number VARCHAR(50) UNIQUE,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        cgpa FLOAT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # user analysis table (history of resume analyses)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) NOT NULL,
                        resume_score INT NOT NULL,
                        recommended_field VARCHAR(255),
                        experience_level VARCHAR(50),
                        timestamp DATETIME NOT NULL
                    )
                """)
                # feedback table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) NOT NULL,
                        rating INT NOT NULL,
                        comments TEXT NOT NULL,
                        timestamp DATETIME NOT NULL
                    )
                """)
                # technical events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS technical_events (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        event_name VARCHAR(255) NOT NULL,
                        event_date DATE NOT NULL,
                        description TEXT,
                        venue VARCHAR(255),
                        organizer VARCHAR(255),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # exam patterns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS exam_patterns (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        exam_name VARCHAR(255) NOT NULL,
                        pattern_details TEXT NOT NULL,
                        syllabus TEXT,
                        duration VARCHAR(100),
                        total_marks INT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # companies table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS companies (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        company_name VARCHAR(255) NOT NULL,
                        industry VARCHAR(255),
                        required_skills TEXT,
                        min_cgpa FLOAT,
                        package_offered VARCHAR(100),
                        job_location VARCHAR(255),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            connection.commit()
        except Exception as e:
            st.error(f"Error initializing database: {e}")
        finally:
            connection.close()

# ==============================
# Resume parsing & analysis
# ==============================
def parse_resume_pdf(path_or_filelike) -> dict:
    """Extracts raw text and quick heuristics from a PDF."""
    try:
        text = extract_text(path_or_filelike)
    except Exception:
        text = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Very light heuristics
    name = lines[0] if lines else ""
    email = ""
    phone = ""
    degree = ""
    for ln in lines:
        if not email:
            m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", ln)
            if m:
                email = m.group(0)
        if not phone:
            m = re.search(r"(\+?\d[\d \-]{7,}\d)", ln)
            if m:
                phone = m.group(0)
        if not degree:
            low = ln.lower()
            if any(k in low for k in ["bachelor", "btech", "b.e", "be ", "master", "mtech", "m.e", "phd", "doctorate"]):
                degree = ln

    # naive skill tokens
    tokens = re.findall(r"[A-Za-z\+#\.]{2,}", text)
    skills = set()
    for token in tokens:
        for sk in SKILLS_MASTER:
            if token.lower() == sk.lower().replace(" ", "") or token.lower() == sk.lower():
                skills.add(sk)

    return {
        "raw_text": text,
        "name": name or "Not found",
        "email": email or "Not found",
        "mobile_number": phone or "Not found",
        "degree": degree or "Not found",
        "skills": sorted(skills),
        "total_experience": 0,  # placeholder unless you extract
        "projects": [],
        "certifications": [],
        "achievements": [],
        "summary": ""
    }

@st.cache_data(show_spinner=False)
def analyze_resume(resume_text: str) -> dict:
    """Try spaCy NER first; fallback to heuristics."""
    result = {
        "name": None,
        "email": None,
        "mobile_number": None,
        "skills": [],
        "education": [],
        "experience": [],
        "degree": None,
        "resume_score": 0
    }

    # Email + phone via regex as baseline
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text)
    if m: result["email"] = m.group(0)
    m = re.search(r"(\+?\d[\d \-]{7,}\d)", resume_text)
    if m: result["mobile_number"] = m.group(0)

    # Degree guess
    deg = re.search(r"(Bachelor|BTech|B\.?E\.?|Master|MTech|M\.?E\.?|PhD|Doctorate)[^\n]*", resume_text, re.IGNORECASE)
    result["degree"] = deg.group(0) if deg else None

    # Skills
    found = set()
    for sk in SKILLS_MASTER:
        pat = r"\b" + re.escape(sk) + r"\b"
        if re.search(pat, resume_text, re.IGNORECASE):
            found.add(sk)
    result["skills"] = sorted(found)

    # Optional spaCy pass (name, maybe better parsing)
    if NLP is not None:
        try:
            doc = NLP(re.sub(r"\s+", " ", resume_text))
            for ent in doc.ents:
                if ent.label_ == "PERSON" and not result["name"]:
                    result["name"] = ent.text
            # You could enrich more here (ORG as college, etc.)
        except Exception:
            pass

    # Score is % of master list covered (cap to 100)
    if SKILLS_MASTER:
        score = round(100 * len(found) / len(SKILLS_MASTER), 2)
    else:
        score = 0
    result["resume_score"] = score

    # Fallback names if missing
    if not result["name"]:
        # Try a simple guess: first non-empty line as name-like
        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
        if lines:
            result["name"] = lines[0]

    return result

# ==============================
# Recommendations & scoring
# ==============================
def recommend_skills(skills: list) -> list:
    remaining = list(set(SKILLS_MASTER) - set(skills))
    random.shuffle(remaining)
    return remaining[:5]

def recommend_field(skills: list) -> str:
    fields = {
        "Data Science": {"Python","Machine Learning","Data Analysis","SQL","Pandas","NumPy","scikit-learn","TensorFlow"},
        "Web Development": {"JavaScript","HTML","CSS","React","Node.js","Angular","Vue.js"},
        "Android Development": {"Java","Kotlin","Android SDK"},
        "iOS Development": {"Swift","Objective-C","iOS"},
        "UI/UX Design": {"Figma","Adobe XD","Sketch","User Research"}
    }
    best = "General Software Development"
    best_match = 0
    sset = set(skills)
    for field, reqs in fields.items():
        score = len(sset & reqs)
        if score > best_match:
            best_match = score
            best = field
    return best

def recommend_courses(field: str) -> list:
    return FIELD_TO_COURSES.get(field, ds_course)

def get_resume_score_breakdown(resume_data: dict) -> dict:
    score_breakdown = {
        "Contact Information": 0,
        "Education": 0,
        "Skills": 0,
        "Experience": 0,
        "Projects": 0,
        "Certifications": 0,
        "Summary/Objective": 0,
        "Achievements": 0,
        "Formatting": 8,  # baseline for now
        "Keywords": 7     # baseline for now
    }
    if resume_data.get('name'): score_breakdown["Contact Information"] += 3
    if resume_data.get('email'): score_breakdown["Contact Information"] += 3
    if resume_data.get('mobile_number'): score_breakdown["Contact Information"] += 4

    if resume_data.get('degree') and resume_data['degree'] != "Not found":
        score_breakdown["Education"] += 5
    if resume_data.get('college_name'):
        score_breakdown["Education"] += 5

    skills = resume_data.get('skills', [])
    score_breakdown["Skills"] = min(len(skills), 10)

    experience = resume_data.get('total_experience', 0) or 0
    score_breakdown["Experience"] = min(int(experience) * 2, 10)

    projects = resume_data.get('projects', [])
    score_breakdown["Projects"] = min(len(projects) * 2, 10)

    certifications = resume_data.get('certifications', [])
    score_breakdown["Certifications"] = min(len(certifications) * 2, 10)

    if resume_data.get('summary'):
        score_breakdown["Summary/Objective"] = 10

    achievements = resume_data.get('achievements', [])
    score_breakdown["Achievements"] = min(len(achievements) * 2, 10)

    return score_breakdown

def calculate_resume_score(resume_data: dict) -> int:
    return sum(get_resume_score_breakdown(resume_data).values())

# ==============================
# Report PDFs (ReportLab)
# ==============================
def generate_pdf_report(resume_data, resume_score, score_breakdown, recommended_skills, recommended_field, recommended_courses):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Resume Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Basic Information", styles['Heading2']))
    basic_info = [
        ["Name", resume_data.get('name', 'Not found')],
        ["Email", resume_data.get('email', 'Not found')],
        ["Phone", resume_data.get('mobile_number', 'Not found')],
        ["Degree", resume_data.get('degree', 'Not found')]
    ]
    t = Table(basic_info, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Resume Score: {resume_score}/100", styles['Heading2']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Score Breakdown", styles['Heading2']))
    rows = [[k, f"{v}/10"] for k,v in score_breakdown.items()]
    tb = Table(rows, colWidths=[200, 100])
    tb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(tb)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Recommendations", styles['Heading2']))
    elements.append(Paragraph(f"Recommended Field: {recommended_field}", styles['Normal']))
    elements.append(Paragraph("Recommended Skills:", styles['Normal']))
    for s in recommended_skills:
        elements.append(Paragraph(f"• {s}", styles['Normal']))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Recommended Courses:", styles['Normal']))
    for c in recommended_courses[:5]:
        elements.append(Paragraph(f"• {c}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_dossier_guide_pdf(steps: list):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    els = []
    els.append(Paragraph("Undergrad Dossier – Step-by-Step Guide", styles["Title"]))
    els.append(Spacer(1, 12))

    flow = ListFlowable(
        [ListItem(Paragraph(step, styles["Normal"])) for step in steps],
        bulletType='1'
    )
    els.append(flow)
    doc.build(els)
    buf.seek(0)
    return buf

# ==============================
# Data persistence helpers
# ==============================
def store_user_data(user_data: dict) -> bool:
    connection = get_database_connection()
    if not connection:
        return False
    try:
        with connection.cursor() as cursor:
            sql = """INSERT INTO users (name, email, resume_score, recommended_field, experience_level, timestamp) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                user_data['name'],
                user_data['email'],
                int(user_data['resume_score']),
                user_data['recommended_field'],
                user_data['experience_level'],
                datetime.datetime.now(),
            ))
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Error storing user data: {e}")
        return False
    finally:
        connection.close()

def store_feedback(feedback_data: dict) -> bool:
    connection = get_database_connection()
    if not connection:
        return False
    try:
        with connection.cursor() as cursor:
            sql = """INSERT INTO feedback (name, email, rating, comments, timestamp) VALUES (%s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                feedback_data['name'],
                feedback_data['email'],
                int(feedback_data['rating']),
                feedback_data['comments'],
                feedback_data['timestamp']
            ))
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Error storing feedback: {e}")
        return False
    finally:
        connection.close()

def get_user_data() -> pd.DataFrame:
    connection = get_database_connection()
    if not connection:
        return pd.DataFrame()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users")
            result = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            return pd.DataFrame(result, columns=cols)
    except Exception as e:
        st.error(f"Error fetching user data: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

def get_feedback_data() -> pd.DataFrame:
    connection = get_database_connection()
    if not connection:
        return pd.DataFrame()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM feedback")
            result = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            return pd.DataFrame(result, columns=cols)
    except Exception as e:
        st.error(f"Error fetching feedback data: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

# ==============================
# Pages
# ==============================
def auth_page():
    image_path = "assets/WhatsApp Image 2025-11-08 at 21.28.59.jpeg"
    if os.path.exists(image_path):
        base64_image = get_base64_image(image_path)
    else:
        base64_image = ""

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.4);
        z-index: -1;
    }}
    .student-login-container {{
        backdrop-filter: blur(8px);
        background: rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        margin-top: 2rem;
    }}
    </style>
    """, unsafe_allow_html=True)


    if bcrypt is None:
        st.error("`bcrypt` is not installed. Please install it: `pip install bcrypt`")
        st.stop()

    # Display student login image
    student_image_path = "assets/WhatsApp Image 2025-11-08 at 21.28.59.jpeg"
    # if os.path.exists(student_image_path):
    #     col1, col2, col3 = st.columns([1, 2, 1])
    #     with col2:
    #         st.image(student_image_path, width=300, caption="Parul University Student Portal")
    # else:
    #     st.markdown('<div class="parul-student-logo">🎓</div>', unsafe_allow_html=True)
    st.markdown('<div class="student-header"><h1>👨‍🎓 Student Portal</h1><h3>Parul University</h3></div>', unsafe_allow_html=True)

    connection = get_database_connection()
    if not connection:
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="student-login-container">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### 🚪 Student Login")
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            if st.button("🚀 Login", use_container_width=True):
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT id, name, password FROM students WHERE email=%s", (email,))
                        row = cursor.fetchone()
                    if row and bcrypt.checkpw(password.encode(), row[2].encode()):
                        st.session_state["student_id"] = row[0]
                        st.session_state["student_name"] = row[1]
                        st.success("🎉 Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password.")
                except Exception as e:
                    st.error(f"Login error: {e}")
        
        with tab2:
            st.markdown("### 📋 Student Registration")
            reg_no = st.text_input("🆔 Registration Number", placeholder="Enter registration number")
            name = st.text_input("👤 Full Name", placeholder="Enter your full name")
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
            cgpa = st.number_input("📊 CGPA", min_value=0.0, max_value=10.0, step=0.1, help="Enter your current CGPA")
            
            if st.button("📝 Register", use_container_width=True):
                if reg_no and name and email and password:
                    try:
                        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO students (registration_number, name, email, password, cgpa)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (reg_no, name, email, hashed_pw, cgpa))
                        connection.commit()
                        st.success("🎉 Registration successful! Please log in.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Please fill all fields.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem; color: #666;">
            <small>🎓 Empowering Students | Parul University</small>
        </div>
        """, unsafe_allow_html=True)
    
    connection.close()

def user_page():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_unique_id()

    st.title("UNDERGRADE DOSSIER")
    st.write("Upload your resume and get insights!")

    uploaded_file = st.file_uploader("Choose your resume (PDF)", type="pdf")

    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing your resume..."):
                # Extract text directly from file-like
                resume_text = extract_text(uploaded_file) or ""
                # Parse and analyze
                quick = parse_resume_pdf(io.BytesIO(uploaded_file.getvalue()))
                enriched = analyze_resume(resume_text)
                # Merge enriched into quick (quick keeps nice base fields)
                for k, v in enriched.items():
                    if k not in quick or not quick[k] or quick[k] == "Not found":
                        quick[k] = v if v else quick.get(k)

                # Compute final scoring/recommendations
                level = "Fresher" if (quick.get('total_experience', 0) or 0) == 0 else ("Intermediate" if (quick.get('total_experience', 0) or 0) < 3 else "Experienced")
                st.subheader("Basic Information")
                st.write(f"Name: {quick.get('name','Not found')}")
                st.write(f"Email: {quick.get('email','Not found')}")
                st.write(f"Phone: {quick.get('mobile_number','Not found')}")
                st.write(f"Degree: {quick.get('degree','Not found')}")
                st.write(f"Experience Level: {level}")

                st.subheader("Skills")
                skills = quick.get('skills', [])
                st.write(", ".join(skills) if skills else "Not found")

                st.subheader("Skills Recommendation")
                rec_skills = recommend_skills(skills)
                st.write(", ".join(rec_skills) if rec_skills else "—")

                st.subheader("Field Recommendation")
                field = recommend_field(skills)
                st.write(f"Based on your skills, we recommend exploring: **{field}**")

                st.subheader("Course Recommendation")
                courses = recommend_courses(field)
                for c in courses[:5]:
                    st.write(f"- {c}")

                st.subheader("Resume Score")
                # If analyze_resume gave a score, use that; else compute from breakdown
                score_from_nlp = enriched.get("resume_score", 0) or 0
                breakdown = get_resume_score_breakdown(quick)
                score_from_breakdown = sum(breakdown.values())
                final_score = int((score_from_nlp + score_from_breakdown) / 2) if score_from_nlp else score_from_breakdown
                st.write(f"Your resume score: {final_score}/100")

                st.subheader("Resume Score Breakdown")
                for cat, sc in breakdown.items():
                    st.write(f"{cat}: {sc}/10")

                # Save to DB (session-based but persisted for analytics)
                data_to_save = {
                    "name": quick.get('name', 'Not found'),
                    "email": quick.get('email', 'Not found'),
                    "resume_score": final_score,
                    "recommended_field": field,
                    "experience_level": level,
                }
                if store_user_data(data_to_save):
                    st.success("Your resume analysis has been saved.")

                # PDF report
                pdf_buffer = generate_pdf_report(quick, final_score, breakdown, rec_skills, field, courses)
                st.download_button(
                    label="Download Resume Analysis Report (PDF)",
                    data=pdf_buffer,
                    file_name="resume_analysis_report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"An error occurred while processing your resume: {e}")
            st.error("Please make sure you've uploaded a valid PDF file and try again.")

    if st.button("Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def find_jobs_page():
    st.title("Find Jobs")
    st.write("Search for jobs based on job title and location.")

    job_title = st.text_input("Job Title")
    location = st.text_input("Location")

    if st.button("Search Jobs"):
        with st.spinner("Searching for jobs..."):
            jobs = scrape_linkedin_jobs(job_title, location)
        if jobs:
            display_job_results(jobs)
        else:
            st.write("No jobs found. Try different search terms.")

def scrape_linkedin_jobs(job_title, location):
    url = f"https://www.linkedin.com/jobs/search/?keywords={job_title}&location={location}"
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = None
    results = []
    try:
        driver = webdriver.Firefox(options=options)
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        job_cards = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "base-card")))
        for card in job_cards[:10]:
            try:
                title_element = card.find_element(By.CLASS_NAME, "base-card__full-link")
                company_element = card.find_element(By.CLASS_NAME, "job-card-container__company-name")
                location_element = card.find_element(By.CLASS_NAME, "job-card-container__metadata-item")
                link_element = card.find_element(By.CLASS_NAME, "base-card__full-link")

                title = title_element.text
                company = company_element.text
                job_location = location_element.text
                link = link_element.get_attribute("href")

                results.append({
                    "title": title,
                    "company": company,
                    "location": job_location,
                    "link": link
                })
            except Exception:
                continue
    except Exception as e:
        st.error(f"Error initializing web driver or scraping: {e}")
    finally:
        if driver:
            driver.quit()
    return results

def display_job_results(jobs):
    st.subheader("Job Results")
    for job in jobs:
        st.write(f"**{job['title']}**")
        st.write(f"Company: {job['company']}")
        st.write(f"Location: {job['location']}")
        st.write(f"[Link]({job['link']})")
        st.write("---")

    if jobs:
        df = pd.DataFrame(jobs)
        csv = df.to_csv(index=False).encode()
        st.download_button(label="Download Results as CSV", data=csv, file_name='job_results.csv', mime='text/csv')

def feedback_page():
    st.title("Feedback")
    st.write("We'd love to hear your thoughts!")
    name = st.text_input("Name", value=st.session_state.get("student_name",""))
    email = st.text_input("Email")
    rating = st.slider("Rating", 1, 5, 3)
    comments = st.text_area("Comments")

    if st.button("Submit Feedback"):
        if name and email and comments:
            ok = store_feedback({
                "name": name,
                "email": email,
                "rating": rating,
                "comments": comments,
                "timestamp": datetime.datetime.now()
            })
            if ok:
                st.success("Thank you for your feedback!")
            else:
                st.error("There was an error submitting your feedback. Please try again.")
        else:
            st.warning("Please fill out all fields before submitting.")

def about_page():
    st.title("About UNDERGRADE DOSSIER")
    st.write("Welcome! This tool helps you improve your resume and find suitable job opportunities.")
    st.subheader("Features")
    st.markdown("""
- **Resume Analysis**: Insights on strengths and weaknesses  
- **Skills Recommendation**: Skills to enhance your profile  
- **Job Field Recommendation**: Suggested fields based on your skills  
- **Course Recommendations**: Courses to upskill  
- **Job Search**: Quick scrape-based job listings  
- **Resume Score**: Quantitative assessment  
- **Dossier Guide**: Step-by-step PDF flow for your dossier  
""")
    st.subheader("Privacy")
    st.write("We value your privacy. Your resume data is used only for analysis. Admin dashboard aggregates anonymized analytics.")

def admin_login():
    # Convert local image to base64
    bg_image = get_base64_image("assets/image.png")

    # Add CSS with base64 image
    if bg_image:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: -1;
        }}
        .login-container {{
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.15);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            margin: 2rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .login-header {{
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }}
        .login-header h1, .login-header h3 {{
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            color: white !important;
        }}
        .parul-logo {{
            text-align: center;
            font-size: 3rem;
            color: #FF6B35;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        .stTextInput > div > div > input {{
            background-color: rgba(255,255,255,0.2);
            border: 2px solid #FF6B35;
            border-radius: 10px;
            color: white;
        }}
        .stTextInput > div > div > input::placeholder {{
            color: rgba(255,255,255,0.7);
        }}
        .stButton > button {{
            background: linear-gradient(45deg, #FF6B35, #F7931E);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            width: 100%;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="parul-logo">🏛️</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-header"><h1>🛡️ Admin Portal</h1><h3>Parul University</h3></div>', unsafe_allow_html=True)

    # Login form container
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        username = st.text_input("👤 Admin Username", placeholder="Enter admin username")
        password = st.text_input("🔒 Admin Password", type="password", placeholder="Enter admin password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Login as Admin"):
            admin_user = os.getenv('ADMIN_USERNAME', 'admin')
            admin_pass = os.getenv('ADMIN_PASSWORD', 'Parul@1234')
            if username == admin_user and password == admin_pass:
                st.session_state.admin_logged_in = True
                st.success("🎉 Admin logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid admin credentials")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-top: 1rem; color: #ccc;">
            <small>🔐 Secure Admin Access | Parul University</small>
        </div>
        """, unsafe_allow_html=True)

def show_admin_dashboard():
    st.title("Admin Dashboard")

    st.subheader("User Data")
    user_data = get_user_data()
    st.dataframe(user_data)

    if not user_data.empty:
        csv = user_data.to_csv(index=False).encode()
        st.download_button("Download User Data (CSV)", data=csv, file_name="user_data.csv", mime="text/csv")

    st.subheader("Feedback Data")
    feedback_data = get_feedback_data()
    st.dataframe(feedback_data)
    if not feedback_data.empty:
        csv2 = feedback_data.to_csv(index=False).encode()
        st.download_button("Download Feedback Data (CSV)", data=csv2, file_name="feedback_data.csv", mime="text/csv")

    show_analytics()

    if st.button("Logout (Admin)"):
        st.session_state.admin_logged_in = False
        st.success("Admin logged out.")
        st.rerun()

def admin_page():
    if not st.session_state.get('admin_logged_in', False):
        admin_login()
    else:
        admin_dashboard_home()

def show_analytics():
    user_data = get_user_data()
    feedback_data = get_feedback_data()
    st.subheader("Analytics")

    st.write("User Activity Over Time")
    if not user_data.empty and 'timestamp' in user_data.columns:
        try:
            df = user_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            last_30 = datetime.datetime.now() - datetime.timedelta(days=30)
            recent = df[df['timestamp'] >= last_30]
            if not recent.empty:
                counts = recent.groupby(recent['timestamp'].dt.date).size().reset_index(name='counts')
                fig = px.line(counts, x='timestamp', y='counts', title='Users (Last 30 Days)')
                fig.update_layout(xaxis_title='Date', yaxis_title='Number of Users')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent user data.")
        except Exception:
            st.info("No valid timestamp data.")
    else:
        st.info("No user data.")

    st.write("Predicted Fields")
    if not user_data.empty and 'recommended_field' in user_data.columns:
        field_counts = user_data['recommended_field'].value_counts()
        fig_fields = px.pie(values=field_counts.values, names=field_counts.index, title="Predicted Fields")
        st.plotly_chart(fig_fields, use_container_width=True)
    else:
        st.info("No field data.")

    st.write("Experience Levels")
    if not user_data.empty and 'experience_level' in user_data.columns:
        level_counts = user_data['experience_level'].value_counts()
        fig_levels = px.pie(values=level_counts.values, names=level_counts.index, title="Experience Levels")
        st.plotly_chart(fig_levels, use_container_width=True)
    else:
        st.info("No experience level data.")

    st.write("Resume Score Distribution")
    if not user_data.empty and 'resume_score' in user_data.columns:
        fig_scores = px.histogram(user_data, x="resume_score", nbins=20, title="Resume Score Distribution")
        fig_scores.update_layout(xaxis_title="Score", yaxis_title="Count")
        st.plotly_chart(fig_scores, use_container_width=True)
    else:
        st.info("No resume score data.")

    st.write("Feedback Ratings")
    if not feedback_data.empty and 'rating' in feedback_data.columns:
        rating_counts = feedback_data['rating'].value_counts().sort_index()
        fig_ratings = px.bar(x=rating_counts.index, y=rating_counts.values, title="Feedback Ratings")
        fig_ratings.update_layout(xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig_ratings, use_container_width=True)
    else:
        st.info("No feedback ratings data.")

# ==============================
# Student Dashboard Functions
# ==============================
def student_dashboard():
    # Parul theme styling for student dashboard
    st.markdown("""
    <style>
    .student-dashboard-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255,107,53,0.3);
    }
    .welcome-text {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .action-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .action-title {
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Welcome header
    st.markdown(f'''
    <div class="student-dashboard-header">
        <h1>🎓 Student Dashboard</h1>
        <div class="welcome-text">Welcome back, {st.session_state.get('student_name', 'Student')}!</div>
        <div style="margin-top: 0.5rem; font-size: 0.9rem;">🏛️ Parul University</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Metrics section
    st.markdown("### 📊 Your Progress")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Profile Completion", "85%", delta="+5%")
    with col2:
        st.metric("🎆 Events Attended", "3", delta="+1")
    with col3:
        st.metric("🏢 Companies Matched", "12", delta="+3")
    
    # Quick Actions section
    st.markdown('<div class="action-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="action-title">🚀 Quick Actions</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Analyze Resume", use_container_width=True):
            st.info("🔍 Navigate to Resume Analysis to get detailed insights!")
        if st.button("🏢 View Matched Companies", use_container_width=True):
            st.info("🎯 Check companies that match your profile!")
    with col2:
        if st.button("📊 View Placement Stats", use_container_width=True):
            st.info("📈 Explore placement statistics and trends!")
        if st.button("📚 Check Exam Patterns", use_container_width=True):
            st.info("📝 Access exam patterns and preparation materials!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity section
    st.markdown("### 🕒 Recent Activity")
    with st.expander("📊 Recent Resume Analysis"):
        st.write("📅 Last analyzed: 2 days ago")
        st.write("🎯 Score: 78/100")
        st.write("📈 Improvement: +5 points")
    
    with st.expander("🎆 Upcoming Events"):
        st.write("📅 Tech Talk: AI in Industry - Tomorrow 2:00 PM")
        st.write("📅 Placement Drive: TCS - Next Week")
        st.write("📅 Workshop: Resume Building - Friday 10:00 AM")

def update_profile_page():
    st.title("Update Profile")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM students WHERE id = %s", (st.session_state['student_id'],))
            student = cursor.fetchone()
            
        if student:
            st.subheader("Personal Information")
            name = st.text_input("Name", value=student[2])
            email = st.text_input("Email", value=student[3])
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=float(student[5] or 0), step=0.1)
            
            st.subheader("Additional Information")
            phone = st.text_input("Phone Number")
            branch = st.selectbox("Branch", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"])
            year = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            skills = st.text_area("Skills (comma separated)")
            
            if st.button("Update Profile"):
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            UPDATE students SET name=%s, email=%s, cgpa=%s WHERE id=%s
                        """, (name, email, cgpa, st.session_state['student_id']))
                    connection.commit()
                    st.success("Profile updated successfully!")
                    st.session_state['student_name'] = name
                except Exception as e:
                    st.error(f"Error updating profile: {e}")
    finally:
        connection.close()

def view_technical_events():
    st.title("Technical Events")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM technical_events ORDER BY event_date DESC")
            events = cursor.fetchall()
            
        if events:
            for event in events:
                with st.expander(f"📅 {event[1]} - {event[2]}"):
                    st.write(f"**Description:** {event[3]}")
                    st.write(f"**Venue:** {event[4]}")
                    st.write(f"**Organizer:** {event[5]}")
        else:
            st.info("No technical events available at the moment.")
    finally:
        connection.close()

def view_placement_statistics():
    st.title("Placement Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Placements", "450")
    with col2:
        st.metric("Average Package", "₹6.5 LPA")
    with col3:
        st.metric("Highest Package", "₹25 LPA")
    with col4:
        st.metric("Placement Rate", "85%")
    
    st.subheader("Branch-wise Placement Statistics")
    data = {
        'Branch': ['CSE', 'IT', 'ECE', 'ME', 'CE'],
        'Placed': [120, 95, 80, 75, 60],
        'Average Package': [7.2, 6.8, 6.0, 5.5, 5.0]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

def view_exam_patterns():
    st.title("Exam Patterns")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM exam_patterns")
            patterns = cursor.fetchall()
            
        if patterns:
            for pattern in patterns:
                with st.expander(f"📝 {pattern[1]}"):
                    st.write(f"**Pattern Details:** {pattern[2]}")
                    st.write(f"**Syllabus:** {pattern[3]}")
                    st.write(f"**Duration:** {pattern[4]}")
                    st.write(f"**Total Marks:** {pattern[5]}")
        else:
            st.info("No exam patterns available.")
    finally:
        connection.close()

def view_matched_companies():
    st.title("Matched Companies")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT cgpa FROM students WHERE id = %s", (st.session_state['student_id'],))
            student_cgpa = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT * FROM companies WHERE min_cgpa <= %s", (student_cgpa,))
            companies = cursor.fetchall()
            
        if companies:
            st.write(f"Companies matching your CGPA ({student_cgpa}):")
            for company in companies:
                with st.expander(f"🏢 {company[1]}"):
                    st.write(f"**Industry:** {company[2]}")
                    st.write(f"**Required Skills:** {company[3]}")
                    st.write(f"**Minimum CGPA:** {company[4]}")
                    st.write(f"**Package:** {company[5]}")
                    st.write(f"**Location:** {company[6]}")
        else:
            st.info("No companies match your current profile. Consider improving your CGPA or skills.")
    finally:
        connection.close()

# ==============================
# Admin Dashboard Functions
# ==============================
def admin_dashboard_home():
    # Parul theme colors
    st.markdown("""
    <style>
    .admin-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .action-button {
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="admin-header"><h1>🛡️ Admin Dashboard</h1><p>Welcome to Parul University Admin Panel</p></div>', unsafe_allow_html=True)
    

    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", "1,250", delta="+50")
    with col2:
        st.metric("Active Events", "5", delta="+2")
    with col3:
        st.metric("Registered Companies", "45", delta="+8")
    with col4:
        st.metric("Exam Patterns", "12", delta="+1")
    
    st.markdown("### 🚀 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 Add New Event", use_container_width=True):
            st.info("Navigate to 'Manage Technical Events'")
        if st.button("🏢 Add New Company", use_container_width=True):
            st.info("Navigate to 'Manage Company Data'")
    with col2:
        if st.button("📝 Add Exam Pattern", use_container_width=True):
            st.info("Navigate to 'Manage Exam Patterns'")
        if st.button("📊 View Analytics", use_container_width=True):
            st.info("Navigate to 'User Analytics'")

def manage_technical_events():
    st.title("Manage Technical Events")
    
    tab1, tab2 = st.tabs(["Add Event", "View/Edit Events"])
    
    with tab1:
        st.subheader("Add New Technical Event")
        event_name = st.text_input("Event Name")
        event_date = st.date_input("Event Date")
        description = st.text_area("Description")
        venue = st.text_input("Venue")
        organizer = st.text_input("Organizer")
        
        if st.button("Add Event"):
            if event_name and event_date:
                connection = get_database_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO technical_events (event_name, event_date, description, venue, organizer)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (event_name, event_date, description, venue, organizer))
                        connection.commit()
                        st.success("Event added successfully!")
                    except Exception as e:
                        st.error(f"Error adding event: {e}")
                    finally:
                        connection.close()
            else:
                st.warning("Please fill in required fields.")
    
    with tab2:
        st.subheader("Existing Events")
        connection = get_database_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM technical_events ORDER BY event_date DESC")
                    events = cursor.fetchall()
                
                if events:
                    for event in events:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{event[1]}** - {event[2]}")
                            st.write(f"{event[3]}")
                        with col2:
                            if st.button(f"Delete", key=f"del_{event[0]}"):
                                try:
                                    with connection.cursor() as cursor:
                                        cursor.execute("DELETE FROM technical_events WHERE id = %s", (event[0],))
                                    connection.commit()
                                    st.success("Event deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting event: {e}")
                        st.divider()
                else:
                    st.info("No events found.")
            finally:
                connection.close()

def manage_exam_patterns():
    st.title("Manage Exam Patterns")
    
    tab1, tab2 = st.tabs(["Add Pattern", "View/Edit Patterns"])
    
    with tab1:
        st.subheader("Add New Exam Pattern")
        exam_name = st.text_input("Exam Name")
        pattern_details = st.text_area("Pattern Details")
        syllabus = st.text_area("Syllabus")
        duration = st.text_input("Duration")
        total_marks = st.number_input("Total Marks", min_value=0, step=1)
        
        if st.button("Add Pattern"):
            if exam_name and pattern_details:
                connection = get_database_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO exam_patterns (exam_name, pattern_details, syllabus, duration, total_marks)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (exam_name, pattern_details, syllabus, duration, total_marks))
                        connection.commit()
                        st.success("Exam pattern added successfully!")
                    except Exception as e:
                        st.error(f"Error adding pattern: {e}")
                    finally:
                        connection.close()
            else:
                st.warning("Please fill in required fields.")
    
    with tab2:
        st.subheader("Existing Patterns")
        connection = get_database_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM exam_patterns")
                    patterns = cursor.fetchall()
                
                if patterns:
                    for pattern in patterns:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{pattern[1]}**")
                            st.write(f"Duration: {pattern[4]} | Marks: {pattern[5]}")
                        with col2:
                            if st.button(f"Delete", key=f"del_pattern_{pattern[0]}"):
                                try:
                                    with connection.cursor() as cursor:
                                        cursor.execute("DELETE FROM exam_patterns WHERE id = %s", (pattern[0],))
                                    connection.commit()
                                    st.success("Pattern deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting pattern: {e}")
                        st.divider()
                else:
                    st.info("No exam patterns found.")
            finally:
                connection.close()

def manage_company_data():
    st.title("Manage Company Data")
    
    tab1, tab2 = st.tabs(["Add Company", "View/Edit Companies"])
    
    with tab1:
        st.subheader("Add New Company")
        company_name = st.text_input("Company Name")
        industry = st.text_input("Industry")
        required_skills = st.text_area("Required Skills")
        min_cgpa = st.number_input("Minimum CGPA", min_value=0.0, max_value=10.0, step=0.1)
        package_offered = st.text_input("Package Offered")
        job_location = st.text_input("Job Location")
        
        if st.button("Add Company"):
            if company_name and industry:
                connection = get_database_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO companies (company_name, industry, required_skills, min_cgpa, package_offered, job_location)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (company_name, industry, required_skills, min_cgpa, package_offered, job_location))
                        connection.commit()
                        st.success("Company added successfully!")
                    except Exception as e:
                        st.error(f"Error adding company: {e}")
                    finally:
                        connection.close()
            else:
                st.warning("Please fill in required fields.")
    
    with tab2:
        st.subheader("Existing Companies")
        connection = get_database_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM companies")
                    companies = cursor.fetchall()
                
                if companies:
                    for company in companies:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{company[1]}** - {company[2]}")
                            st.write(f"Min CGPA: {company[4]} | Package: {company[5]}")
                        with col2:
                            if st.button(f"Delete", key=f"del_company_{company[0]}"):
                                try:
                                    with connection.cursor() as cursor:
                                        cursor.execute("DELETE FROM companies WHERE id = %s", (company[0],))
                                    connection.commit()
                                    st.success("Company deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting company: {e}")
                        st.divider()
                else:
                    st.info("No companies found.")
            finally:
                connection.close()

def dossier_guide_page():
    st.title("Undergrad Dossier – Guided Flow")
    steps = [
        "Identify the purpose of the dossier (job, further studies, scholarships).",
        "Gather Documents: transcripts, certificates, recommendation letters, proof of activities.",
        "Check Requirements: create a checklist based on institution/employer’s needs.",
        "Write Statement: draft a statement of purpose/personal statement.",
        "Prepare Resume/CV: highlight achievements and skills.",
        "Add Proof: include research, internships, or project evidence.",
        "Request Recommendations: ask professors/employers for letters.",
        "Include Awards: add awards, scholarships, recognitions.",
        "Attach Portfolio: (if applicable) creative/technical work.",
        "Review & Edit: check for errors, consistency, professionalism.",
        "Format & Organize: clean visual structure and consistent style.",
        "Get Feedback: mentors/peers review and suggest improvements.",
        "Finalize: last changes, ensure completeness.",
        "Submit: send digitally or physically as required.",
        "Save Copy: keep a copy; track submission and prepare for next steps."
    ]

    st.markdown("#### Flow")
    for i, s in enumerate(steps, 1):
        st.write(f"{i}. {s}")

    # PDF Download (session-based)
    if st.button("Generate Dossier Guide PDF"):
        pdf = generate_dossier_guide_pdf(steps)
        st.download_button(
            "Download Dossier Guide (PDF)",
            data=pdf,
            file_name="undergrad_dossier_guide.pdf",
            mime="application/pdf"
        )

# ==============================
# Main
# ==============================
def main():
    try:
        init_database()

        if 'session_token' not in st.session_state:
            st.session_state.session_token = generate_session_token()

        latlng, city, state, country = get_geolocation()
        device_info = get_device_info()

        # Parul theme styling for sidebar
        st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #5AB9EA 0%, #F7931E 100%);
        }
        .stSidebar > div:first-child {
            background: linear-gradient(180deg, #5AB9EA 0%, #F7931E 100%);
        }
        .sidebar-title {
            color: white;
            text-align: center;
            padding: 1rem;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .parul-brand {
            text-align: center;
            color: white;
            padding: 0.5rem;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown('<div class="sidebar-title">🏛️ UNDERGRAD DOSSIER</div>', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="parul-brand">Parul University</div>', unsafe_allow_html=True)
        
        image_path = "c:/Users/gopib/Desktop/mahesh/AI_Resume_Analyzer/resum.jpg"
        if os.path.exists(image_path):
            st.sidebar.image(image_path, width=200)
        else:
            st.sidebar.markdown("🎓 Student Portal")

        # Navigation - Admin is always available, others require student login
        if "student_id" not in st.session_state and "admin_logged_in" not in st.session_state:
            # Show limited navigation for non-logged in users
            st.sidebar.markdown("### 🚀 Welcome to Undergrad Dossier")
            st.sidebar.markdown("Please choose your login type:")
            pages = ["🎓 Student Login", "🛡️ Admin Login", "📚 About"]
            page = st.sidebar.radio("Select Portal", pages)
            
            if page == "🎓 Student Login":
                auth_page()
                return
            elif page == "🛡️ Admin Login":
                admin_page()
                return
            elif page == "📚 About":
                about_page()
                return
        
        # Student logged in
        elif "student_id" in st.session_state:
            st.sidebar.success(f"Logged in as: {st.session_state.get('student_name','Student')}")
            st.sidebar.text(f"Session ID: {st.session_state.session_token[:8]}...")
            if city and country:
                st.sidebar.text(f"Location: {city}, {country}")
            
            pages = ["Dashboard", "Update Profile", "Resume Analysis", "Technical Events", "Placement Statistics", "Exam Patterns", "Matched Companies", "Find Jobs", "Feedback", "About"]
            page = st.sidebar.radio("Student Navigation", pages)
        
        # Admin logged in
        elif st.session_state.get("admin_logged_in", False):
            st.sidebar.success("Logged in as: Admin")
            pages = ["Admin Dashboard", "Manage Technical Events", "Manage Exam Patterns", "Manage Company Data", "User Analytics", "About"]
            page = st.sidebar.radio("Admin Navigation", pages)
            
            if page == "Admin Dashboard":
                admin_dashboard_home()
                return
            elif page == "Manage Technical Events":
                manage_technical_events()
                return
            elif page == "Manage Exam Patterns":
                manage_exam_patterns()
                return
            elif page == "Manage Company Data":
                manage_company_data()
                return
            elif page == "User Analytics":
                show_admin_dashboard()
                return

        # Handle navigation for logged-in students
        if "student_id" in st.session_state:
            if page == "Dashboard":
                student_dashboard()
            elif page == "Update Profile":
                update_profile_page()
            elif page == "Resume Analysis":
                user_page()
            elif page == "Technical Events":
                view_technical_events()
            elif page == "Placement Statistics":
                view_placement_statistics()
            elif page == "Exam Patterns":
                view_exam_patterns()
            elif page == "Matched Companies":
                view_matched_companies()
            elif page == "Find Jobs":
                find_jobs_page()
            elif page == "Feedback":
                feedback_page()
            elif page == "About":
                about_page()

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; color: white; padding: 1rem; background: rgba(0,0,0,0.1); border-radius: 10px;">
            <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">🎆 © 2025 UNDERGRADE DOSSIER</div>
            <div style="font-size: 0.7rem; opacity: 0.8;">Designed by Parul University Students</div>
            <div style="font-size: 0.7rem; margin-top: 0.5rem;">🚀 Empowering Future Leaders</div>
        </div>
        """, unsafe_allow_html=True)

        # Show appropriate logout button with styling
        if "student_id" in st.session_state:
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Student Logout", use_container_width=True):
                # Clear only student session, keep admin if logged in
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('student')]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.success("Student logged out successfully! Redirecting to main page...")
                time.sleep(1)
                st.rerun()
        
        if st.session_state.get("admin_logged_in", False):
            st.sidebar.markdown("---")
            if st.sidebar.button("🛡️ Admin Logout", use_container_width=True):
                st.session_state.admin_logged_in = False
                st.success("Admin logged out successfully! Redirecting to main page...")
                time.sleep(1)
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()

# ==============================
# Student Dashboard Functions
# ==============================
def student_dashboard():
    st.title("Student Dashboard")
    st.write(f"Welcome, {st.session_state.get('student_name', 'Student')}!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Profile Completion", "85%")
    with col2:
        st.metric("Events Attended", "3")
    with col3:
        st.metric("Companies Matched", "12")
    
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Analyze Resume"):
            st.info("Navigate to Resume Analysis")
        if st.button("🏢 View Matched Companies"):
            st.info("Navigate to Matched Companies")
    with col2:
        if st.button("📊 View Placement Stats"):
            st.info("Navigate to Placement Statistics")
        if st.button("📚 Check Exam Patterns"):
            st.info("Navigate to Exam Patterns")

def update_profile_page():
    st.title("Update Profile")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM students WHERE id = %s", (st.session_state['student_id'],))
            student = cursor.fetchone()
            
        if student:
            st.subheader("Personal Information")
            name = st.text_input("Name", value=student[2])
            email = st.text_input("Email", value=student[3])
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=float(student[5] or 0), step=0.1)
            
            st.subheader("Additional Information")
            phone = st.text_input("Phone Number")
            branch = st.selectbox("Branch", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"])
            year = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            skills = st.text_area("Skills (comma separated)")
            
            if st.button("Update Profile"):
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            UPDATE students SET name=%s, email=%s, cgpa=%s WHERE id=%s
                        """, (name, email, cgpa, st.session_state['student_id']))
                    connection.commit()
                    st.success("Profile updated successfully!")
                    st.session_state['student_name'] = name
                except Exception as e:
                    st.error(f"Error updating profile: {e}")
    finally:
        connection.close()

def view_technical_events():
    st.title("Technical Events")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM technical_events ORDER BY event_date DESC")
            events = cursor.fetchall()
            
        if events:
            for event in events:
                with st.expander(f"📅 {event[1]} - {event[2]}"):
                    st.write(f"**Description:** {event[3]}")
                    st.write(f"**Venue:** {event[4]}")
                    st.write(f"**Organizer:** {event[5]}")
        else:
            st.info("No technical events available at the moment.")
    finally:
        connection.close()

def view_placement_statistics():
    st.title("Placement Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Placements", "450")
    with col2:
        st.metric("Average Package", "₹6.5 LPA")
    with col3:
        st.metric("Highest Package", "₹25 LPA")
    with col4:
        st.metric("Placement Rate", "85%")
    
    st.subheader("Branch-wise Placement Statistics")
    data = {
        'Branch': ['CSE', 'IT', 'ECE', 'ME', 'CE'],
        'Placed': [120, 95, 80, 75, 60],
        'Average Package': [7.2, 6.8, 6.0, 5.5, 5.0]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

def view_exam_patterns():
    st.title("Exam Patterns")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM exam_patterns")
            patterns = cursor.fetchall()
            
        if patterns:
            for pattern in patterns:
                with st.expander(f"📝 {pattern[1]}"):
                    st.write(f"**Pattern Details:** {pattern[2]}")
                    st.write(f"**Syllabus:** {pattern[3]}")
                    st.write(f"**Duration:** {pattern[4]}")
                    st.write(f"**Total Marks:** {pattern[5]}")
        else:
            st.info("No exam patterns available.")
    finally:
        connection.close()

def view_matched_companies():
    st.title("Matched Companies")
    
    connection = get_database_connection()
    if not connection:
        st.error("Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT cgpa FROM students WHERE id = %s", (st.session_state['student_id'],))
            student_cgpa = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT * FROM companies WHERE min_cgpa <= %s", (student_cgpa,))
            companies = cursor.fetchall()
            
        if companies:
            st.write(f"Companies matching your CGPA ({student_cgpa}):")
            for company in companies:
                with st.expander(f"🏢 {company[1]}"):
                    st.write(f"**Industry:** {company[2]}")
                    st.write(f"**Required Skills:** {company[3]}")
                    st.write(f"**Minimum CGPA:** {company[4]}")
                    st.write(f"**Package:** {company[5]}")
                    st.write(f"**Location:** {company[6]}")
        else:
            st.info("No companies match your current profile. Consider improving your CGPA or skills.")
    finally:
        connection.close()

# ==============================
# Admin Dashboard Functions
# ==============================
def admin_dashboard_home():
    st.title("Admin Dashboard")
    st.write("Welcome to the Admin Panel")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", "1,250")
    with col2:
        st.metric("Active Events", "5")
    with col3:
        st.metric("Registered Companies", "45")
    with col4:
        st.metric("Exam Patterns", "12")
    
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 Add New Event"):
            st.info("Navigate to 'Manage Technical Events'")
        if st.button("🏢 Add New Company"):
            st.info("Navigate to 'Manage Company Data'")
    with col2:
        if st.button("📝 Add Exam Pattern"):
            st.info("Navigate to 'Manage Exam Patterns'")
        if st.button("📊 View Analytics"):
            st.info("Navigate to 'User Analytics'")

def manage_technical_events():
    st.title("Manage Technical Events")
    
    tab1, tab2 = st.tabs(["Add Event", "View/Edit Events"])
    
    with tab1:
        st.subheader("Add New Technical Event")
        event_name = st.text_input("Event Name")
        event_date = st.date_input("Event Date")
        description = st.text_area("Description")
        venue = st.text_input("Venue")
        organizer = st.text_input("Organizer")
        
        if st.button("Add Event"):
            if event_name and event_date:
                connection = get_database_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO technical_events (event_name, event_date, description, venue, organizer)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (event_name, event_date, description, venue, organizer))
                        connection.commit()
                        st.success("Event added successfully!")
                    except Exception as e:
                        st.error(f"Error adding event: {e}")
                    finally:
                        connection.close()
            else:
                st.warning("Please fill in required fields.")
    
    with tab2:
        st.subheader("Existing Events")
        connection = get_database_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM technical_events ORDER BY event_date DESC")
                    events = cursor.fetchall()
                
                if events:
                    for event in events:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{event[1]}** - {event[2]}")
                            st.write(f"{event[3]}")
                        with col2:
                            if st.button(f"Delete", key=f"del_{event[0]}"):
                                try:
                                    with connection.cursor() as cursor:
                                        cursor.execute("DELETE FROM technical_events WHERE id = %s", (event[0],))
                                    connection.commit()
                                    st.success("Event deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting event: {e}")
                        st.divider()
                else:
                    st.info("No events found.")
            finally:
                connection.close()

def manage_exam_patterns():
    st.title("Manage Exam Patterns")
    
    tab1, tab2 = st.tabs(["Add Pattern", "View/Edit Patterns"])
    
    with tab1:
        st.subheader("Add New Exam Pattern")
        exam_name = st.text_input("Exam Name")
        pattern_details = st.text_area("Pattern Details")
        syllabus = st.text_area("Syllabus")
        duration = st.text_input("Duration")
        total_marks = st.number_input("Total Marks", min_value=0, step=1)
        
        if st.button("Add Pattern"):
            if exam_name and pattern_details:
                connection = get_database_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO exam_patterns (exam_name, pattern_details, syllabus, duration, total_marks)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (exam_name, pattern_details, syllabus, duration, total_marks))
                        connection.commit()
                        st.success("Exam pattern added successfully!")
                    except Exception as e:
                        st.error(f"Error adding pattern: {e}")
                    finally:
                        connection.close()
            else:
                st.warning("Please fill in required fields.")
    
    with tab2:
        st.subheader("Existing Patterns")
        connection = get_database_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM exam_patterns")
                    patterns = cursor.fetchall()
                
                if patterns:
                    for pattern in patterns:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{pattern[1]}**")
                            st.write(f"Duration: {pattern[4]} | Marks: {pattern[5]}")
                        with col2:
                            if st.button(f"Delete", key=f"del_pattern_{pattern[0]}"):
                                try:
                                    with connection.cursor() as cursor:
                                        cursor.execute("DELETE FROM exam_patterns WHERE id = %s", (pattern[0],))
                                    connection.commit()
                                    st.success("Pattern deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting pattern: {e}")
                        st.divider()
                else:
                    st.info("No exam patterns found.")
            finally:
                connection.close()

def manage_company_data():
    st.title("Manage Company Data")
    
    tab1, tab2 = st.tabs(["Add Company", "View/Edit Companies"])
    
    with tab1:
        st.subheader("Add New Company")
        company_name = st.text_input("Company Name")
        industry = st.text_input("Industry")
        required_skills = st.text_area("Required Skills")
        min_cgpa = st.number_input("Minimum CGPA", min_value=0.0, max_value=10.0, step=0.1)
        package_offered = st.text_input("Package Offered")
        job_location = st.text_input("Job Location")
        
        if st.button("Add Company"):
            if company_name and industry:
                connection = get_database_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO companies (company_name, industry, required_skills, min_cgpa, package_offered, job_location)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (company_name, industry, required_skills, min_cgpa, package_offered, job_location))
                        connection.commit()
                        st.success("Company added successfully!")
                    except Exception as e:
                        st.error(f"Error adding company: {e}")
                    finally:
                        connection.close()
            else:
                st.warning("Please fill in required fields.")
    
    with tab2:
        st.subheader("Existing Companies")
        connection = get_database_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM companies")
                    companies = cursor.fetchall()
                
                if companies:
                    for company in companies:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{company[1]}** - {company[2]}")
                            st.write(f"Min CGPA: {company[4]} | Package: {company[5]}")
                        with col2:
                            if st.button(f"Delete", key=f"del_company_{company[0]}"):
                                try:
                                    with connection.cursor() as cursor:
                                        cursor.execute("DELETE FROM companies WHERE id = %s", (company[0],))
                                    connection.commit()
                                    st.success("Company deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting company: {e}")
                        st.divider()
                else:
                    st.info("No companies found.")
            finally:
                connection.close()