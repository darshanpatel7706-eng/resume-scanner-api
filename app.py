from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import spacy
import pdfplumber
import pandas as pd
import re
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ============ CONFIG ============
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resumescanner.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'resume-scanner-secret-2026'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

nlp = spacy.load('en_core_web_sm')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load datasets
try:
    jobs_df = pd.read_csv(os.path.join(DATA_DIR, 'jobs_data.csv'))
    skills_df = pd.read_csv(os.path.join(DATA_DIR, 'skills_data.csv'))
    salary_df = pd.read_csv(os.path.join(DATA_DIR, 'salary_data.csv'))
    print("✅ Datasets loaded!")
except Exception as e:
    print(f"❌ Dataset error: {e}")
    jobs_df = pd.DataFrame()
    skills_df = pd.DataFrame()
    salary_df = pd.DataFrame()

# ============ DATABASE MODELS ============
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scans = db.relationship('ResumeScan', backref='user', lazy=True)

class ResumeScan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200))
    score = db.Column(db.Integer)
    strength = db.Column(db.String(50))
    skills = db.Column(db.Text)
    experience = db.Column(db.String(100))
    education = db.Column(db.String(100))
    ats_score = db.Column(db.Integer)
    matched_jobs = db.Column(db.Text)
    scanned_at = db.Column(db.DateTime, default=datetime.utcnow)

class JobApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_title = db.Column(db.String(200))
    company = db.Column(db.String(200))
    status = db.Column(db.String(50), default='Applied')
    applied_at = db.Column(db.DateTime, default=datetime.utcnow)

# ============ EDUCATION-BASED JOB CATEGORIES ============
EDUCATION_JOB_MAP = {
    'science': ['Python Developer', 'ML Engineer', 'Data Scientist', 'Backend Developer',
                'AI Engineer', 'DevOps Engineer', 'Full Stack Developer', 'NLP Engineer'],
    'commerce': ['Data Analyst', 'Business Analyst', 'Product Manager', 'Financial Analyst',
                 'ERP Consultant', 'Digital Marketing', 'Operations Manager'],
    'arts': ['Content Writer', 'UI/UX Designer', 'Digital Marketing', 'HR Manager',
             'Social Media Manager', 'Graphic Designer', 'SEO Specialist'],
    'engineering': ['Python Developer', 'ML Engineer', 'Backend Developer', 'DevOps Engineer',
                    'Cloud Architect', 'Flutter Developer', 'Data Engineer'],
    'medical': ['Healthcare Data Analyst', 'Medical Content Writer', 'Health Informatics',
                'Clinical Data Manager'],
    'general': ['Data Analyst', 'Business Analyst', 'HR Manager', 'Operations', 'Sales']
}

def detect_education_stream(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ['b.tech', 'btech', 'm.tech', 'mtech', 'b.e', 'be ',
                                       'engineering', 'computer science', 'information technology',
                                       'electronics', 'mechanical', 'electrical']):
        return 'engineering'
    elif any(w in text_lower for w in ['bca', 'mca', 'b.sc computer', 'b.sc it',
                                        'physics', 'chemistry', 'mathematics', 'statistics',
                                        'b.sc', 'bsc', 'm.sc', 'msc']):
        return 'science'
    elif any(w in text_lower for w in ['b.com', 'bcom', 'm.com', 'mcom', 'mba',
                                        'commerce', 'accounting', 'finance', 'economics',
                                        'business administration', 'chartered accountant', 'ca ']):
        return 'commerce'
    elif any(w in text_lower for w in ['b.a', 'ba ', 'm.a', 'ma ', 'arts', 'humanities',
                                        'english literature', 'history', 'political science',
                                        'sociology', 'psychology', 'journalism']):
        return 'arts'
    elif any(w in text_lower for w in ['mbbs', 'bds', 'nursing', 'pharmacy', 'medical',
                                        'doctor', 'physician']):
        return 'medical'
    return 'general'

def get_skills_for_stream(stream):
    stream_skills = {
        'engineering': ['python', 'java', 'sql', 'git', 'docker', 'aws', 'javascript',
                        'react', 'flask', 'django', 'machine learning', 'data structures'],
        'science': ['python', 'r', 'statistics', 'sql', 'machine learning', 'data analysis',
                    'excel', 'tableau', 'power bi', 'numpy', 'pandas'],
        'commerce': ['excel', 'tally', 'sap', 'sql', 'power bi', 'tableau', 'accounting',
                     'finance', 'gst', 'taxation', 'erp'],
        'arts': ['content writing', 'seo', 'social media', 'adobe photoshop', 'canva',
                 'figma', 'communication', 'ms office', 'google analytics'],
        'medical': ['medical coding', 'ehr', 'clinical data', 'healthcare analytics',
                    'python', 'sql', 'excel'],
        'general': ['excel', 'ms office', 'communication', 'sql', 'data entry', 'crm']
    }
    return stream_skills.get(stream, stream_skills['general'])

def get_skills_list():
    if not skills_df.empty:
        return skills_df['skill'].str.lower().tolist()
    return ['python', 'sql', 'java', 'react', 'machine learning', 'docker', 'aws', 'git']

def extract_experience(text):
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*experience',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return f"{match.group(1)} Years"
    return "Fresher"

def extract_education(text):
    text_lower = text.lower()
    if 'ph.d' in text_lower or 'phd' in text_lower: return "Ph.D"
    elif 'm.tech' in text_lower or 'mtech' in text_lower: return "M.Tech"
    elif 'mba' in text_lower: return "MBA"
    elif 'b.tech' in text_lower or 'btech' in text_lower: return "B.Tech"
    elif 'b.com' in text_lower or 'bcom' in text_lower: return "B.Com"
    elif 'b.a' in text_lower: return "B.A"
    elif 'b.sc' in text_lower or 'bsc' in text_lower: return "B.Sc"
    elif 'bachelor' in text_lower: return "Bachelor's Degree"
    elif 'diploma' in text_lower: return "Diploma"
    return "Graduate"

def extract_skills_smart(text, stream):
    general_skills = get_skills_list()
    stream_specific = get_skills_for_stream(stream)
    all_skills = list(set(general_skills + stream_specific))
    text_lower = text.lower()
    found = []
    for skill in all_skills:
        if skill in text_lower:
            found.append(skill.title())
    return found

def match_jobs_smart(found_skills, stream):
    if jobs_df.empty:
        return []
    allowed_roles = EDUCATION_JOB_MAP.get(stream, EDUCATION_JOB_MAP['general'])
    skills_lower = [s.lower() for s in found_skills]
    matched = []
    for _, job in jobs_df.iterrows():
        role_allowed = any(role.lower() in job['title'].lower() for role in allowed_roles)
        if not role_allowed:
            continue
        job_skills = str(job['skills']).lower().split(',')
        job_skills = [s.strip() for s in job_skills]
        common = [s for s in job_skills if s in skills_lower]
        if len(common) > 0 or len(skills_lower) == 0:
            score = int((len(common) / max(len(job_skills), 1)) * 100) if common else 25
            matched.append({
                "id": int(job['id']),
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "type": job['type'],
                "salary": f"₹{int(job['salary_min'])//100000}-{int(job['salary_max'])//100000} LPA",
                "match": f"{score}%",
                "hot": bool(job['hot']),
            })
    matched.sort(key=lambda x: int(x['match'].replace('%', '')), reverse=True)
    return matched[:8]

def analyze_strength(text, found_skills):
    text_lower = text.lower()
    strong, weak, improve = [], [], []

    if len(found_skills) >= 8:
        strong.append(f"Strong skill set detected ({len(found_skills)} skills)")
    elif len(found_skills) >= 4:
        strong.append(f"Good skills found ({len(found_skills)})")
        improve.append("Add more relevant technical skills")
    else:
        weak.append("Very few skills found")
        improve.append("Add key skills relevant to your field")

    if re.search(r'(\d+)\+?\s*years?', text_lower):
        strong.append("Experience clearly mentioned")
    else:
        weak.append("Experience not mentioned")
        improve.append("Clearly state your years of experience")

    if any(w in text_lower for w in ['b.tech', 'btech', 'bachelor', 'mba', 'm.tech', 'b.com', 'b.sc']):
        strong.append("Education details found")
    else:
        weak.append("Education details missing")
        improve.append("Add your degree and college name clearly")

    if 'project' in text_lower:
        strong.append("Projects mentioned")
    else:
        weak.append("No projects mentioned")
        improve.append("Add 2-3 projects with description")

    if 'github' in text_lower:
        strong.append("GitHub profile found")
    else:
        weak.append("GitHub profile missing")
        improve.append("Add GitHub profile link")

    if 'linkedin' in text_lower:
        strong.append("LinkedIn profile found")
    else:
        weak.append("LinkedIn profile missing")
        improve.append("Add LinkedIn profile link")

    score = min(len(strong) * 15, 100)
    score = max(score, 20)

    if score >= 75: strength, color = "Strong 💪", "green"
    elif score >= 50: strength, color = "Average 👍", "orange"
    else: strength, color = "Weak ⚠️", "red"

    return {"score": score, "strength": strength, "strength_color": color,
            "strong_points": strong, "weak_points": weak, "improvements": improve}

def calculate_ats(text, found_skills):
    score = 0
    text_lower = text.lower()
    reasons = []
    skill_score = min(len(found_skills) * 5, 40)
    score += skill_score
    reasons.append("✅ Good skill set" if len(found_skills) >= 6 else "⚠️ Add more relevant skills")
    keywords = ['experience', 'project', 'developed', 'implemented', 'managed', 'built', 'led']
    found_kw = [k for k in keywords if k in text_lower]
    score += min(len(found_kw) * 3, 20)
    reasons.append("✅ Good action keywords" if len(found_kw) >= 4 else "⚠️ Use action words: built, developed, led")
    if re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text):
        score += 5
        reasons.append("✅ Email found")
    if 'github' in text_lower:
        score += 8
        reasons.append("✅ GitHub profile found")
    else:
        reasons.append("❌ Add GitHub profile link")
    if 'linkedin' in text_lower:
        score += 7
        reasons.append("✅ LinkedIn found")
    else:
        reasons.append("❌ Add LinkedIn profile link")
    if any(w in text_lower for w in ['b.tech', 'bachelor', 'mba', 'b.com', 'b.sc']):
        score += 10
        reasons.append("✅ Education found")
    else:
        reasons.append("❌ Add education details")
    if 'project' in text_lower:
        score += 5
        reasons.append("✅ Projects mentioned")
    else:
        reasons.append("❌ Add projects with description")
    return min(score, 100), reasons

def skill_gap(found_skills, stream):
    stream_required = get_skills_for_stream(stream)
    skills_lower = [s.lower() for s in found_skills]
    have = [s for s in stream_required if s in skills_lower]
    missing = [s for s in stream_required if s not in skills_lower]
    readiness = int((len(have) / max(len(stream_required), 1)) * 100)
    return {
        "role": stream.title() + " Field",
        "readiness_score": readiness,
        "verdict": "Ready! 🎉" if readiness >= 70 else "Almost Ready 💪" if readiness >= 50 else "Need More Skills 📚",
        "have_required": have,
        "missing_required": missing[:5],
        "have_advanced": [],
        "missing_advanced": missing[5:10]
    }

def validate_experience(text):
    text_lower = text.lower()
    issues, valid = [], []
    trust = 100
    if re.search(r'(\d+)\+?\s*years?', text_lower):
        valid.append("✅ Experience duration mentioned")
    else:
        issues.append("⚠️ Experience duration not clear")
        trust -= 15
    if len(re.findall(r'20\d{2}', text)) >= 2:
        valid.append("✅ Work dates mentioned")
    else:
        issues.append("⚠️ Work dates not clearly mentioned")
        trust -= 10
    if re.findall(r'\d+%|\d+ users|\d+ clients|\d+x', text_lower):
        valid.append("✅ Achievements quantified")
    else:
        issues.append("⚠️ Add quantified achievements like 50% improvement")
        trust -= 10
    trust = max(trust, 0)
    return {
        "trust_score": trust,
        "verdict": "Highly Credible ✅" if trust >= 80 else "Mostly Credible 👍" if trust >= 60 else "Needs Verification ⚠️",
        "valid_points": valid,
        "issues": issues
    }

def rewriter_tips(found_skills, experience, education):
    top_skills = found_skills[:5] if found_skills else ['Communication', 'MS Office', 'Problem Solving']
    trending = []
    if not skills_df.empty:
        trending = skills_df[skills_df['trending'] == True]['skill'].tolist()[:3]
    return {
        "summary": f"Results-driven professional with expertise in {', '.join(top_skills)} and {experience} of hands-on experience. Strong educational background in {education} with proven track record of delivering results.",
        "skills_section": f"Technical Skills: {' | '.join(found_skills) if found_skills else 'Add your skills here'}",
        "project_template": "• Developed [Project Name] using [Technologies], resulting in [Quantified Impact]",
        "experience_template": "• [Action Verb] [What you did] using [Tools/Tech] which resulted in [Measurable Impact]",
        "tips": [
            "Start each bullet with action verb: Built, Developed, Led, Managed",
            "Quantify everything: 50% improvement, 10K users, 3x faster",
            "Add GitHub link with active repositories",
            "Tailor resume keywords to match job description",
            "Keep resume to 1 page if under 5 years experience",
        ] + [f"🔥 {s} is trending — consider adding it!" for s in trending]
    }

# ============ APIS ============

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({"success": False, "message": "All fields are required"}), 400

    if '@' not in email or '.' not in email:
        return jsonify({"success": False, "message": "Please enter a valid email address"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email already registered. Please login."}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(name=name, email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "success": True,
        "message": "Registration successful!",
        "token": token,
        "user": {"id": user.id, "name": user.name, "email": user.email}
    })

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()

    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"success": False, "message": "Invalid email or password"}), 401

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "success": True,
        "message": "Login successful",
        "token": token,
        "user": {"id": user.id, "name": user.name, "email": user.email}
    })

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files['resume']
    user_id = request.form.get('user_id')

    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        return jsonify({"success": False, "message": "Could not read PDF file"}), 400

    if len(text.strip()) < 50:
        return jsonify({"success": False, "message": "PDF appears to be empty or unreadable"}), 400

    stream = detect_education_stream(text)
    found_skills = extract_skills_smart(text, stream)
    experience = extract_experience(text)
    education = extract_education(text)
    strength_data = analyze_strength(text, found_skills)
    ats_score, ats_reasons = calculate_ats(text, found_skills)
    skill_gap_data = skill_gap(found_skills, stream)
    exp_validation = validate_experience(text)
    rewriter = rewriter_tips(found_skills, experience, education)
    matched_jobs = match_jobs_smart(found_skills, stream)

    # Save to database if user logged in
    if user_id:
        try:
            scan = ResumeScan(
                user_id=int(user_id),
                filename=file.filename,
                score=strength_data['score'],
                strength=strength_data['strength'],
                skills=','.join(found_skills),
                experience=experience,
                education=education,
                ats_score=ats_score,
                matched_jobs=str(len(matched_jobs))
            )
            db.session.add(scan)
            db.session.commit()
        except:
            pass

    return jsonify({
        "success": True,
        "data": {
            "score": strength_data['score'],
            "strength": strength_data['strength'],
            "strength_color": strength_data['strength_color'],
            "strong_points": strength_data['strong_points'],
            "weak_points": strength_data['weak_points'],
            "improvements": strength_data['improvements'],
            "skills": found_skills if found_skills else ["Add relevant skills"],
            "experience": experience,
            "education": education,
            "education_stream": stream,
            "matched_jobs": matched_jobs,
            "ats_score": ats_score,
            "ats_reasons": ats_reasons,
            "skill_gap": skill_gap_data,
            "experience_validation": exp_validation,
            "rewriter": rewriter,
        }
    })

@app.route('/api/scan-history/<int:user_id>', methods=['GET'])
def scan_history(user_id):
    scans = ResumeScan.query.filter_by(user_id=user_id).order_by(ResumeScan.scanned_at.desc()).all()
    history = []
    for scan in scans:
        history.append({
            "id": scan.id,
            "filename": scan.filename,
            "score": scan.score,
            "strength": scan.strength,
            "skills": scan.skills.split(',') if scan.skills else [],
            "experience": scan.experience,
            "education": scan.education,
            "ats_score": scan.ats_score,
            "scanned_at": scan.scanned_at.strftime("%b %d, %Y")
        })
    return jsonify({"success": True, "history": history})

@app.route('/api/apply-job', methods=['POST'])
def apply_job():
    data = request.json
    application = JobApplication(
        user_id=data.get('user_id', 1),
        job_title=data.get('job_title'),
        company=data.get('company'),
        status='Applied'
    )
    db.session.add(application)
    db.session.commit()
    return jsonify({"success": True, "message": "Application submitted successfully!"})

@app.route('/api/user-stats/<int:user_id>', methods=['GET'])
def user_stats(user_id):
    scans = ResumeScan.query.filter_by(user_id=user_id).all()
    applications = JobApplication.query.filter_by(user_id=user_id).all()
    best_score = max([s.score for s in scans], default=0)
    return jsonify({
        "success": True,
        "stats": {
            "total_scans": len(scans),
            "total_applications": len(applications),
            "best_score": best_score,
            "member_since": User.query.get(user_id).created_at.strftime("%b %Y") if User.query.get(user_id) else "2026"
        }
    })

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    if jobs_df.empty:
        return jsonify({"success": False, "message": "No jobs data"}), 500
    job_type = request.args.get('type', 'all')
    filtered = jobs_df if job_type == 'all' else jobs_df[jobs_df['type'].str.lower() == job_type.lower()]
    jobs_list = []
    for _, job in filtered.head(20).iterrows():
        jobs_list.append({
            "id": int(job['id']),
            "title": job['title'],
            "company": job['company'],
            "location": job['location'],
            "type": job['type'],
            "salary": f"₹{int(job['salary_min'])//100000}-{int(job['salary_max'])//100000} LPA",
            "skills": str(job['skills']).split(','),
            "posted": f"{int(job['posted_days_ago'])} days ago",
            "hot": bool(job['hot']),
        })
    return jsonify({"success": True, "jobs": jobs_list})

@app.route('/api/skills', methods=['GET'])
def get_skills():
    if skills_df.empty:
        return jsonify({"success": False}), 500
    trending_only = request.args.get('trending', 'false') == 'true'
    filtered = skills_df[skills_df['trending'] == True] if trending_only else skills_df
    return jsonify({"success": True, "skills": filtered.to_dict('records')})

@app.route('/api/salary', methods=['GET'])
def get_salary():
    role = request.args.get('role', 'Python Developer')
    experience = request.args.get('experience', 'Junior')
    city = request.args.get('city', 'Bangalore')
    if salary_df.empty:
        return jsonify({"success": False}), 500
    filtered = salary_df[
        (salary_df['role'].str.lower() == role.lower()) &
        (salary_df['experience_level'].str.lower() == experience.lower())
    ]
    if not filtered.empty:
        row = filtered.iloc[0]
        return jsonify({"success": True, "salary": {
            "min": int(row['min_salary']) // 100000,
            "max": int(row['max_salary']) // 100000,
            "avg": int(row['avg_salary']) // 100000,
            "currency": row['currency'],
            "country": row['country']
        }})
    return jsonify({"success": False, "message": "Salary data not found"}), 404

@app.route('/api/market-stats', methods=['GET'])
def market_stats():
    if jobs_df.empty:
        return jsonify({"success": False}), 500
    return jsonify({
        "success": True,
        "stats": {
            "total_jobs": len(jobs_df),
            "total_companies": jobs_df['company'].nunique(),
            "avg_salary_lpa": round(jobs_df['salary_max'].mean() / 100000, 1),
            "hot_jobs": len(jobs_df[jobs_df['hot'] == True]),
            "remote_jobs": len(jobs_df[jobs_df['type'] == 'Remote']),
            "top_skills": skills_df[skills_df['demand_level'] == 'Very High']['skill'].tolist()[:5] if not skills_df.empty else [],
            "trending_skills": skills_df[skills_df['trending'] == True]['skill'].tolist()[:5] if not skills_df.empty else [],
        }
    })

@app.route('/api/trending-skills', methods=['GET'])
def trending_skills():
    if skills_df.empty:
        return jsonify({"success": False}), 500
    trending = skills_df[skills_df['trending'] == True].sort_values('avg_salary_boost', ascending=False)
    return jsonify({"success": True, "trending_skills": trending.to_dict('records')})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ Database created!")
    app.run(debug=True, host='0.0.0.0', port=5000)