"""
Converts a structured JD form dict into (jd_text, rubric).
jd_text  → fed into TF-IDF pipeline (same path as a pasted JD)
rubric   → structured criteria for the scoring layer (knockouts, bonus weights)
"""

from __future__ import annotations

# ── Profession skill maps ─────────────────────────────────────────────────────
# Each profession maps to ordered skill categories; each category holds selectable skills.

PROFESSION_SKILLS: dict[str, dict[str, list[str]]] = {
    "Software Engineer": {
        "Programming Languages": [
            "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust",
            "C", "C++", "C#", "Kotlin", "Swift", "Scala", "Ruby", "PHP",
        ],
        "Web Frameworks": [
            "React", "Angular", "Vue", "Next.js", "Node.js", "Express",
            "Django", "Flask", "FastAPI", "Spring Boot", "Rails", "ASP.NET", "Svelte",
        ],
        "Cloud & DevOps": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "Ansible", "Jenkins", "GitHub Actions", "CircleCI", "Helm", "CI/CD",
        ],
        "Agentic / AI Tools": [
            "LLM", "GPT", "Hugging Face", "BERT", "Transformers",
            "TensorFlow", "PyTorch", "Scikit-learn", "MLOps", "LangChain",
            "OpenAI API", "Anthropic API", "RAG",
        ],
        "Databases": [
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
            "SQLite", "DynamoDB", "Elasticsearch", "Cassandra", "Firebase",
        ],
        "Engineering Practices": [
            "Git", "GitHub", "Agile", "Scrum", "REST API", "GraphQL",
            "Microservices", "gRPC", "TDD", "BDD", "Design Patterns",
            "Code Review", "CI/CD", "SOLID",
        ],
    },
    "Data Scientist / ML Engineer": {
        "Programming Languages": ["Python", "R", "MATLAB", "Scala", "Julia", "SQL"],
        "ML Frameworks": [
            "TensorFlow", "PyTorch", "Scikit-learn", "Keras",
            "XGBoost", "LightGBM", "Hugging Face", "BERT", "GPT", "Transformers",
        ],
        "Data & Analytics Tools": [
            "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
            "Jupyter", "Tableau", "Power BI", "Spark", "Kafka", "Airflow",
        ],
        "MLOps & Infrastructure": [
            "MLOps", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
            "Airflow", "Kubeflow", "MLflow", "Git",
        ],
        "Statistics & Methods": [
            "Statistical Analysis", "A/B Testing", "Time Series", "Feature Engineering",
            "NLP", "Computer Vision", "Reinforcement Learning", "Bayesian Statistics",
        ],
    },
    "Marketing & Sales": {
        "Digital Marketing Channels": [
            "SEO", "SEM", "Google Ads", "Facebook Ads", "Instagram", "LinkedIn",
            "Email Marketing", "Content Marketing", "Affiliate Marketing",
            "Influencer Marketing", "YouTube", "WhatsApp Marketing",
        ],
        "Analytics & CRM Tools": [
            "Google Analytics", "HubSpot", "Salesforce", "Zoho CRM",
            "Mailchimp", "Marketo", "Tableau", "Power BI", "A/B Testing",
        ],
        "Skills": [
            "Communication", "Presentation", "Negotiation", "Lead Generation",
            "Brand Management", "Market Research", "Copywriting",
            "Customer Acquisition", "Retention Strategy",
        ],
        "Spoken Languages": [
            "English", "Hindi", "Tamil", "Telugu", "Kannada",
            "Bengali", "Marathi", "Gujarati", "Malayalam",
        ],
    },
    "Human Resources": {
        "HR Competencies": [
            "Recruitment", "Talent Acquisition", "Onboarding", "Performance Management",
            "Employee Relations", "Compensation & Benefits", "Training & Development",
            "Succession Planning", "HR Analytics",
        ],
        "HRIS & Tools": [
            "Workday", "SAP HR", "BambooHR", "Zoho People",
            "Darwinbox", "PeopleSoft", "Lever", "Greenhouse", "Jira",
        ],
        "Compliance & Policies": [
            "Labor Law", "GDPR", "POSH Act", "PF & ESI", "Payroll Processing",
            "HR Policies", "HRBP", "Industrial Relations",
        ],
        "Soft Skills": [
            "Communication", "Conflict Resolution", "Empathy", "Leadership",
            "Negotiation", "Confidentiality", "Change Management",
        ],
    },
    "Finance & Accounting": {
        "Accounting Standards": [
            "GAAP", "IFRS", "GST", "TDS", "Income Tax", "Auditing",
            "Financial Reporting", "ICAI Standards",
        ],
        "Tools & Software": [
            "Excel", "Tally", "SAP FICO", "QuickBooks", "Zoho Books",
            "Bloomberg", "Xero", "Power BI",
        ],
        "Skills": [
            "Financial Analysis", "Risk Management", "Budget Forecasting",
            "Cost Accounting", "Treasury Management", "Valuation",
            "Mergers & Acquisitions", "Investment Analysis",
        ],
    },
    "Operations / Management": {
        "Methodologies": [
            "Agile", "Scrum", "Kanban", "Lean", "Six Sigma",
            "PMP", "Prince2", "PRINCE2",
        ],
        "Tools": [
            "SAP ERP", "Oracle ERP", "Jira", "Asana", "Trello",
            "Monday.com", "MS Project", "Tableau", "Power BI",
        ],
        "Skills": [
            "Project Management", "Supply Chain", "Vendor Management",
            "Process Improvement", "Resource Planning", "Risk Management",
            "Stakeholder Management", "Change Management",
        ],
    },
}

SOFT_SKILLS_COMMON = [
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Critical Thinking", "Time Management", "Adaptability",
    "Collaboration", "Presentation", "Mentoring", "Documentation",
]

SENIORITY_LEVELS = ["Junior", "Mid-level", "Senior", "Lead / Principal", "Manager", "Director"]
WORK_MODES = ["Remote", "Hybrid", "On-site"]
DEGREE_REQUIREMENTS = ["Any", "Diploma", "Bachelor's", "Master's", "PhD"]
GENDER_OPTIONS = ["No preference", "Male", "Female", "Non-binary"]


# ── Core conversion function ──────────────────────────────────────────────────

def build_jd(form: dict) -> tuple[str, dict]:
    """
    Convert a filled JD form dict to (jd_text, rubric).

    Returns
    -------
    jd_text : str
        Richly worded paragraph block suitable for TF-IDF vectorisation.
    rubric : dict
        Structured scoring criteria consumed by rubric_scorer.apply_rubric().
    """
    lines: list[str] = []
    profession = form.get("profession", "")
    seniority = form.get("seniority", "")
    exp_min = form.get("experience_min", 0)
    exp_max = form.get("experience_max", 0)
    work_modes = form.get("work_mode", [])
    degree = form.get("degree", "Any")
    field = form.get("field_of_study", "")
    tenth_min = form.get("tenth_min", 0.0)
    twelfth_min = form.get("twelfth_min", 0.0)
    cgpa_min = form.get("cgpa_min", 0.0)
    gap_max = form.get("gap_max_months", 0)
    relocate = form.get("willing_to_relocate", False)
    soft_skills = form.get("soft_skills", [])
    gender_pref = form.get("gender_preference", "No preference")
    selected_skills: dict[str, list[str]] = form.get("skills", {})
    spoken_langs = form.get("spoken_languages", [])

    # Role heading
    role_desc = f"{seniority} {profession}".strip() if seniority else profession
    lines.append(f"Job Title: {role_desc}")

    # Experience
    if exp_max and exp_max > exp_min:
        lines.append(f"Required Experience: {exp_min}–{exp_max} years of relevant experience.")
    elif exp_min:
        lines.append(f"Required Experience: Minimum {exp_min} years of relevant experience.")

    # Work mode
    if work_modes:
        lines.append(f"Work Mode: {', '.join(work_modes)}.")

    # Skills by category
    all_required: list[str] = []
    skill_weights: dict[str, float] = {}
    for category, skills in selected_skills.items():
        if not skills:
            continue
        lines.append(f"{category}: {', '.join(skills)}.")
        all_required.extend(skills)
        skill_weights[category] = 1.0

    # Soft skills
    if soft_skills:
        lines.append(f"Soft Skills: {', '.join(soft_skills)}.")

    # Spoken languages (for relevant roles)
    if spoken_langs:
        lines.append(f"Spoken Languages: {', '.join(spoken_langs)}.")

    # Education
    edu_parts: list[str] = []
    if degree and degree != "Any":
        edu_parts.append(f"{degree}'s degree required")
        if field:
            edu_parts[-1] += f" in {field}"
    elif field:
        edu_parts.append(f"Relevant degree in {field} preferred")
    if tenth_min:
        edu_parts.append(f"minimum {tenth_min:.0f}% in 10th grade")
    if twelfth_min:
        edu_parts.append(f"minimum {twelfth_min:.0f}% in 12th grade")
    if cgpa_min:
        edu_parts.append(f"minimum {cgpa_min:.1f} CGPA in undergraduate degree")
    if edu_parts:
        lines.append("Education: " + "; ".join(edu_parts) + ".")

    # Career gap
    if gap_max:
        lines.append(
            f"Career Gap Policy: Maximum {gap_max} months of career gap is acceptable."
        )
    else:
        lines.append("Career Gap Policy: Career gaps are acceptable.")

    # Relocation
    if relocate:
        lines.append("Candidates must be willing to relocate.")

    jd_text = "\n".join(lines)

    rubric = {
        "required_skills": all_required,
        "education_floor": {
            "degree": degree,
            "tenth_min": tenth_min,
            "twelfth_min": twelfth_min,
            "cgpa_min": cgpa_min,
            "field_of_study": field,
        },
        "max_gap_months": gap_max,
        "gender_preference": gender_pref,
        "skill_weights": skill_weights,
        "profession": profession,
        "seniority": seniority,
    }

    return jd_text, rubric
