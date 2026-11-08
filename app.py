
import streamlit as st
import PyPDF2
from docx import Document
import pandas as pd
import re
import os
import json
import hashlib
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HireSense",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────
@st.cache_resource
def load_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not found in .env — AI features disabled.")
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

client = load_client()

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
for key, default in {
    "results_df": None,
    "raw_results": [],
    "ideal_profile": {},
    "processing_complete": False,
    "gpt_model": "openai/gpt-4o",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header{font-size:2.5rem;font-weight:bold;
        background:linear-gradient(90deg,#1E88E5,#7E57C2);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        text-align:center;margin-bottom:.5rem}
    .sub-header{font-size:1.1rem;color:#666;text-align:center;margin-bottom:2rem}
    .section-header{font-size:1.4rem;font-weight:bold;color:#1E88E5;
        margin-top:1.5rem;margin-bottom:1rem;padding-bottom:.5rem;
        border-bottom:2px solid #E3F2FD}
    .info-box{background:linear-gradient(135deg,#E3F2FD 0%,#BBDEFB 100%);
        padding:1.2rem;border-radius:12px;border-left:4px solid #1E88E5;margin-bottom:1rem}
    .score-high{color:#4CAF50;font-weight:bold;font-size:1.3rem}
    .score-medium{color:#FF9800;font-weight:bold;font-size:1.3rem}
    .score-low{color:#F44336;font-weight:bold;font-size:1.3rem}
    .metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        padding:1.5rem;border-radius:15px;color:white;text-align:center;
        box-shadow:0 4px 15px rgba(0,0,0,.1)}
    .metric-value{font-size:2rem;font-weight:bold}
    .metric-label{font-size:.9rem;opacity:.9}
    .reasoning-box{background:#f8f9fa;border-left:3px solid #7E57C2;
        padding:.8rem 1rem;border-radius:0 8px 8px 0;font-size:.9rem;color:#444;margin-top:.5rem}
    .footer{text-align:center;padding:2rem;color:#666;
        border-top:1px solid #E0E0E0;margin-top:2rem}
    .skill-chip{display:inline-block;background:#E3F2FD;color:#1565C0;
        border-radius:20px;padding:2px 10px;font-size:.8rem;margin:2px}
    .missing-chip{display:inline-block;background:#FFEBEE;color:#C62828;
        border-radius:20px;padding:2px 10px;font-size:.8rem;margin:2px}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text(file) -> str:
    """
    Unified text extractor for PDF, DOCX, and plain-text uploads.
    Returns empty string on failure (never raises).
    """
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs).strip()

        elif file.type == "text/plain":
            return file.read().decode("utf-8").strip()

    except Exception as e:
        st.warning(f"⚠️ Could not read {file.name}: {e}")
    return ""


def truncate_text(text: str, max_words: int = 2500) -> str:
    """Keeps token usage predictable by capping resume length."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "\n[... resume truncated for token limit ...]"
    return text


def file_hash(file) -> str:
    """SHA-256 of file bytes — used for result caching."""
    file.seek(0)
    h = hashlib.sha256(file.read()).hexdigest()
    file.seek(0)
    return h


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — LLM CALL WRAPPER
# ═══════════════════════════════════════════════════════════════
def _call_llm_once(system_prompt: str, user_prompt: str, model: str,
             max_tokens: int = 700, temperature: float = 0.1) -> dict | None:
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max(max_tokens, 2000),
        )

        # ── Guard: check response structure before accessing content ──
        if not resp.choices:
            st.warning("⚠️ LLM returned an empty choices list.")
            return None

        message = resp.choices[0].message
        raw = getattr(message, "content", None)

        if raw is None:
            # Model may have refused or returned a tool-use response
            refusal = getattr(message, "refusal", None)
            if refusal:
                st.warning(f"⚠️ Model refused the request: {refusal}")
            else:
                st.warning("⚠️ LLM returned no content. The model may have filtered this request.")
            return None

        raw = raw.strip()
        if not raw:
            st.warning("⚠️ LLM returned an empty response.")
            return None

        # Strip markdown fences if model added them
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

        # ── Attempt 1: parse as-is ────────────────────────────────────
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # ── Attempt 2: truncation repair ──────────────────────────────
        last_comma = raw.rfind(",")
        last_brace = raw.rfind("}")
        if last_comma > last_brace:
            repaired = raw[:last_comma] + "\n}"
        else:
            repaired = raw + "\n}"

        try:
            result = json.loads(repaired)
            st.warning("⚠️ LLM response was truncated — some fields may be incomplete.")
            return result
        except json.JSONDecodeError as e:
            st.warning(f"⚠️ LLM returned non-JSON output: {e}")
            return None

    except Exception as e:
        st.warning(f"⚠️ LLM API error: {e}")
        return None

def call_llm(system_prompt, user_prompt, model, max_tokens=700, temperature=0.1):
    for attempt in range(2):
        result = _call_llm_once(system_prompt, user_prompt, model, max_tokens, temperature)
        if result is not None:
            return result
        time.sleep(5)
    return None

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PROMPTS  (all prompts live here, easy to tune)
# ═══════════════════════════════════════════════════════════════

# ---------- Prompt 1: Ideal Profile Extraction ----------
IDEAL_PROFILE_SYSTEM = (
    "You are an expert technical recruiter and structured-data extractor. "
    "Your only job is to return valid, minified JSON — no markdown, no prose, no trailing commas."
)

IDEAL_PROFILE_USER = """Analyze the job description below and extract a precise, structured ideal candidate profile.

JOB DESCRIPTION:
{job_description}

Return ONLY this JSON object — fill every field, use empty list [] if nothing found:
{{
  "job_title": "<string>",
  "seniority_level": "<Junior|Mid|Senior|Lead|Principal>",
  "required_skills": ["<must-have skill/tool>"],
  "preferred_skills": ["<nice-to-have skill/tool>"],
  "min_experience_years": <integer>,
  "education_requirement": "<degree + field, e.g. Bachelor in Computer Science>",
  "key_responsibilities": ["<responsibility>"],
  "certifications": ["<certification name or empty list>"],
  "domain_keywords": ["<domain-specific terms, acronyms, tech stack>"],
  "red_flags": ["<deal-breaker gaps to penalise strongly>"]
}}

Rules:
- Extract ONLY what is explicitly stated or directly implied
- Do NOT invent skills or requirements
- Normalise skill names (e.g. \"JS\" → \"JavaScript\")
- Split compound requirements into separate list items"""


# ---------- Prompt 2: Resume Scoring ----------
SCORING_SYSTEM = (
    "You are a senior technical recruiter performing a strict, evidence-based resume evaluation. "
    "Return ONLY valid JSON — no markdown, no prose, no trailing commas."
)

SCORING_USER = """Score the candidate resume against the ideal profile below.

=== IDEAL CANDIDATE PROFILE ===
{ideal_profile}

=== CANDIDATE RESUME ===
{resume_text}

=== SCORING INSTRUCTIONS ===

Evaluate on FIVE dimensions. Be STRICT — do not assume or infer missing skills.

1. skills_match (weight 40%)
   - Check each required_skill against resume explicitly
   - Partial credit for synonyms/equivalent tech only when clearly equivalent
   - Heavy penalty for missing required_skills; mild penalty for missing preferred_skills
   - Formula hint: (matched_required / total_required) * 100, then adjust for preferred

2. experience_relevance (weight 30%)
   - Does the candidate's past roles match the responsibilities?
   - Penalise if total relevant years < min_experience_years
   - Count ONLY relevant experience, not total career length

3. project_relevance (weight 15%)
   - Do listed projects/work samples align with key_responsibilities?
   - Reward quantifiable impact (metrics, scale, business outcome)
   - Penalise generic/vague project descriptions

4. education_fit (weight 10%)
   - 100: exact degree + field match
   - 75: related field
   - 50: different field but compensated by strong experience
   - 25: no degree but relevant certifications
   - 0: no education info found

5. keyword_coverage (weight 5%)
   - What % of domain_keywords appear in resume?

=== PENALISATION RULES ===
- Each MISSING required_skill: -5 points from skills_match (floor 0)
- Each detected red_flag: -8 points from overall_score (floor 0)
- Seniority mismatch (e.g. Junior applying for Senior): -10 from overall_score

=== OUTPUT FORMAT (strict JSON) ===
{{
  "overall_score": <weighted integer 0-100>,
  "dimension_scores": {{
    "skills_match": <0-100>,
    "experience_relevance": <0-100>,
    "project_relevance": <0-100>,
    "education_fit": <0-100>,
    "keyword_coverage": <0-100>
  }},
  "experience_years_detected": <integer or null>,
  "matched_required_skills": ["<skill>"],
  "missing_required_skills": ["<skill>"],
  "matched_preferred_skills": ["<skill>"],
  "red_flags_triggered": ["<flag or empty list>"],
  "strengths": ["<1-sentence strength>"],
  "concerns": ["<1-sentence concern>"],
  "recruiter_summary": "<2-3 sentence objective evaluation focusing on fit>"
}}

Compute overall_score as:
  round(skills_match*0.40 + experience_relevance*0.30 + project_relevance*0.15 + education_fit*0.10 + keyword_coverage*0.05)
Then apply penalties from PENALISATION RULES."""


# ---------- Prompt 3: Ranking Refinement ----------
REFINEMENT_SYSTEM = (
    "You are a hiring manager performing a cross-candidate consistency review. "
    "Return ONLY valid JSON — no markdown, no prose."
)

REFINEMENT_USER = """Review the scores below for {count} candidates applying for: {job_title}

=== IDEAL PROFILE SUMMARY ===
{ideal_profile_summary}

=== CANDIDATE SCORES ===
{candidates_summary}

=== YOUR TASK ===
Check for scoring inconsistencies across candidates. Apply SMALL adjustments (±8 points MAX) where:
- A well-rounded candidate appears underscored vs a one-dimensional candidate
- Two near-identical profiles have surprisingly different scores
- A candidate is clearly stronger/weaker than their score suggests

Return ONLY this JSON:
{{
  "refined_rankings": [
    {{
      "filename": "<exact filename>",
      "refined_score": <integer 0-100>,
      "delta": <signed integer, 0 if unchanged>,
      "adjustment_reason": "<one sentence or 'No change needed'>"
    }}
  ],
  "panel_notes": "<optional 1-2 sentence overall observation for HR>"
}}

Rules:
- Include ALL {count} candidates (even unchanged ones, delta=0)
- Never exceed ±8 point adjustment
- Prefer NO change unless clearly warranted"""


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def extract_ideal_profile(job_description: str, model: str) -> dict:
    """
    Step 1: Convert raw JD text into a structured scoring rubric.
    Falls back to safe empty defaults so downstream code never crashes.
    """

    jd_words = job_description.split()
    if len(jd_words) > 1500:
        job_description = " ".join(jd_words[:1500]) + "\n[truncated]"


    result = call_llm(
        system_prompt=IDEAL_PROFILE_SYSTEM,
        user_prompt=IDEAL_PROFILE_USER.format(job_description=job_description),
        model=model,
        max_tokens=700,
        temperature=0.05,   # near-deterministic extraction
    )

    # Safe defaults — every key the rest of the code depends on
    defaults = {
        "job_title": "the role",
        "seniority_level": "Mid",
        "required_skills": [],
        "preferred_skills": [],
        "min_experience_years": 0,
        "education_requirement": "Not specified",
        "key_responsibilities": [],
        "certifications": [],
        "domain_keywords": [],
        "red_flags": [],
    }

    if not result or not isinstance(result, dict):
        st.warning("⚠️ Could not extract ideal profile — using empty defaults.")
        return defaults

    # Merge: keep extracted values, fill missing with defaults
    for k, v in defaults.items():
        result.setdefault(k, v)

    return result


def score_resume_against_profile(
    resume_text: str,
    ideal_profile: dict,
    filename: str,
    model: str,
) -> dict:
    """
    Step 2: Score a single resume against the ideal profile.
    Returns a fully-populated result dict (never None — falls back to zeros).
    """
    ideal_str = json.dumps(ideal_profile, indent=2)
    clean_resume = truncate_text(resume_text, max_words=1500)

    result = call_llm(
        system_prompt=SCORING_SYSTEM,
        user_prompt=SCORING_USER.format(
            ideal_profile=ideal_str,
            resume_text=clean_resume,
        ),
        model=model,
        max_tokens=700,
        temperature=0.15,   # low but not zero — allows semantic reasoning
    )

    # Guaranteed fallback so the UI never breaks
    fallback = {
        "filename": filename,
        "overall_score": 0,
        "dimension_scores": {
            "skills_match": 0,
            "experience_relevance": 0,
            "project_relevance": 0,
            "education_fit": 0,
            "keyword_coverage": 0,
        },
        "experience_years_detected": None,
        "matched_required_skills": [],
        "missing_required_skills": ideal_profile.get("required_skills", []),
        "matched_preferred_skills": [],
        "red_flags_triggered": [],
        "strengths": [],
        "concerns": ["Resume could not be evaluated — check file content"],
        "recruiter_summary": "Evaluation failed due to a parsing or API error.",
    }

    if not result or not isinstance(result, dict):
        return fallback

    result["filename"] = filename

    # Ensure every key exists so UI can safely .get() without checking
    for k, v in fallback.items():
        result.setdefault(k, v)

    # Clamp overall_score to valid range
    result["overall_score"] = max(0, min(100, int(result.get("overall_score", 0))))

    return result


def refine_rankings(
    scored_results: list[dict],
    ideal_profile: dict,
    model: str,
) -> list[dict]:
    """
    Step 3 (optional): Cross-candidate consistency pass.
    Returns the same list with 'refined_score' and 'adjustment_reason' merged in.
    Silently returns original list if refinement fails.
    """
    if not client or len(scored_results) < 2:
        return scored_results

    job_title = ideal_profile.get("job_title", "the role")

    # Compact ideal profile summary to save tokens
    ideal_summary = (
        f"Required skills: {', '.join(ideal_profile.get('required_skills', [])[:10])}\n"
        f"Min experience: {ideal_profile.get('min_experience_years', 0)} years\n"
        f"Seniority: {ideal_profile.get('seniority_level', 'Mid')}"
    )

    # Candidate summary table
    lines = []
    for r in scored_results:
        ds = r.get("dimension_scores", {})
        lines.append(
            f"- file={r['filename']} | overall={r['overall_score']} | "
            f"skills={ds.get('skills_match','?')} | "
            f"exp={ds.get('experience_relevance','?')} | "
            f"proj={ds.get('project_relevance','?')} | "
            f"matched={len(r.get('matched_required_skills',[]))} | "
            f"missing={len(r.get('missing_required_skills',[]))} | "
            f'summary="{r.get("recruiter_summary","")[:120]}"'
        )
    candidates_summary = "\n".join(lines)

    refined_data = call_llm(
        system_prompt=REFINEMENT_SYSTEM,
        user_prompt=REFINEMENT_USER.format(
            count=len(scored_results),
            job_title=job_title,
            ideal_profile_summary=ideal_summary,
            candidates_summary=candidates_summary,
        ),
        model=model,
        max_tokens=800,
        temperature=0.1,
    )

    # Merge refined scores back into results
    if refined_data and "refined_rankings" in refined_data:
        refined_map = {
            r["filename"]: r for r in refined_data["refined_rankings"]
        }
        for result in scored_results:
            fname = result["filename"]
            if fname in refined_map:
                entry = refined_map[fname]
                result["refined_score"] = max(
                    0, min(100, int(entry.get("refined_score", result["overall_score"])))
                )
                result["delta"] = entry.get("delta", 0)
                result["adjustment_reason"] = entry.get("adjustment_reason", "")

        # Attach panel notes to first result for display
        if scored_results and "panel_notes" in refined_data:
            scored_results[0]["panel_notes"] = refined_data.get("panel_notes", "")

    return scored_results


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_screening_pipeline(
    job_description: str,
    files: list,
    model: str,
    refine: bool = True,
) -> tuple[pd.DataFrame | None, dict, list[dict]]:
    """
    Orchestrates the full 3-step pipeline.
    Returns (DataFrame, ideal_profile, raw_results).
    DataFrame is None if no resumes could be scored.
    """

    # ── Step 1: Generate ideal profile ──────────────────────────
    with st.status("🧠 Step 1 — Analysing job description...", expanded=True) as status:
        ideal_profile = extract_ideal_profile(job_description, model)
        status.update(
            label=f"✅ Ideal profile extracted for: **{ideal_profile['job_title']}**",
            state="complete",
        )

    if not ideal_profile["required_skills"]:
        st.error("❌ Could not read the job description. The AI call failed. See warnings above.")
        return None, ideal_profile, []
    # ── Step 2: Score each resume ────────────────────────────────
    raw_results: list[dict] = []
    score_cache: dict[str, dict] = {}   # hash → result (deduplicates identical files)

    progress_bar = st.progress(0, text="Scoring resumes...")
    total = len(files)

    for i, file in enumerate(files):
        progress_bar.progress((i) / total, text=f"Scoring {file.name} ({i+1}/{total})…")

        fhash = file_hash(file)
        if fhash in score_cache:
            # Duplicate file — reuse cached result with updated filename
            cached = dict(score_cache[fhash])
            cached["filename"] = file.name
            raw_results.append(cached)
            continue

        text = extract_text(file)
        if not text.strip():
            st.warning(f"⚠️ No extractable text in {file.name} — skipping.")
            continue

        scored = score_resume_against_profile(text, ideal_profile, file.name, model)
        score_cache[fhash] = scored
        raw_results.append(scored)
        time.sleep(10)

    progress_bar.progress(1.0, text="Scoring complete.")

    if not raw_results:
        st.error("❌ No resumes could be scored. Check file formats.")
        return None, ideal_profile, []

    if all(r.get("overall_score", 0) == 0 for r in raw_results):
        st.error("❌ Every resume scored 0. The AI calls failed. Check warnings above.")
        return None, ideal_profile, []
        
    # ── Step 3: Optional refinement ─────────────────────────────
    if refine and len(raw_results) >= 2:
        with st.status("🔄 Step 3 — Refining rankings across all candidates...", expanded=False) as status:
            raw_results = refine_rankings(raw_results, ideal_profile, model)
            status.update(label="✅ Rankings refined.", state="complete")

    # ── Sort & build DataFrame ───────────────────────────────────
    raw_results.sort(
        key=lambda r: r.get("refined_score", r.get("overall_score", 0)),
        reverse=True,
    )

    rows = []
    for rank, r in enumerate(raw_results, start=1):
        final_score = r.get("refined_score", r.get("overall_score", 0))
        ds = r.get("dimension_scores", {})
        rows.append({
            "Rank":           rank,
            "Filename":       r["filename"],
            "Final Score":    final_score,
            "Raw Score":      r.get("overall_score", 0),
            "Δ Refined":      r.get("delta", 0),
            "Skills Match":   ds.get("skills_match", "—"),
            "Experience":     ds.get("experience_relevance", "—"),
            "Projects":       ds.get("project_relevance", "—"),
            "Education":      ds.get("education_fit", "—"),
            "Keywords":       ds.get("keyword_coverage", "—"),
            "Exp. Years":     r.get("experience_years_detected", "—"),
            "Label":          score_label(final_score),
        })

    return pd.DataFrame(rows), ideal_profile, raw_results


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — UI HELPERS
# ═══════════════════════════════════════════════════════════════

def score_label(score: int) -> str:
    if score >= 80: return "Excellent Match"
    if score >= 70: return "Good Match"
    if score >= 55: return "Moderate Match"
    if score >= 35: return "Weak Match"
    return "Poor Match"

def score_css_class(score: int) -> str:
    if score >= 70: return "score-high"
    if score >= 50: return "score-medium"
    return "score-low"

def score_emoji(score: int) -> str:
    if score >= 80: return "⭐"
    if score >= 70: return "✅"
    if score >= 50: return "⚠️"
    return "❌"

def chips_html(items: list, chip_class: str) -> str:
    if not items:
        return "<span style='color:#999'>None</span>"
    return "".join(f'<span class="{chip_class}">{i}</span>' for i in items)


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — SIDEBAR
# ═══════════════════════════════════════════════════════════════

MODELS = {
    "GPT-OSS 120B (Groq)": "openai/gpt-oss-120b",
}

with st.sidebar:
    st.markdown("<h2 style='text-align:center'>📄 HireSense</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#666'>Intelligent Resume Screening</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🤖 Model")
    chosen_label = st.selectbox("AI Model", list(MODELS.keys()), index=0)
    st.session_state.gpt_model = MODELS[chosen_label]

    refine_enabled = st.checkbox(
        "✨ Enable ranking refinement (Step 3)",
        value=True,
        help="Cross-compares all candidates for scoring consistency. Adds one extra LLM call.",
    )

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📤 Upload & Screen", "📊 Results", "ℹ️ How It Works"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🛠️ Pipeline")
    st.markdown("""
1. **JD Analysis** — Extract rubric  
2. **Resume Scoring** — 5-dimension evaluation  
3. **Rank Refinement** — Cross-candidate pass  
    """)

    st.markdown("---")
    if client:
        st.success("✅ OpenRouter Connected")
        st.caption(f"Model: `{st.session_state.gpt_model}`")
    else:
        st.error("⚠️ OpenRouter not connected")
        st.caption("Add OPENROUTER_API_KEY to .env")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — HOME PAGE
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown('<p class="main-header">📄 HireSense</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Production-grade Resume Screening via Structured LLM Prompt Engineering</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🎯", "5-Dimension Scoring", "Skills · Experience · Projects · Education · Keywords"),
        ("⚖️", "Weighted Scoring", "40/30/15/10/5% weights, with penalty rules"),
        ("🧠", "Strict JSON Prompts", "Deterministic extraction, zero hallucination tolerance"),
        ("🏆", "Rank Refinement", "Cross-candidate consistency pass"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{icon}</div>
            <div class="metric-label" style="font-size:1rem;font-weight:bold">{title}</div>
            <div class="metric-label">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Navigate to **Upload & Screen** to start screening.")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — UPLOAD & SCREEN PAGE
# ═══════════════════════════════════════════════════════════════

elif page == "📤 Upload & Screen":
    st.markdown('<p class="section-header">📤 Upload & Screen Resumes</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">📝 <strong>Step 1 — Paste the Job Description</strong></div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description",
        height=260,
        placeholder="Paste the full job description here — the more detail you include, the more accurate the scoring.",
        help="Include must-have skills, experience requirements, responsibilities, and preferred tools.",
    )

    st.markdown('<div class="info-box">📎 <strong>Step 2 — Upload Candidate Resumes</strong></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        start = st.button(
            "🚀 Start Screening",
            type="primary",
            use_container_width=True,
            disabled=not (job_description.strip() and uploaded_files),
        )

    if start:
        if not client:
            st.error("❌ OpenRouter API key missing. Add OPENROUTER_API_KEY to .env.")
        elif not job_description.strip():
            st.error("❌ Please enter a job description.")
        elif not uploaded_files:
            st.error("❌ Please upload at least one resume.")
        else:
            df, ideal_profile, raw_results = run_screening_pipeline(
                job_description,
                uploaded_files,
                st.session_state.gpt_model,
                refine=refine_enabled,
            )

            if df is not None:
                st.session_state.results_df = df
                st.session_state.ideal_profile = ideal_profile
                st.session_state.raw_results = raw_results
                st.session_state.processing_complete = True
                st.success("✅ Screening complete! Go to **Results** to view rankings.")


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — RESULTS PAGE
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Results":
    st.markdown('<p class="section-header">📊 Screening Results</p>', unsafe_allow_html=True)

    if not st.session_state.processing_complete:
        st.info("👆 No results yet. Go to Upload & Screen first.")
        st.stop()

    df          = st.session_state.results_df
    raw_results = st.session_state.raw_results
    ideal       = st.session_state.ideal_profile
    raw_map     = {r["filename"]: r for r in raw_results}

    # ── Ideal profile expander ───────────────────────────────────
    with st.expander("🎯 Ideal Candidate Profile (extracted from JD)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Job Title:** {ideal.get('job_title','N/A')}")
            st.markdown(f"**Seniority:** {ideal.get('seniority_level','N/A')}")
            st.markdown(f"**Min Experience:** {ideal.get('min_experience_years','N/A')} years")
            st.markdown(f"**Education:** {ideal.get('education_requirement','N/A')}")
        with c2:
            st.markdown("**Required Skills:**")
            st.markdown(chips_html(ideal.get("required_skills",[]), "skill-chip"), unsafe_allow_html=True)
            st.markdown("**Preferred Skills:**")
            st.markdown(chips_html(ideal.get("preferred_skills",[]), "skill-chip"), unsafe_allow_html=True)
        if ideal.get("red_flags"):
            st.warning("🚩 Deal-breaker gaps: " + " · ".join(ideal["red_flags"]))

    st.markdown("---")

    # ── Summary metrics ──────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Candidates", len(df))
    c2.metric("Best Score",       f"{df.iloc[0]['Final Score']}%")
    c3.metric("Average Score",    f"{df['Final Score'].mean():.1f}%")
    c4.metric("Strong Matches ≥70", len(df[df["Final Score"] >= 70]))
    c5.metric("Weak Matches <40",   len(df[df["Final Score"] < 40]))

    # Panel notes from refinement step
    if raw_results and "panel_notes" in raw_results[0]:
        st.info(f"🔍 **Hiring Panel Notes:** {raw_results[0]['panel_notes']}")

    st.markdown("---")
    st.markdown("### 🏆 Ranked Candidates")

    # ── Per-candidate cards ──────────────────────────────────────
    for _, row in df.iterrows():
        score  = row["Final Score"]
        raw    = raw_map.get(row["Filename"], {})
        ds     = raw.get("dimension_scores", {})
        delta  = row.get("Δ Refined", 0)
        delta_str = (f" *(refined {'+' if delta >= 0 else ''}{delta})*" if delta != 0 else "")

        with st.container():
            col_rank, col_info, col_bar = st.columns([0.7, 3.5, 1.8])

            with col_rank:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1E88E5,#7E57C2);
                            color:white;width:58px;height:58px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;font-weight:bold">
                    #{int(row['Rank'])}
                </div>""", unsafe_allow_html=True)

            with col_info:
                st.markdown(f"**{row['Filename']}**{delta_str}")
                st.markdown(
                    f"<span class='{score_css_class(score)}'>"
                    f"{score_emoji(score)} {score}% — {score_label(score)}</span>",
                    unsafe_allow_html=True,
                )

                # Dimension bar
                if ds:
                    dim_cols = st.columns(5)
                    dims = [
                        ("Skills\n40%",  "skills_match"),
                        ("Exp.\n30%",    "experience_relevance"),
                        ("Projects\n15%","project_relevance"),
                        ("Education\n10%","education_fit"),
                        ("Keywords\n5%", "keyword_coverage"),
                    ]
                    for dcol, (label_name, key) in zip(dim_cols, dims):
                        val = ds.get(key, 0)
                        color = "#4CAF50" if val >= 70 else ("#FF9800" if val >= 50 else "#F44336")
                        dcol.markdown(
                            f"<div style='text-align:center;font-size:.7rem;color:#888'>{label_name}</div>"
                            f"<div style='text-align:center;font-size:1rem;font-weight:bold;color:{color}'>{val}</div>",
                            unsafe_allow_html=True,
                        )

            with col_bar:
                st.progress(score / 100)
                exp = raw.get("experience_years_detected")
                if exp is not None:
                    st.caption(f"🗓 {exp} yr(s) detected")
                if raw.get("red_flags_triggered"):
                    st.caption(f"🚩 {len(raw['red_flags_triggered'])} red flag(s)")

            # Expandable detail panel
            with st.expander(f"📋 Full Details — {row['Filename']}"):
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("**✅ Matched Required Skills:**")
                    st.markdown(chips_html(raw.get("matched_required_skills",[]), "skill-chip"), unsafe_allow_html=True)
                    st.markdown("**✅ Matched Preferred Skills:**")
                    st.markdown(chips_html(raw.get("matched_preferred_skills",[]), "skill-chip"), unsafe_allow_html=True)
                    st.markdown("**💪 Strengths:**")
                    for s in raw.get("strengths", []):
                        st.markdown(f"- {s}")

                with tc2:
                    st.markdown("**❌ Missing Required Skills:**")
                    st.markdown(chips_html(raw.get("missing_required_skills",[]), "missing-chip"), unsafe_allow_html=True)
                    if raw.get("red_flags_triggered"):
                        st.markdown("**🚩 Red Flags:**")
                        for f_ in raw["red_flags_triggered"]:
                            st.markdown(f"- {f_}")
                    st.markdown("**⚠️ Concerns:**")
                    for c in raw.get("concerns", []):
                        st.markdown(f"- {c}")

                if raw.get("recruiter_summary"):
                    st.markdown(
                        f'<div class="reasoning-box">🤖 <strong>Recruiter Summary:</strong> '
                        f'{raw["recruiter_summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                if raw.get("adjustment_reason") and raw.get("delta", 0) != 0:
                    st.caption(f"🔧 Refinement: {raw['adjustment_reason']}")

            st.markdown("---")

    # ── Score breakdown table ────────────────────────────────────
    st.markdown("### 📊 Score Breakdown Table")
    table_cols = [
        "Rank", "Filename", "Final Score", "Δ Refined",
        "Skills Match", "Experience", "Projects", "Education",
        "Keywords", "Exp. Years", "Label",
    ]
    st.dataframe(df[table_cols], use_container_width=True, hide_index=True)

    # ── Score distribution chart ─────────────────────────────────
    st.markdown("### 📈 Score Distribution")
    chart_df = df[["Filename", "Final Score"]].set_index("Filename")
    st.bar_chart(chart_df)

    # ── CSV export ───────────────────────────────────────────────
    st.download_button(
        "📥 Download Results as CSV",
        df.to_csv(index=False),
        file_name="hiresense_results.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — HOW IT WORKS PAGE
# ═══════════════════════════════════════════════════════════════

elif page == "ℹ️ How It Works":
    st.markdown('<p class="section-header">ℹ️ How It Works</p>', unsafe_allow_html=True)
    st.markdown("""
### 🏗️ 3-Step LLM Pipeline

**Step 1 — JD Analysis (`extract_ideal_profile`)**  
The job description is sent to the LLM with a strict JSON-only prompt. It returns a structured
scoring rubric containing required skills, preferred skills, seniority level, minimum experience,
education requirement, responsibilities, domain keywords, and deal-breaker red flags.
This becomes the single source of truth for all subsequent scoring.

**Step 2 — 5-Dimension Resume Scoring (`score_resume_against_profile`)**  
Each resume is individually evaluated against the rubric across five weighted dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Skills Match | **40%** | % of required (and preferred) skills explicitly present |
| Experience Relevance | **30%** | Relevance & length of past roles to this JD |
| Project Relevance | **15%** | Alignment of projects/work samples to responsibilities |
| Education Fit | **10%** | Degree + field match to requirement |
| Keyword Coverage | **5%** | Domain-specific terminology coverage |

**Penalty rules** are applied on top of the weighted score:
- Each missing required skill: −5 pts from skills_match
- Each triggered red flag: −8 pts from overall
- Seniority mismatch: −10 pts from overall

**Step 3 — Ranking Refinement (`refine_rankings`)**  
All scored candidates are sent together for a comparative review. The model applies small
adjustments (±8 pts max) where scores appear inconsistent relative to the full pool — e.g., a
strong all-rounder that was underscored against a one-dimensional candidate.

---

### 💡 Why this beats naive keyword matching

Keyword matching scores a resume higher if it's written in the style of the JD, even if skills
are absent. This system instead:
- Scores based on **explicit skill presence** (not text similarity)
- Understands semantic equivalence ("React" ≈ "ReactJS")
- Penalises **missing critical skills** explicitly
- Separates **experience relevance** from raw years count
- Rewards **quantifiable project impact**, not just project length

### ⚡ Performance Design
- **File deduplication**: identical files are scored once and reused
- **Token budgeting**: resumes are capped at 2 500 words before being sent to LLM
- **Caching**: `@st.cache_resource` prevents client re-initialisation on every rerun
    """)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Powered by <strong>OpenRouter LLM API</strong> — Structured Prompt Engineering</p>
    <p style="font-size:.8rem">HireSense • 3-Step AI Screening Pipeline</p>
</div>
""", unsafe_allow_html=True)
