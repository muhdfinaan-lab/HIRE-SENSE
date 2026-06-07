# HireSense — AI-Powered Resume Screener

> Production-grade resume screening via a structured 3-step LLM pipeline. Beats naive keyword matching with semantic skill evaluation, weighted scoring, and cross-candidate ranking refinement.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![OpenRouter](https://img.shields.io/badge/OpenRouter-API-purple)

---

##  Demo

Upload a job description + candidate resumes → get ranked results with dimension-level scores, skill gap analysis, and recruiter summaries in seconds.

---

##  How It Works — 3-Step Pipeline

### Step 1 — JD Analysis (`extract_ideal_profile`)
The job description is sent to the LLM with a strict JSON-only prompt. It returns a structured scoring rubric containing required skills, preferred skills, seniority level, minimum experience, education requirement, responsibilities, domain keywords, and deal-breaker red flags. This becomes the single source of truth for all subsequent scoring.

### Step 2 — 5-Dimension Resume Scoring (`score_resume_against_profile`)
Each resume is individually evaluated against the rubric across five weighted dimensions:

| Dimension | Weight | What It Measures |
|---|---|---|
| Skills Match | **40%** | % of required (and preferred) skills explicitly present |
| Experience Relevance | **30%** | Relevance & length of past roles to the JD |
| Project Relevance | **15%** | Alignment of projects/work samples to responsibilities |
| Education Fit | **10%** | Degree + field match to requirement |
| Keyword Coverage | **5%** | Domain-specific terminology coverage |

**Penalty rules applied on top of the weighted score:**
- Each missing required skill: −5 pts from `skills_match`
- Each triggered red flag: −8 pts from `overall_score`
- Seniority mismatch (e.g. Junior → Senior role): −10 pts from `overall_score`

### Step 3 — Ranking Refinement (`refine_rankings`)
All scored candidates are sent together for a comparative review. The model applies small adjustments (±8 pts max) where scores appear inconsistent relative to the full pool — e.g., a strong all-rounder that was underscored against a one-dimensional candidate.

---

## Why This Beats Naive Keyword Matching

| Naive Keyword Matching | HireSense |
|---|---|
| Scores resumes that _look like_ the JD | Scores based on explicit skill presence |
| No semantic understanding | Understands equivalences ("React" ≈ "ReactJS") |
| No penalty for missing skills | Explicit penalty per missing required skill |
| Raw years count as experience | Measures relevance, not just length |
| No project quality signal | Rewards quantifiable impact and metrics |

---

## Features

- **Multi-format support** — PDF, DOCX, and plain text resumes
- **5-dimension scoring** with configurable weights
- **Penalty rules** for missing skills, red flags, and seniority mismatch
- **Ranking refinement** — cross-candidate consistency pass
- **File deduplication** — identical files scored once, reused
- **Token budgeting** — resumes capped at 2,500 words before LLM call
- **CSV export** — download full results table
- **Multiple LLM backends** via OpenRouter (GPT-4o, Claude, Llama, Phi-3)
- **Recruiter summaries** — 2–3 sentence objective evaluation per candidate

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hiresense.git
cd hiresense
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
Get your key at [openrouter.ai](https://openrouter.ai).

### 4. Run the app
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
openai
python-docx
PyPDF2
pandas
python-dotenv
```

> Full list in `requirements.txt`

---

## Supported Models

| Label | Model ID |
|---|---|
| GPT-4o mini (fast) | `openai/gpt-4o-mini` |
| GPT-4o (best quality) | `openai/gpt-4o` |
| Claude 3 Haiku (fast · cheap) | `anthropic/claude-3-haiku` |
| Llama 3 70B (open source) | `meta-llama/llama-3-70b-instruct` |
| Phi-3 Mini 128k (free) | `microsoft/phi-3-mini-128k-instruct:free` |

Switch models anytime from the sidebar — no code changes needed.

---

## Project Structure

```
hiresense/
├── app.py              # Main Streamlit application
├── .env                # API key (not committed)
├── .env.example        # Template for .env
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Performance Design

- `@st.cache_resource` — prevents OpenRouter client re-initialisation on every rerun
- SHA-256 file hashing — deduplicates identical resume files before scoring
- 2,500-word resume cap — keeps token usage predictable and cost-controlled
- 1,500-word JD cap — prevents prompt overflow on verbose job descriptions

---

## Output

Each candidate card shows:
- Overall score (raw + refined) with label (Excellent / Good / Moderate / Weak / Poor)
- Dimension-level breakdown (5 scores)
- Matched and missing required skills (colour-coded chips)
- Red flags triggered
- Strengths and concerns
- AI recruiter summary
- Refinement delta with reason

Full results are also available as a downloadable CSV.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |

---


---

<div align="center">
  <strong>Built with Streamlit · Powered by OpenRouter LLM API</strong>
</div>

