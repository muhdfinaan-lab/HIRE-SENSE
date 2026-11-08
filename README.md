# HireSense

AI-assisted resume screening. Paste a job description, upload resumes, and get a ranked list of candidates with a score breakdown and a short explanation for each one.

**Live demo:** https://hire-sense-demo.streamlit.app

> The demo runs on a free API tier with shared limits. If it shows a rate-limit error, wait a few minutes and try again. Please use sample or fake resumes only. Do not upload real people's CVs to the public demo.

---

## What it does

- Reads a job description and turns it into a structured scoring checklist (required skills, preferred skills, experience, education, red flags).
- Reads each resume (PDF or DOCX) and scores it against that checklist.
- Ranks all candidates, with a score out of 100 and a plain-English summary of strengths, concerns and missing skills.
- Runs an optional cross-candidate review that can adjust scores by up to 8 points when the ranking looks inconsistent.
- Exports the results as a CSV file.

## How it works

The app runs a 3-step pipeline:

1. **Job description analysis.** The language model extracts a structured profile from the job description (as JSON).
2. **Resume scoring.** Each resume is scored on five dimensions. The model gives a 0-100 score for each one.
3. **Ranking refinement (optional).** All scores are sent back to the model for a consistency check. Changes are limited to +/- 8 points.

### Scoring weights

| Dimension | Weight |
|---|---|
| Skills match | 40% |
| Experience relevance | 30% |
| Project relevance | 15% |
| Education fit | 10% |
| Keyword coverage | 5% |

The overall score is calculated **in Python code**, not by the model, so it always matches the dimension scores. Each red flag found on a resume subtracts 8 points, up to a maximum of 20.

### Design choices

- Strict JSON prompts, with one automatic retry if the model returns broken JSON.
- Duplicate files are scored once and the result is reused.
- Resume text is capped at 1,500 words to keep token use and cost low.
- If the job description cannot be read, or every resume scores 0, the app shows an error instead of fake results.
- A short pause between resumes keeps the app under free-tier rate limits.

## Tech stack

- Python 3.10+
- Streamlit (interface)
- Groq API through the OpenAI Python SDK (model: `openai/gpt-oss-120b`)
- PyPDF2 and python-docx (file reading)
- pandas (results table and CSV export)
- python-dotenv (local configuration)

## Run it locally

1. Install Python 3.10 or newer.
2. Clone the repo and open the folder:
   ```
   git clone https://github.com/<your-username>/HIRE-SENSE.git
   cd HIRE-SENSE
   ```
3. Install the packages:
   ```
   python -m pip install -r requirements.txt
   ```
4. Get a free API key at [console.groq.com](https://console.groq.com) and create a file named `.env` in the project folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
5. Start the app:
   ```
   python -m streamlit run app.py
   ```
6. Open http://localhost:8501.

Never commit your `.env` file. It is listed in `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (without `.env`).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and choose this repo, the `main` branch and `app.py`.
3. Open **Advanced settings** and paste this into **Secrets**:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Click **Deploy**.

## Project structure

```
HIRE-SENSE/
├── app.py              # Whole app: extraction, prompts, scoring, UI
├── requirements.txt    # Python dependencies
├── .gitignore          # Keeps .env out of the repo
└── README.md
```

## Try it with sample data

The `sample_data/` folder has a job description for a Junior AI Engineer and three fake resumes (strong, medium and weak match). A normal run should rank them in that order.

## Limitations

- **Scores are not exact.** The same resume can score a few points differently between runs. Treat the output as a first-pass ranking and have a person review it.
- **Close candidates can swap places.** The test resumes are far apart in quality. Ranking between near-equal candidates is less reliable.
- **Text-only reading.** Scanned or image-based PDFs have no extractable text and are skipped.
- **Limited bias checks.** The app does not test for bias. It should not be used to make hiring decisions on its own.
- **Privacy.** Resume text is sent to a third-party API. Do not upload real candidate data unless you have permission and have reviewed the provider's data terms.
- **Free-tier limits.** Large batches may hit rate limits. The app waits between resumes, so big runs are slow.

## Possible improvements

- Test the scoring on more realistic, closely matched resumes.
- Add OCR for scanned PDFs.
- Fix the education score for candidates with unrelated but valid degrees.
- Add a score-consistency test that runs the same resume several times.
- Let the user edit the extracted job profile before scoring.

## Author

Built by Muhammed Finan.
