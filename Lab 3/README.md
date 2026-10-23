# Lab 3 — Prompt Engineering with Groq API

**Assignment:** Six Python implementations based on the previous Groq tutorial.

The lab uses prompt anatomy, reusable templates, system/user roles, zero-shot
classification, response constraints, structured JSON, and iterative refinement.
The default model is `openai/gpt-oss-20b`, matching the earlier tutorial.

## Exercise checklist

| Exercise | Required inputs | Output | File |
| --- | --- | --- | --- |
| 1. Doctor Summary Generator | `patient_notes`, `patient_name` | Fixed Symptoms / Diagnosis / Recommendation sections | `exercise1_doctor_summary.py` |
| 2. Review Classifier | `review_text` | One label: Positive, Negative, Neutral, or Mixed | `exercise2_review_classifier.py` |
| 3. Resume Field Extractor | `resume_text`, `fields_to_extract` | JSON with exactly the requested keys; missing values `null` | `exercise3_resume_extractor.py` |
| 4. Customer Support Reply | `customer_message`, `company_name`, `max_words` | Polite reply respecting the word limit | `exercise4_support_reply.py` |
| 5. Multi-language Translator | `sentence`, `target_language`, `formality` | Translation only, with formal/neutral/informal tone | `exercise5_translator.py` |
| 6. Iterative Content Refiner | `draft_text`, `issue_to_fix` | At least three sequential revision rounds | `exercise6_iterative_refiner.py` |

## Setup

1. Open this folder in VS Code and create a Python virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the environment:

   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`

3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and set your own `GROQ_API_KEY`. Never commit
   or screenshot a real API key. `.env` is excluded by `.gitignore`.

5. Run all six examples:

   ```bash
   python run_all_examples.py
   ```

   Or run any exercise separately, for example:

   ```bash
   python exercise1_doctor_summary.py
   python exercise2_review_classifier.py
   ```

6. Run all offline tests (requires no Groq key):

   ```bash
   python -m unittest discover -s tests -v
   ```

**Important:** The tests verify Python logic, schema construction and formatting
using simulated API responses. They do **not** demonstrate live model accuracy.
Capture your real run output only after setting up your own Groq account.

## Design and prompt techniques

### 1. Doctor Summary Generator

- **Role:** a clinical documentation formatter, not a diagnostic doctor.
- **Constraint:** never infer a diagnosis or treatment from absent evidence.
- **Reliability:** Groq strict JSON schema constrains three fields; Python
  formats them with exact headings, so output structure stays stable.
- **Example fictional input:** patient Asha Patil, cough for 3 days, advised
  rest and fluids, no diagnosis explicitly documented.
- **Expected shape** (illustrative, not a recorded API result):

  ```text
  Patient: Asha Patil

  Symptoms: Cough for three days.
  Diagnosis: Not documented
  Recommendation: Rest and fluids.
  ```

**Privacy/safety:** These examples are fabricated. Don't send identifiable
medical notes to an external API without proper consent and approved safeguards.
The summary is not a medical diagnosis, and factual content still requires
human review.

### 2. Review Classifier

- **Rules:** Positive = primarily praise; Negative = primarily criticism;
  Neutral = factual/no strong sentiment; Mixed = both meaningful praise and
  criticism.
- **Output contract:** the function returns a single category label. The API
  uses a strict JSON enum internally so free-form answers are not accepted.
- **Example:** `Delivery was fast, but packaging was damaged.` → likely `Mixed`.

### 3. Resume Field Extractor

- **Template:** request the list of fields dynamically.
- **Output contract:** strict JSON; only those exact keys; strings or `null`.
- **Example:** a resume with name, email and skills; request `Name`, `Email`,
  `Skills`, `Phone` to demonstrate a missing `Phone` field.
- This program returns a Python dictionary; use `json.dumps(result, indent=2)`
  to print strictly valid JSON.

### 4. Customer Support Reply

- **Persona:** calm, professional and empathetic support agent.
- **Constraint:** at most `max_words` words, no invented refunds or deadlines.
- The function checks the word count, retries up to twice after an overlong
  response, and uses a safe brief fallback if needed. Very small limits can
  force a generic response; this is an explicit trade-off.

### 5. Multi-language Translator

- **Parameters:** target language and one of `formal`, `neutral`, `informal`.
- **Constraint:** preserve meaning, names and key facts; produce only the
  translated sentence. Human review may still be needed for nuanced language.

### 6. Iterative Content Refiner

- **Input:** a draft plus **one specific issue** (for example, `grammar`).
- **Round 1:** fix that issue.
- **Round 2:** find remaining instances of the same issue.
- **Round 3:** final focused consistency review.
- Each round's output becomes the next round's input. The function returns all
  intermediate drafts so the iteration can be demonstrated.

## What to submit

1. Push the six exercise files, helper, examples, tests and this README to the
   **existing** repository named `generative_ai_<student-name-or-roll-no>`.
2. After running with your own Groq API key, take screenshots of representative
   console outputs if your instructor expects execution evidence. Do not show
   your `.env` or key in screenshots.
3. Confirm that `.env` is not tracked with `git status`.
4. Share the repository link in the required format:

   `https://github.com/<username>/generative_ai_<student-name-or-roll-no>`

Sample local Git commands **inside your existing repository** after copying
this folder into it:

```bash
git add lab3_prompt_engineering
git commit -m "Complete Lab 3 prompt engineering exercises"
git push
```

For exercise-by-exercise commits, add and commit each exercise file separately
along with dependencies, and then push after each commit as instructed.

## References

- [Groq GPT-OSS 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b)
- [Groq structured outputs and JSON schema](https://console.groq.com/docs/structured-outputs)

