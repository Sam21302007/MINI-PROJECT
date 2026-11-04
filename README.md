# Review integrity analyzer

A simple Streamlit app for:

- Uploading review screenshots, extracting text locally with Tesseract OCR, and scoring it.
- Uploading a CSV of reviews and downloading scored results.
- Showing suspicious/genuine scores and highlighting words that influenced the model.

No API or key is needed. The bundled 70 reviews are synthetic demo examples, not verified
real-world labels. Scores are experimental triage suggestions, not proof that a review is
fake or genuine.

## Run on your Windows computer

Tesseract OCR must be installed for screenshot analysis. This project can find it at
`C:\Program Files\Tesseract-OCR\tesseract.exe` or through PATH. CSV analysis does not
need Tesseract.

From `D:\MINI PROJECT`, run:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

The app trains and saves its model automatically the first time if model files are
missing. Open the local URL printed by Streamlit.

## Push the project to GitHub

1. Sign in to GitHub and create a **new empty repository**. Do not initialize it with a
   README, license, or `.gitignore` because these already exist in this project.
2. Open PowerShell in `D:\MINI PROJECT` and run:

```powershell
git init
git add .
git status
```

3. Before committing, check `git status`: make sure it does not include `.venv`,
   `models/*.pkl`, screenshots, private review exports, keys, or other private files.
   The project's `.gitignore` excludes common local/generated files.
4. Commit and connect to your new GitHub repository (replace the URL with the one GitHub
   shows for your repository):

```powershell
git commit -m "Initial review analyzer app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Do not upload customer data or secrets to a public repository. Use a private repository
if you are unsure whether the training samples or project details should be public.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub using the steps above.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and authorize
   access to the repository.
3. Choose **Create app**, select the repository, branch `main`, and main file path
   `app.py`, then deploy.
4. In the deploy settings, select Python 3.12. Streamlit Cloud installs Python packages
   from `requirements.txt` and system packages from `packages.txt` (including Tesseract
   OCR). On first startup, the app trains its model from `data/sample_reviews.csv`;
   allow the initial launch to finish.
5. Test the deployed page with a screenshot and a CSV.

If deployment fails, open the app's **Manage app** page and inspect its logs. Check
that `app.py`, `train.py`, `review_monitor.py`, `image_ocr.py`, `requirements.txt`,
`packages.txt`, and `data/sample_reviews.csv` are all in GitHub.

## Privacy on deployment

When running locally, processing stays on your computer. When hosted on Streamlit Cloud,
uploaded screenshots and CSVs are sent to the hosted app for processing. Do not upload
private or personal review data unless your privacy and hosting arrangements permit it.

## Model limitation

The 70 bundled examples are synthetic. A larger synthetic set is still not evidence of
real-world accuracy. For real moderation, obtain a large, representative and
appropriately licensed labeled dataset, evaluate it on reviews separated by product or
reviewer, and keep a human in the decision loop. Do not automatically remove reviews
or penalize people based on this prototype.

If this is for a patent submission, consider qualified advice about prior art and
filing before public disclosure; filing implications vary by jurisdiction.
