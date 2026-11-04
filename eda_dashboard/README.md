# Movie EDA Dashboard

This is a separate Plotly Dash app for the 12 EDA plots in
`Movie_Analysis_Combined_ML (1).ipynb`. It does not run the notebook's
descriptive analysis, hypothesis tests, or machine-learning sections.

## Run locally

From the project root, install the app dependencies and start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\app.py
```

Open <http://127.0.0.1:8050>. Keep `movies_raw.csv` beside `app.py`.
If `movie_project_output/movies_cleaned.csv` is present, the dashboard uses
that cleaned file; otherwise it derives the EDA columns from the raw CSV.
The raw file is used to leave originally missing genres out of genre plots,
matching the notebook's EDA logic.

The year, genre, and original-language filters update all 12 plots. Financial
plots use `budget_analysis` and `revenue_analysis` and exclude values below
$1,000, matching the notebook's EDA rules.

## Deploy on Render

1. Push the project folder to a GitHub repository, including `app.py`,
   `requirements.txt`, and `movies_raw.csv`. The cleaned CSV is optional.
2. In Render, create a **New Web Service** connected to that repository.
3. Set **Root Directory** to the repository root.
4. Set **Build Command** to `pip install -r requirements.txt`.
5. Set **Start Command** to `gunicorn app:server`.
6. Choose a Python runtime and deploy. The app reads the same relative cleaned
   CSV path on the server.
