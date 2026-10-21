# AI Agent Technical Challenge

* **Part 1**: a RAG assistant over structured documents (CSV, JSON, tabular PDF) with query planning, hybrid
  retrieval, deterministic aggregation and verified citations.
* **Part 2**: an autonomous agent that reconciles a Google Calendar event with Gmail RSVPs, verifies people against a
  public web directory and DNS, and outputs a cleansed attendee list with every issue flagged.

The design write-up is in [docs/TECHNICAL_SUMMARY.md](docs/TECHNICAL_SUMMARY.md), and the 25-minute walkthrough is in
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## Run locally
```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env            # add OPENAI_API_KEY
.venv/bin/python -m pytest      # 30 offline tests, no API key needed
.venv/bin/streamlit run app.py  # http://localhost:8501
```
Optional regeneration steps:
* `python sample_data/make_sample_data.py` rebuilds the bundled sample documents. Set `RAG_DATA_DIR` to load
  your own folder of documents at start-up instead.
* `python -m part1_rag.evaluate` rebuilds `docs/eval_results.md`.
* `python -m part2_agent.scenario` rebuilds `static/partner_directory.html`.

## Google setup (live Part 2)
Use a personal or throwaway Google account. Workspace tenants usually block unverified OAuth apps.
1. In Google Cloud Console, create a project and enable the **Gmail API** and the **Google Calendar API**.
2. Set up the OAuth consent screen: choose External and Testing, then add your account as a test user.
3. Under Credentials, create an OAuth client ID of type **Desktop app**. Download the JSON and save it as
   `credentials.json` in this folder.
4. Run `python -m part2_agent.google_setup auth`. A browser opens for consent, and `token.json` is written.
5. Run `python -m part2_agent.google_setup seed` to create the demo event and insert the demo emails. No invites or
   emails are sent.
6. In the app, choose **Google (live Gmail + Calendar)** as the data source.

`python -m part2_agent.google_setup reset` removes the demo calendar events again.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo. `.gitignore` excludes `.env`, `credentials.json` and `token.json`.
2. On share.streamlit.io, create a new app with main file `app.py` and Python 3.12.
3. Under Secrets, add the following:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_CHAT_MODEL = "gpt-5.6"
   APP_USERNAME = "admin"
   APP_PASSWORD = "<choose a strong password>"
   DIRECTORY_URL = "https://<your-app>.streamlit.app/app/static/partner_directory.html"
   ```
The hosted app runs Part 2 on the offline scenario by default. Live Google mode is demoed locally (see the
limitations section of the technical summary).

## Layout
```
app.py                   Streamlit entry (3 pages)
common/llm.py            OpenAI client, secrets, structured-output helper
rag_core/pipeline/       reusable hybrid-RAG pipeline (Qdrant, BM25, RRF, eval) - vendored from db_sql_explainer
part1_rag/               loaders, query planner, aggregation, grounded answerer, engine, evaluation
sample_data/             bundled sample documents, their generator, and labelled eval cases (eval/)
part2_agent/             scenario, connectors (Google/fixture/web), gmail query evaluator, extraction,
                         cleansing rules, workspace (consolidation + evidence checks), agent loop, google_setup
ui/                      Streamlit pages
static/                  public partner directory page (served at /app/static/)
tests/                   offline tests
docs/                    technical summary, evaluation results, demo script
```
