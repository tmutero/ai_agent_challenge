# AI Agent Technical Challenge

**Live app:** <https://aiagentchallenge.streamlit.app> (login required)

* **Part 1**: a RAG assistant over structured documents (CSV, Excel, JSON, PDF, Word) with query planning, hybrid
  retrieval, deterministic aggregation and verified citations.
* **Part 2**: an autonomous agent that reconciles a Google Calendar event with Gmail RSVPs, verifies people against a
  public web directory and DNS, and outputs a cleansed attendee list with every issue flagged.

The design write-up is in [docs/TECHNICAL_SUMMARY.md](docs/TECHNICAL_SUMMARY.md).

## Run locally

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env            # add OPENAI_API_KEY, APP_USERNAME, APP_PASSWORD
.venv/bin/python -m pytest      # 30 offline tests, no API key needed
.venv/bin/streamlit run app.py  # http://localhost:8501
```

Optional regeneration steps:

* `python sample_data/make_sample_data.py` rebuilds the bundled sample documents. Set `RAG_DATA_DIR` to load
  your own folder of documents at start-up instead.
* `python -m part1_rag.evaluate` runs the retrieval evaluation and writes `docs/eval_results.md`, which the app's
  Evaluation tab displays.
* `python -m part2_agent.scenario` rebuilds `static/partner_directory.html`.

## Google setup (live Part 2)

Use a personal or throwaway Google account. Workspace tenants usually block unverified OAuth apps.
1. In Google Cloud Console, create a project and enable the **Gmail API** and the **Google Calendar API**.
2. Set up the OAuth consent screen: choose External, then add your account as a test user. While the app is in
   Testing, refresh tokens expire after 7 days; publish the app to avoid that (verification is not required).
3. Under Credentials, create an OAuth client ID of type **Desktop app**. Download the JSON and save it as
   `credentials.json` in this folder.
4. Run `python -m part2_agent.google_setup auth`. A browser opens for consent, and `token.json` is written.
5. Optional: run `python -m part2_agent.google_setup seed` to create the demo event and insert the demo emails. The
   app also does this automatically before the first live run if no upcoming demo event exists. No invites or emails
   are sent.
6. In the app, choose **Google (live Gmail + Calendar)** as the data source.

`python -m part2_agent.google_setup reset` removes the demo calendar events again. The demo emails have to be deleted
in Gmail (search `subject:(roadshow OR factsheets)`), because the app has no delete permission by design.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo. `.gitignore` excludes `.env`, `token.json` and `.streamlit/secrets.toml`.
2. On share.streamlit.io, create a new app with main file `app.py` and Python 3.12.
3. Under Secrets, add the following:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_CHAT_MODEL = "gpt-5.6"
   APP_USERNAME = "admin"
   APP_PASSWORD = "<choose a strong password>"
   DIRECTORY_URL = "https://<your-app>.streamlit.app/app/static/partner_directory.html"
   ```

4. Optional, for live Google mode: after the Google setup above, add the token as a secret. Running
   `python -c "import json; print(\"GOOGLE_TOKEN_JSON = '''\" + json.dumps(json.load(open('token.json'))) + \"'''\")"`
   prints the line to paste. The **Google (live Gmail + Calendar)** data source then appears in the hosted app, which
   seeds the demo data itself. Without this secret, Part 2 runs on the offline scenario.

## Layout

```text
app.py                   Streamlit entry (3 pages)
common/llm.py            OpenAI client, secrets, structured-output helper
rag_core/pipeline/       reusable hybrid-RAG pipeline (Qdrant, BM25, RRF, eval)
part1_rag/               loaders, query planner, aggregation, grounded answerer, engine, evaluation
sample_data/             bundled sample documents, their generator, and labelled eval cases (eval/)
part2_agent/             scenario, connectors (Google/fixture/web), gmail query evaluator, extraction,
                         cleansing rules, workspace (consolidation + evidence checks), agent loop, google_setup
ui/                      Streamlit pages
static/                  public partner directory page (served at /app/static/)
tests/                   offline tests
docs/                    technical summary
```
