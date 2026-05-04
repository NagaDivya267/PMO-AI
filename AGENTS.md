# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Enterprise PMO (Project Management Office) AI Assistant — a single-service Streamlit app (`app.py`) with data visualizations, an AI chatbot (OpenAI GPT-4o-mini), and a RAG pipeline over bundled PDFs (FAISS + LangChain). All data is file-based (CSVs and PDFs in the repo root).

### Running the app

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

The dashboards (Director / PM / CIO) and alert engine work without an OpenAI key. The chat assistant and RAG features require a valid `OPENAI_API_KEY` in `.streamlit/secrets.toml`.

### Key caveats

- **NumPy compatibility**: `faiss-cpu==1.8.0` requires `numpy<2`. The update script pins this. If you see `numpy.core.multiarray failed to import`, run `pip install "numpy<2"`.
- **Secrets file**: Streamlit reads secrets from `.streamlit/secrets.toml` (git-ignored). Create it with `OPENAI_API_KEY = "sk-..."` to enable AI chat features. Without it, the app starts but the LLM/embedding calls will error on first use.
- **No tests, linter, or CI**: The repo has no test suite, no linter config, and no CI pipeline. Validation is manual (run the app, interact with the UI).
- **No build step**: The app is run directly via `streamlit run app.py`.
