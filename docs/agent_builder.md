# Agent Builder UI

The Agent Builder is now a standalone SvelteKit + FastAPI app in
the **`agents7_ui/`** project (sibling to `agents7_framework/` in
this monorepo). It replaced the previous Streamlit-based wizard
(`agents7[ui]` / `agents7.ui`), which has been removed from the
framework.

See `agents7_ui/` for the current builder — its FastAPI backend
(`agents7_ui/backend/`) and SvelteKit frontend
(`agents7_ui/frontend/`), plus `docker-compose` for running both.
