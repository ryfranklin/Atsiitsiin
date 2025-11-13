# Atsiitsʼiin — Second Brain Scaling Plan

This document captures the staged roadmap for growing Atsiitsʼiin from a local prototype to a secure, multi-user cloud product.

## Phase 0 — Foundation ✅

- Local prototype functional in Python.
- Data persisted in Snowflake.
- OpenAI API via LiteLLM integrated.
- Notes and NoteChunk entities implemented.

## Phase 1 — Local UI (MVP Interface)

**Goal:** Create a usable UI for personal refinement.

**Tech options:** Streamlit (recommended), Gradio, or lightweight FastAPI + HTML. Manage configuration with `.env` + `config.yaml`.

**Features:**

- View, search, and edit notes.
- Display note chunks in an expandable interface.
- “Save to Snowflake” button.
- Session memory panel (recent prompts + responses).

**Deliverable:** Local app (`streamlit run main.py`) that persists to Snowflake.

## Phase 2 — Personal Cloud Access

**Goal:** Reach the app from anywhere, securely and privately.

1. Containerise with Docker.
2. Deploy to Render, Railway, or Fly.io.
3. Create subdomain `brain.ms3dm.tech`.
4. Add password or Auth0-protected access.
5. Manage secrets via platform tools or `.envrc`.

**Deliverable:** Running instance at `https://brain.ms3dm.tech` with basic authentication.

## Phase 3 — Feature Refinement & Data Expansion

**Goal:** Add intelligence and structured recall.

- Implement vector embeddings (`NOTE_CHUNKS.CONTENT_EMBEDDING`).
- Add semantic search and similarity scoring.
- Introduce tagging (`#work`, `#personal`, etc.).
- Enable nightly summarisation job (Prefect or Snowflake Tasks).

**Deliverable:** Second brain that remembers context and summarises itself.

## Phase 4 — Multi-User & Public Access

**Goal:** Support multiple users with secure authentication and rate limits.

- **Backend:** FastAPI (Python).
- **Frontend:** Streamlit or Next.js.
- **Auth:** Auth0 or Supabase.
- **Database:** Snowflake (multi-tenant schema).
- **Rate limiting:** Redis (Upstash) via `fastapi-limiter`.

**Deployment:** Subdomain routing (`brain.ms3dm.tech`), JWT authentication, role-based access (ADMIN/USER).

**Deliverable:** Multi-user system with authentication, isolation, and per-user limits.

## Phase 5 — Production-Grade Infrastructure

**Goal:** Harden, automate, and scale securely.

1. CI/CD pipeline (GitHub Actions → Render).
2. Centralised logging (Datadog, CloudWatch, or OpenTelemetry).
3. Rate limiting + usage metrics in `USAGE_METRICS` table.
4. Domain security via Cloudflare.
5. Optional Stripe integration for paid plans.

**Deliverable:** Automated, secure, multi-user platform under the ms3dm.tech brand.

