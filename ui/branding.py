"""Static presentation assets for Measured RAG; no application services."""

from pathlib import Path

STYLESHEET_PATH = Path(__file__).resolve().with_name("styles.css")
LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "measured-rag-mark.svg"

HEADER_HTML = """
        <header id="product-header" aria-labelledby="product-title">
            <div class="mr-header-top">
                <span class="mr-eyebrow">Document intelligence, measured end to end</span>
                <span class="mr-version" aria-label="Version 1.0">v1.0</span>
            </div>
            <div class="mr-title-lockup">
                <img class="mr-logo" src="gradio_api/file=assets/measured-rag-mark.svg" alt="" aria-hidden="true" width="42" height="42">
                <h1 class="mr-title" id="product-title">Measured RAG</h1>
            </div>
            <p class="mr-tagline">Grounded answers. Inspectable retrieval. Reproducible evaluation.</p>
            <p class="mr-positioning">Tune chunking and retrieval · Ask your documents · Trace each answer</p>
            <div class="mr-header-footer">
                <div class="mr-author-row">
                <span class="mr-author"><b>Built by Filipe Braiman Carvalho</b></span>
                <nav class="mr-project-links" aria-label="Author links">
                    <a href="https://github.com/filipe-braiman" target="_blank" rel="noopener noreferrer" aria-label="Filipe Braiman Carvalho on GitHub (opens in a new tab)">GitHub</a>
                    <a href="https://www.linkedin.com/in/filipe-b-carvalho" target="_blank" rel="noopener noreferrer">LinkedIn<span class="sr-only"> (opens in a new tab)</span></a>
                    <a href="mailto:filipebraiman@gmail.com">Contact</a>
                </nav>
                </div>
            </div>
        </header>
        """

ABOUT_HTML = """
        <section id="about-project" aria-labelledby="about-project-title">
            <h2 id="about-project-title">About This Project</h2>
            <p>
                Measured RAG is a lightweight, open-source, production-style system for question answering across PDF and Word documents. Its interface exposes chunking and retrieval controls alongside hybrid dense and keyword retrieval, reranking, grounded answer generation, runtime telemetry, and reproducible benchmarking—making the path from document ingestion to evaluated answer visible, configurable, and deployment-validated.
            </p>
            <p>
                Project-owned code is licensed under GNU AGPL version 3 only.
            </p>
            <nav class="mr-resource-links" aria-label="Project resources">
                <a href="https://github.com/filipe-braiman/measured-rag">View Source on GitHub</a>
                <a href="https://github.com/filipe-braiman/measured-rag/blob/main/benchmarking/publication/qasper-publication-v1-28p/evaluation_report.md">Read the Publication Benchmark</a>
            </nav>
        </section>
        """

EVIDENCE_NOTE_HTML = """
            <p class="mr-system-note" role="note">
                <strong>Evidence-first behavior:</strong> when the indexed material does not support an answer,
                the system may abstain instead of filling the gap.
            </p>
            """

DEMO_NOTICE_HTML = """
    <aside id="portfolio-demo-notice" aria-labelledby="portfolio-demo-title">
        <h2 id="portfolio-demo-title">Public Portfolio Demo</h2>
        <p>This hosted version applies upload, prompt-length, per-session query,
        queue, and concurrency limits to keep shared access reliable. Run the
        open-source project locally for development and full configuration.</p>
    </aside>
"""
