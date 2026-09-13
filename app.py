from pathlib import Path

import gradio as gr
from config import DEPLOYMENT_MODE, SHARE
from deployment.runtime import gradio_launch_options, social_metadata_app_kwargs
from core.models import initialize_models
from core.state import new_kb_state
from chat.service import chat
from deployment.guardrails import guard_chat
from utils.ids import new_conversation_id
from utils.helpers import prepare_session_submission as prepare_submission
from ui.controls import update_chunk_overlap_limit
from ui.branding import ABOUT_HTML, DEMO_NOTICE_HTML, EVIDENCE_NOTE_HTML, HEADER_HTML, LOGO_PATH, STYLESHEET_PATH
from ui.reset import reset_chat_only, on_mode_change, on_scope_change
from ui.upload import upload_wrapper
from ui.view_toggles import render_indexed_chunks

APP_DIR = Path(__file__).resolve().parent
FAVICON_PATH = APP_DIR / "assets" / "measured-rag-favicon-64.png"
SOCIAL_CARD_PATH = APP_DIR / "assets" / "measured-rag-card.png"

chat = guard_chat(chat)

initialize_models()

gr.set_static_paths(paths=[LOGO_PATH, SOCIAL_CARD_PATH])

with gr.Blocks(css_paths=STYLESHEET_PATH, title="Measured RAG") as app:

    gr.HTML(
        HEADER_HTML,
        elem_classes="mr-full-width",
    )

    if DEPLOYMENT_MODE == "cloud":
        gr.HTML(DEMO_NOTICE_HTML)

    with gr.Row(elem_id="workspace-grid"):
      with gr.Column(scale=35, min_width=0, elem_classes="mr-panel", elem_id="configuration-panel"):
        gr.Markdown(
            "## Knowledge Base\nUpload PDF or Word documents and configure how they are indexed.\n\n"
            '<p class="mr-language-scope"><strong>Language scope:</strong> '
            'v1.0 is optimized and evaluated for English-language documents.</p>',
            elem_classes="mr-section-heading",
        )

        mode_selector = gr.Radio(
            choices=["Single-document", "Multi-document"],
            value="Single-document",
            label="Document Mode",
            info="Replace one document at a time or build a collection.",
        )

        file = gr.File(
            file_types=[".pdf", ".doc", ".docx"],
            file_count="single",
            label="Upload Document",
        )

        gr.Markdown("### Knowledge Base Status", elem_classes="mr-section-heading")
        file_status = gr.Markdown(
            value="*No documents indexed.*",
            sanitize_html=True,
            latex_delimiters=[],
            max_height=300,
            elem_id="file-status",
        )

        with gr.Accordion("Indexing Configuration", open=True, elem_id="indexing-configuration"):
            gr.Markdown("Controls how uploaded documents are segmented before dense and keyword indexing.")
            chunk_size_slider = gr.Slider(
                200, 2000, value=700, step=50,
                label="Chunk Size (characters)"
            )
            chunk_overlap_slider = gr.Slider(
                0, 300, value=100, step=10,
                label="Chunk Overlap (characters)"
            )

        chunk_size_slider.change(
            update_chunk_overlap_limit,
            inputs=[chunk_size_slider, chunk_overlap_slider],
            outputs=[chunk_overlap_slider],
            queue=False,
        )

        query_scope_mode = gr.Radio(
            choices=[
                ("All uploaded documents", "all"),
                ("Only selected documents", "selected"),
            ],
            value="all",
            label="Document Scope",
            visible=False
        )

        selected_docs = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Selected Documents",
            elem_id="selected-docs",
            visible=False
        )

        gr.Markdown(
            "## Retrieval\nFind evidence with dense retrieval, BM25 keyword retrieval, hybrid fusion, and reranking.",
            elem_classes="mr-section-heading",
        )
        retrieval_mode = gr.Radio(
            choices=[
                ("Auto", "auto"),
                ("Semantic", "semantic"),
                ("Keyword", "keyword"),
                ("Balanced", "balanced"),
            ],
            value="auto",
            label="Retrieval Mode",
            info="Auto selects a strategy from the query; Balanced combines dense and BM25 keyword retrieval.",
        )

        with gr.Accordion("Retrieval Configuration", open=True, elem_id="retrieval-configuration"):
            gr.Markdown("Controls candidate retrieval, fusion filtering, reranking, and final context selection.")
            retrieval_top_n = gr.Slider(
                5,
                50,
                value=20,
                step=1,
                label="Retrieval Top-N (per retriever)",
                info="Maximum results returned by each active retriever before fusion.",
            )

            fusion_threshold = gr.Slider(
                0.0,
                1.0,
                value=0.30,
                step=0.01,
                label="Hybrid Fusion Threshold",
                info="Minimum fused candidate score accepted for reranking.",
            )

            rerank_candidate_n = gr.Slider(
                5,
                40,
                value=15,
                step=1,
                label="Reranker Candidate Limit",
                info="Maximum fused candidates evaluated by the reranker.",
            )

            rerank_top_k = gr.Slider(
                1,
                10,
                value=5,
                step=1,
                label="Final Rerank Top-K",
                info="Highest-ranked passages passed to answer generation.",
            )

      with gr.Column(scale=65, min_width=0, elem_classes="mr-panel", elem_id="conversation-panel"):
        gr.Markdown(
            "## Conversation\nAnswers are generated from retrieved evidence and accompanied by optional inspection views.",
            elem_classes="mr-section-heading",
        )

        chatbot = gr.Chatbot(height=500, label="Document Conversation", elem_id="chat-panel")
        query = gr.Textbox(
            label="Question",
            placeholder="Ask a specific question about the indexed document(s)…",
            elem_id="query-input",
        )

        with gr.Row(elem_id="action-row"):
            send = gr.Button("Ask Measured RAG", interactive=True, variant="primary", elem_id="send-button")
            reset_btn = gr.Button("Clear Conversation", variant="secondary", elem_id="reset-button")

        gr.HTML(
            EVIDENCE_NOTE_HTML
        )

        gr.Markdown(
            "## Inspection & Debugging\nInspect indexed chunks, retrieved sources, and diagnostics from the most recent answer.",
            elem_classes="mr-section-heading",
        )

        with gr.Accordion("Indexed Chunks", open=True, elem_classes="mr-inspection"):
            chunks_view = gr.Textbox(
                label="Indexed Chunks",
                show_label=False,
                lines=12,
                visible=True,
                interactive=False,
                elem_id="chunks-panel",
            )

        with gr.Accordion("Retrieved Sources", open=True, elem_classes="mr-inspection"):
            sources = gr.Textbox(
                label="Retrieved Sources",
                show_label=False,
                lines=12,
                visible=True,
                interactive=False,
                elem_id="sources-panel",
            )

        with gr.Accordion("Debug Last Answer", open=True, elem_classes="mr-inspection"):
            debug_box = gr.Textbox(
                label="Debug Last Answer",
                show_label=False,
                lines=12,
                visible=True,
                interactive=False,
                elem_id="diagnostics-panel",
            )

    gr.HTML(
        ABOUT_HTML,
        elem_classes="mr-full-width",
    )

    # chat() returns debug text unconditionally; retain its existing boolean input.
    debug_mode = gr.State(False)
    state_chat = gr.State([])
    state_kb = gr.State(new_kb_state(mode="single"))
    state_chunks = gr.State([])
    state_conversation_id = gr.State(new_conversation_id())
    state_uploaded_files = gr.State([])
    state_pending_query = gr.State("")
    state_query_count = gr.State(0)

    # Mode switching resets everything
    mode_selector.change(
        on_mode_change,
        inputs=[mode_selector, state_kb],
        outputs=[
            state_kb,
            state_chat,
            state_chunks,
            file_status,
            sources,
            debug_box,
            state_conversation_id,
            file,
            selected_docs,
            query_scope_mode,
        ],
    ).then(
        lambda: "",
        outputs=query,
    ).then(
        lambda x: x,
        inputs=state_chat,
        outputs=chatbot,
    ).then(
        render_indexed_chunks,
        inputs=[state_chunks],
        outputs=chunks_view,
    )

    # Scope change
    query_scope_mode.change(
        on_scope_change,
        inputs=[query_scope_mode],
        outputs=[selected_docs],
    )

    file.change(
        lambda: gr.Button(interactive=False),
        outputs=send,
    ).then(
        upload_wrapper,
        inputs=[
            file,
            state_uploaded_files,
            state_kb,
            mode_selector,
            chunk_size_slider,
            chunk_overlap_slider,
        ],
        outputs=[
            state_kb,
            state_chat,
            file_status,
            state_chunks,
            selected_docs,
            state_conversation_id,
            sources,
            debug_box,
            state_uploaded_files,
            file,
        ],
        **({"concurrency_id": "cloud-heavy-work", "concurrency_limit": 1}
           if DEPLOYMENT_MODE == "cloud" else {}),
   ).then(
        lambda _: gr.Button(interactive=True),
        outputs=send,
    ).then(
        lambda: "",
        outputs=query,
    ).then(
        render_indexed_chunks,
        inputs=[state_chunks],
        outputs=chunks_view,
    ).then(
        lambda x: x,
        inputs=state_chat,
        outputs=chatbot,
    )

    # Chat
    send_submission = send.click(
        prepare_submission,
        inputs=[
            query,
            state_chat,
            state_query_count,
        ],
        outputs=[
            chatbot,
            state_pending_query,
            query,
            state_query_count,
        ],
        concurrency_id="query-admission" if DEPLOYMENT_MODE == "cloud" else None,
        concurrency_limit=1,
    )
    (send_submission.success if DEPLOYMENT_MODE == "cloud" else send_submission.then)(
        chat,
        inputs=[
            state_pending_query,
            state_chat,
            state_kb,
            retrieval_mode,
            retrieval_top_n,
            rerank_candidate_n,
            rerank_top_k,
            fusion_threshold,
            debug_mode,
            query_scope_mode,
            selected_docs,
            state_conversation_id,
        ],
        outputs=[
            state_chat,
            state_kb,
            sources,
            debug_box,
            state_conversation_id,
        ],
        **({"concurrency_id": "cloud-heavy-work", "concurrency_limit": 1}
           if DEPLOYMENT_MODE == "cloud" else {}),
    ).then(
        lambda x: x,
        inputs=state_chat,
        outputs=chatbot,
    )

    enter_submission = query.submit(
        prepare_submission,
        inputs=[
            query,
            state_chat,
            state_query_count,
        ],
        outputs=[
            chatbot,
            state_pending_query,
            query,
            state_query_count,
        ],
        concurrency_id="query-admission" if DEPLOYMENT_MODE == "cloud" else None,
        concurrency_limit=1,
    )
    (enter_submission.success if DEPLOYMENT_MODE == "cloud" else enter_submission.then)(
        chat,
        inputs=[
            state_pending_query,
            state_chat,
            state_kb,
            retrieval_mode,
            retrieval_top_n,
            rerank_candidate_n,
            rerank_top_k,
            fusion_threshold,
            debug_mode,
            query_scope_mode,
            selected_docs,
            state_conversation_id,
        ],
        outputs=[
            state_chat,
            state_kb,
            sources,
            debug_box,
            state_conversation_id,
        ],
        **({"concurrency_id": "cloud-heavy-work", "concurrency_limit": 1}
           if DEPLOYMENT_MODE == "cloud" else {}),
    ).then(
        lambda x: x,
        inputs=state_chat,
        outputs=chatbot,
    )

    reset_btn.click(
        reset_chat_only,
        inputs=[state_kb],
        outputs=[
            state_kb,
            state_chat,
            state_chunks,
            file_status,
            sources,
            debug_box,
            state_conversation_id,
        ],
    ).then(
        lambda: "",
        outputs=query,
    ).then(
        lambda: "",
        outputs=state_pending_query,
    ).then(
        lambda x: x,
        inputs=state_chat,
        outputs=chatbot,
    ).then(
        render_indexed_chunks,
        inputs=[state_chunks],
        outputs=chunks_view,
    )

if DEPLOYMENT_MODE == "cloud":
    app.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    app.launch(
        favicon_path=str(FAVICON_PATH),
        app_kwargs=social_metadata_app_kwargs(),
        **gradio_launch_options(
            deployment_mode=DEPLOYMENT_MODE,
            share=SHARE,
        )
    )