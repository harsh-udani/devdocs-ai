
import streamlit as st

from src.ingestion import extract_documentation
from src.file_ingestion import extract_uploaded_document
from src.chunking import chunk_documentation
from src.embeddings import create_embedding_model
from src.retrieval import build_search_index, search_documentation
from src.llm import generate_answer


st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="centered",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #344155;
        color: #F3F4F6;
        border: 1px solid #46546A;
        border-radius: 8px;
        min-height: 42px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #43536B;
        color: white;
        border-color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedding_model():
    return create_embedding_model()


if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(page: dict):
    """Chunk, embed, and index a document."""
    chunks = chunk_documentation(page)
    model = load_embedding_model()
    index, _ = build_search_index(chunks, model)

    # Replace the active document only after indexing succeeds.
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.document_name = page["title"]
    st.session_state.messages = []


st.title("📚 DevDocs AI")
st.caption("Upload a document or add a documentation URL to get started.")


# Both ingestion options
upload_tab, url_tab = st.tabs(
    ["Upload document", "Add documentation URL"]
)


with upload_tab:
    st.subheader("Upload your file")
    st.caption("Supported formats: PDF, DOCX, TXT, MD. Maximum size: 10 MB.")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt", "md"],
        key="document_uploader",
    )

    if st.button("Process uploaded document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a document first.")
        else:
            try:
                with st.spinner("Preparing your document..."):
                    page = extract_uploaded_document(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                    process_document(page)

                st.success("Your document is ready for questions.")

            except Exception as exc:
                st.error(f"Could not process this document: {exc}")


with url_tab:
    st.subheader("Add a public documentation page")

    with st.form("url_form"):
        url = st.text_input(
            "Documentation URL",
            placeholder="https://docs.python.org/3/tutorial/",
        )
        url_submitted = st.form_submit_button(
            "Load documentation",
            use_container_width=True,
        )

    if url_submitted:
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Preparing your documentation..."):
                    page = extract_documentation(url.strip())
                    process_document(page)

                st.success("Documentation is ready for questions.")

            except Exception as exc:
                st.error(f"Could not load this URL: {exc}")


# Shared question-answer interface

# Professional chat interface
st.divider()

st.subheader("Ask your documentation")
st.caption(
    "Answers are generated from your indexed content using semantic search."
)

if not st.session_state.document_name:
    st.info("Add a document or public documentation URL to begin.")
else:
    st.markdown(
        f"**Active knowledge source:** "
        f"`{st.session_state.document_name}`"
    )

    # Suggested questions for first-time users
    if not st.session_state.messages:
        st.markdown("##### Try asking")

        suggestions = [
            "Summarize this document",
            "What are the key technical details?",
            "Find important requirements",
        ]

        cols = st.columns(3)

        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f"suggestion_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()

    # Render previous conversation
    for message in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(message["question"])

        with st.chat_message("assistant"):
            st.markdown(message["answer"])

    # Accept a new question
    pending_question = st.session_state.pop(
        "pending_question", None
    )

    question = st.chat_input(
        "Ask anything about your documentation..."
    )

    question = question or pending_question

    if question:
        if st.session_state.index is None:
            st.warning("Please process a document first.")
        else:
            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Searching for an answer..."):
                        model = load_embedding_model()

                        results = search_documentation(
                            query=question,
                            chunks=st.session_state.chunks,
                            model=model,
                            index=st.session_state.index,
                            top_k=3,
                        )

                        if results:
                            answer = generate_answer(
                                question, results
                            )
                        else:
                            answer = (
                               "I couldn't find sufficiently relevant information "
                                "in the active documentation to answer this question. "
                              "Try rephrasing your question or adding more relevant content."
                            )

                    st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )

            except Exception:
                st.error(
                    "Unable to generate an answer. "
                    "Please try again or check the application logs."
                )
