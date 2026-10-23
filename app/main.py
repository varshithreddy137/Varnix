
import streamlit as st
import pymupdf

from app.services.document_search import (
    chunk_document,
    retrieve_relevant_chunks,
)

from app.services.gemini_service import (
    summarize_document,
    extract_key_points,
    answer_document_question,
)


st.set_page_config(
    page_title="Varnix | Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("Varnix")
st.subheader("Document Intelligence Platform")
st.write(
    "Upload a PDF to extract its text, generate summaries, "
    "identify important information, and ask questions about the document."
)

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()

        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as document:
            page_count = len(document)

            page_texts = [
                {
                    "page": page_number,
                    "text": page.get_text(),
                }
                for page_number, page in enumerate(document, start=1)
            ]

        extracted_text = "\n\n".join(
            item["text"] for item in page_texts
        )

        
        chunks = []

        for item in page_texts:
            for chunk_text in chunk_document(item["text"]):
                chunks.append({
                    "page": item["page"],
                    "text": chunk_text,
                })

        st.success("PDF processed successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total pages", page_count)

        with col2:
            st.metric(
                "Characters extracted",
                len(extracted_text),
            )

        if extracted_text.strip():
            st.subheader("Document Intelligence")

            summary_tab, keypoints_tab, qa_tab, text_tab = st.tabs(
                [
                    "AI Summary",
                    "Key Points",
                    "Ask Your Document",
                    "Extracted Text",
                ]
            )

            
            with summary_tab:
                if st.button(
                    "Generate Summary",
                    key="generate_summary",
                ):
                    with st.spinner(
                        "Varnix is summarizing your document..."
                    ):
                        try:
                            result = summarize_document(extracted_text)
                            st.session_state["summary"] = result

                        except Exception as error:
                            st.error(
                                f"Summary generation failed: {error}"
                            )

                if st.session_state.get("summary"):
                    st.markdown(st.session_state["summary"])

                    st.download_button(
                        "Download Summary",
                        data=st.session_state["summary"],
                        file_name="varnix_summary.md",
                        mime="text/markdown",
                        key="download_summary",
                    )

            
            with keypoints_tab:
                if st.button(
                    "Extract Key Points",
                    key="extract_key_points",
                ):
                    with st.spinner(
                        "Varnix is analyzing your document..."
                    ):
                        try:
                            result = extract_key_points(extracted_text)
                            st.session_state["key_points"] = result

                        except Exception as error:
                            st.error(
                                f"Key-point extraction failed: {error}"
                            )

                if st.session_state.get("key_points"):
                    st.markdown(
                        st.session_state["key_points"]
                    )

                    st.download_button(
                        "Download Key Points",
                        data=st.session_state["key_points"],
                        file_name="varnix_key_points.md",
                        mime="text/markdown",
                        key="download_key_points",
                    )


           
            with qa_tab:
                st.subheader("Chat with Your Document")

                st.write(
                    "Ask questions about your PDF, ask follow-ups, "
                    "and explore the document through conversation."
                )

                # Reset chat when a different PDF is uploaded.
                document_key = (
                    f"{uploaded_file.name}:{len(pdf_bytes)}"
                )

                if (
                    st.session_state.get("qa_document_key")
                    != document_key
                ):
                    st.session_state["qa_document_key"] = document_key
                    st.session_state["qa_history"] = []

                if "qa_history" not in st.session_state:
                    st.session_state["qa_history"] = []

                # Clear conversation without removing the uploaded PDF.
                if st.button(
                    "Clear chat",
                    key="clear_document_chat",
                ):
                    st.session_state["qa_history"] = []
                    st.rerun()

                history = st.session_state["qa_history"]

                # Display previous messages and their citations.
                for message in history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                        if message["role"] == "assistant":
                            sources = message.get("sources", [])

                            if sources:
                                with st.expander("View sources"):
                                    for index, source in enumerate(
                                        sources, start=1
                                    ):
                                        st.markdown(
                                            f"**Source {index} — "
                                            f"Page {source['page']}**"
                                        )
                                        st.write(source["text"])
                                        st.caption(
                                            "Semantic similarity: "
                                            f"{source['score']:.3f}"
                                        )

                # Chat input supports follow-up questions.
                question = st.chat_input(
                    "Ask a question or follow up...",
                    key="document_chat_input",
                )

                if question:
                    # Display and save the user's question.
                    st.session_state["qa_history"].append({
                        "role": "user",
                        "content": question,
                    })

                    with st.chat_message("user"):
                        st.markdown(question)

                    history = st.session_state["qa_history"]

                    # Build a retrieval query using recent context.
                    # This helps resolve follow-ups such as
                    # "Explain the second point."
                    recent_context = []

                    for message in history[-6:]:
                        if message["role"] in ("user", "assistant"):
                            recent_context.append(message["content"])

                    retrieval_query = "\n".join(recent_context)

                    with st.chat_message("assistant"):
                        with st.spinner(
                            "Searching your document and preparing an answer..."
                        ):
                            try:
                                relevant_chunks = retrieve_relevant_chunks(
                                    retrieval_query,
                                    chunks,
                                    top_k=4,
                                )

                                if not relevant_chunks:
                                    answer = (
                                        "I couldn't find sufficiently relevant "
                                        "passages to answer that question. "
                                        "Try asking more specifically about "
                                        "the document."
                                    )
                                else:
                                    # Use the conversation history to
                                    # interpret the current question.
                                    answer = answer_document_question(
                                        question,
                                        relevant_chunks,
                                        conversation_history=history[:-1],
                                    )

                                st.markdown(answer)

                                sources = relevant_chunks

                                if sources:
                                    with st.expander("View sources"):
                                        for index, source in enumerate(
                                            sources, start=1
                                        ):
                                            st.markdown(
                                                f"**Source {index} — "
                                                f"Page {source['page']}**"
                                            )
                                            st.write(source["text"])
                                            st.caption(
                                                "Semantic similarity: "
                                                f"{source['score']:.3f}"
                                            )

                                # Save the answer and source passages
                                # so they remain visible after reruns.
                                st.session_state["qa_history"].append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": [
                                        {
                                            "page": source["page"],
                                            "text": source["text"],
                                            "score": source["score"],
                                        }
                                        for source in sources
                                    ],
                                })

                            except Exception as error:
                                st.error(
                                    f"Question answering failed: {error}"
                                )


            
            with text_tab:
                st.text_area(
                    "Extracted document content",
                    value=extracted_text,
                    height=400,
                )

                st.download_button(
                    "Download Extracted Text",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain",
                    key="download_text",
                )

        else:
            st.warning(
                "No selectable text was found. "
                "This may be a scanned PDF that needs OCR."
            )

    except Exception:
        st.error("Could not process this PDF.")
        st.exception(__import__("sys").exc_info()[1])
