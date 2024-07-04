import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI


TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["OPENAI_API_KEY"] = "SET YOUR API KEY HERE"


def init_session_state():
    variables = ["relevant_documents", "selected_document", "result", "docsearch", "document"]
    for var in variables:
        if not st.session_state.get(var):
            st.session_state[var] = None

def get_relevant_documents(document, question, chunk_size, chunk_overlap):
    if st.session_state.docsearch:
        search_result = st.session_state.docsearch.similarity_search(question)
        return search_result

    tempfile_path = os.path.join(TEMP_DIR.name, document.name)
    with open(tempfile_path, "wb") as f:
        f.write(document.getvalue())

    loader = PyPDFLoader(tempfile_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n"]
    )
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    st.session_state.docsearch = Chroma.from_documents(docs, embeddings)
    search_result = st.session_state.docsearch.similarity_search(question)

    return search_result

def restart_vectorstore():
    if st.session_state.docsearch:
        st.session_state.docsearch.delete_collection()
        st.session_state.docsearch = None

def get_response(context_content, question):
    if st.session_state.get("selected_document"):
        prompt = PromptTemplate.from_template("""
        Use the provided articles delimited by triple quotes to answer questions.
        If the answer cannot be found in the articles, write "I could not find an answer".
        
        Article:
        ```
        {article}
        ```
        
        Question:
        {question}
        """)
   
        prompt_value = prompt.invoke({"article": context_content, "question": question})

        llm = ChatOpenAI(model="gpt-3.5-turbo")
        response = llm.invoke(prompt_value)
        return response.content
    return None

def main():
    
    init_session_state()
    st.set_page_config(page_title="Homework Maker", page_icon=":robot_face:")
    st.title(":robot_face: - Homework Maker")

    st.session_state.document = st.file_uploader(
        on_change=restart_vectorstore(),
        type="pdf",
        label="Upload your document here. Note: it doesn't work with digitized documents.",
    )
    
    with st.form("optionsForm"):
        st.markdown("### Set options and make the question:")
        st.session_state["chunk_size"] = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=1000,
            step=100,
        )
        st.session_state["chunk_overlap"] = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=100,
            step=10,
        )
        st.session_state["temperature"] = st.slider(
            "Temperature",
            min_value=0,
            max_value=100,
            step=10,
        )
        st.session_state["question"] = st.text_input(
            "Enter your question:",
        )
        submitted_options = st.form_submit_button(
            "Confirmar",
            disabled=not st.session_state.document
        )
        if submitted_options:
            with st.spinner("Carregando..."):
                st.session_state.relevant_documents = get_relevant_documents(
                    document=st.session_state.document,
                    question=st.session_state.question,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap
                )

    if st.session_state.relevant_documents:
        st.markdown("## Trechos relevantes encontrados:")
        for i, doc in enumerate(st.session_state.relevant_documents):
            st.markdown(f"#{i + 1}")
            st.markdown(doc.page_content)
            st.divider()

        with st.form("selectDocumentForm"):            
            st.radio(
                "Selecione qual trecho deseja enviar.",
                key="selected_document",
                options=[int(i) for i in range(1, len(st.session_state.relevant_documents) + 1)],
            )

            submitted_request_response = st.form_submit_button(
                "Obter Resposta da IA :robot_face:",
            )

            if submitted_request_response:
                with st.spinner("Carregando..."):
                    st.session_state.result = get_response(
                        context_content=st.session_state.relevant_documents[st.session_state.selected_document - 1].page_content,
                        question=st.session_state.question
                    )
        
    if st.session_state.result:
        st.markdown("# Resultado:")
        st.markdown(st.session_state.result)

if __name__ == "__main__":
    main()