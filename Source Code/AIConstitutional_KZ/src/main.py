import os
import tempfile
import json
from datetime import datetime
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List

DATA_PATH = "chroma_db"
BASE_LAW_FILE = "constitution.txt"
LOG_DIR = "chat_logs"
ALLOWED_FORMATS = [".txt", ".pdf", ".docx"]

def setup_session():
    defaults = {
        "chat_memory": [],
        "doc_store": None,
        "user_uploads": [],
        "context_pref": "Only Constitution",
        "current_chat": None,
        "active_log": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def extract_constitution_articles() -> List[Document]:
    raw_text = TextLoader(BASE_LAW_FILE, encoding='utf-8').load()
    parsed = []
    temp_content, article_id = "", ""
    for entry in raw_text[0].page_content.split('\n'):
        entry = entry.strip()
        if entry.startswith("\u0421\u0442\u0430") or entry.startswith("Article"):
            if temp_content:
                parsed.append(assemble_article(temp_content, article_id))
            parts = entry.split()
            article_id = parts[1].replace(".", "") if len(parts) > 1 else "Unknown"
            temp_content = entry + "\n"
        else:
            temp_content += entry + "\n"
    if temp_content:
        parsed.append(assemble_article(temp_content, article_id))
    return parsed

def assemble_article(text: str, num: str) -> Document:
    return Document(
        page_content=text.strip(),
        metadata={"source": "Kazakhstan Constitution", "type": "law", "article": num}
    )

def parse_uploaded_documents(uploaded_files) -> List[Document]:
    sections = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    for doc in uploaded_files:
        extension = os.path.splitext(doc.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            temp.write(doc.getbuffer())
            temp_path = temp.name
        try:
            if extension == ".txt":
                loader = TextLoader(temp_path, encoding="utf-8")
            elif extension == ".pdf":
                loader = PyMuPDFLoader(temp_path)
            elif extension == ".docx":
                loader = UnstructuredWordDocumentLoader(temp_path)
            else:
                continue
            content = loader.load()
            st.session_state.user_uploads.append(content[0].page_content)
            parts = splitter.split_documents(content)
            for part in parts:
                part.metadata.update({"source": doc.name, "type": "upload"})
            sections.extend(parts)
        except Exception as err:
            st.error(f"Couldn't load {doc.name}: {str(err)}")
        finally:
            os.unlink(temp_path)
    return sections

def init_vector_storage(documents: List[Document] = None):
    encoder = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(DATA_PATH) and not documents:
        return Chroma(persist_directory=DATA_PATH, embedding_function=encoder)
    else:
        return Chroma.from_documents(documents=documents, embedding=encoder, persist_directory=DATA_PATH)

def custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a legal assistant specialized in the Constitution of Kazakhstan.
        Answer the question using ONLY the provided context below. If the answer is not found in the context,
        simply reply: "⚠️ This topic is not covered in the Constitution of Kazakhstan."

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

def list_saved_chats():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    return sorted([f.replace(".json", "") for f in os.listdir(LOG_DIR) if f.endswith(".json")])

def read_chat(chat_key):
    path = os.path.join(LOG_DIR, f"{chat_key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def store_chat(chat_key, content):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{chat_key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

def discard_chat(chat_key):
    path = os.path.join(LOG_DIR, f"{chat_key}.json")
    if os.path.exists(path):
        os.remove(path)

def inject_chat_to_vectorstore(history, store):
    cache = []
    prev_q = None
    for item in history:
        if item["role"] == "user":
            prev_q = item["content"]
        elif item["role"] == "assistant" and prev_q:
            record = Document(
                page_content=f"Q: {prev_q}\nA: {item['content']}",
                metadata={"source": "chatlog", "type": "qa"}
            )
            cache.append(record)
    if cache:
        store.add_documents(cache)

def run_app():
    st.set_page_config(page_title="KZ Constitution Helper", layout="wide")
    st.title("\U0001F1F0\U0001F1FF Constitution QA: Ask Your Legal Questions")
    setup_session()

    with st.sidebar:
        st.subheader("\U0001F4C2 Add Documents")
        files = st.file_uploader("Drop files here", type=ALLOWED_FORMATS, accept_multiple_files=True)
        if files:
            sections = parse_uploaded_documents(files)
            if sections:
                law_articles = extract_constitution_articles()
                st.session_state.doc_store = init_vector_storage(sections + law_articles)
                st.success("Files loaded successfully.")

        st.subheader("\U0001F4D1 Context Settings")
        st.session_state.context_pref = st.radio("Use: ", ["Only Constitution", "Include Uploaded Documents"])

        st.subheader("\U0001F4AC Sessions")
        options = list_saved_chats()
        st.session_state.active_log = st.selectbox("Load previous chat", options) if options else None

        if st.session_state.active_log and st.session_state.active_log != st.session_state.current_chat:
            st.session_state.current_chat = st.session_state.active_log
            st.session_state.chat_memory = read_chat(st.session_state.current_chat)
            if st.session_state.doc_store:
                inject_chat_to_vectorstore(st.session_state.chat_memory, st.session_state.doc_store)

        if st.button("\U0001F5D1️ Remove Chat") and st.session_state.current_chat:
            discard_chat(st.session_state.current_chat)
            st.session_state.current_chat = None
            st.session_state.chat_memory = []
            st.rerun()

        if st.button("➕ Start New Chat"):
            new_key = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.current_chat = new_key
            st.session_state.chat_memory = []
            store_chat(new_key, [])
            st.rerun()

    if not st.session_state.doc_store:
        law_articles = extract_constitution_articles()
        st.session_state.doc_store = init_vector_storage(law_articles)

    for message in st.session_state.chat_memory:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Type your legal question here..."):
        st.session_state.chat_memory.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            model = OllamaLLM(model="llama3.2", temperature=0.1)
            fetcher = st.session_state.doc_store.as_retriever(search_kwargs={"k": 5})
            qa_engine = RetrievalQA.from_chain_type(
                llm=model,
                retriever=fetcher,
                return_source_documents=True,
                chain_type_kwargs={"prompt": custom_prompt()}
            )

            if st.session_state.context_pref == "Include Uploaded Documents" and st.session_state.user_uploads:
                joined = "\n---\n".join(st.session_state.user_uploads)
                modified_query = f"{question}\nAdditional context:\n{joined}"
            else:
                modified_query = question

            output = qa_engine({"query": modified_query})
            evidence = output.get("source_documents", [])
            reply = output["result"].strip()

            if not evidence or all("kazakhstan" not in doc.metadata.get("source", "").lower() for doc in evidence):
                reply = "⚠️ This topic is not covered in the Constitution of Kazakhstan."

            st.session_state.chat_memory.append({"role": "assistant", "content": reply})
            st.markdown(reply)

            if evidence:
                with st.expander("\U0001F4D6 Sources"):
                    for src in evidence:
                        article = src.metadata.get("article")
                        if article:
                            st.markdown(f"**Article {article}** — *{src.metadata.get('source', '')}*\n\n{src.page_content}")
                        else:
                            st.markdown(f"**{src.metadata.get('source', '')}**\n\n{src.page_content}")

            doc = Document(
                page_content=f"Q: {question}\nA: {reply}",
                metadata={"source": "chatlog", "type": "qa"}
            )
            st.session_state.doc_store.add_documents([doc])

            if st.session_state.current_chat:
                store_chat(st.session_state.current_chat, st.session_state.chat_memory)

if __name__ == "__main__":
    run_app()
