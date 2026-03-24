import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. CONFIG ---
st.set_page_config(page_title="AI PDF Explorer (Groq Version)", page_icon="📄")
st.title("📄 PDF AI Explorer (Powered by Groq)")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Masukkan GROQ_API_KEY di Secrets!")
    st.stop()

# --- 2. FUNGSI RAG (STABIL) ---
def process_pdf(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(pages)
        
        # Tetap pakai HuggingFace karena sudah terbukti jalan di servermu
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="pdf_store_groq"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Gagal proses PDF: {e}")
        return None

# --- 3. UI & CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload")
    pdf_file = st.file_uploader("Pilih PDF", type="pdf")
    if st.button("🚀 Proses"):
        if pdf_file:
            with st.spinner("Menganalisis..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Siap!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya isi PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
    with st.chat_message("assistant"):
        with st.spinner("Berpikir cepat dengan Groq..."):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            full_prompt = f"Gunakan info ini: {context}\n\nJawab pertanyaan: {prompt}"
            
            # Coba model terbaru, jika gagal coba model versatile
            try:
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
                response = llm.invoke(full_prompt)
            except:
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
                response = llm.invoke(full_prompt)
                
            answer = response.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("Upload PDF dulu.")
