import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

# Cara import yang lebih aman di tahun 2026:
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIG & API SETUP ---
st.set_page_config(page_title="PDF AI Chat 2026", layout="wide")

# Mengambil API Key dari secrets (Streamlit Cloud)
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("API Key tidak ditemukan! Pastikan sudah diatur di Secrets.")

# --- FUNGSI RAG ---
def process_pdf(uploaded_file):
    # Simpan file sementara
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load & Split
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

# --- UI STREAMLIT ---
st.title("📄 PDF AI Chatbot")
st.write("Tanya apapun isi dokumenmu dengan bantuan Google Gemini.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar untuk Upload
with st.sidebar:
    st.header("Upload Dokumen")
    pdf_file = st.file_uploader("Upload PDF Anda", type="pdf")
    if st.button("Proses Dokumen"):
        if pdf_file:
            st.session_state.vectorstore = process_pdf(pdf_file)
            st.success("Dokumen berhasil diproses!")
        else:
            st.warning("Silakan upload file dulu.")

# Menampilkan Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat
if prompt := st.chat_input("Apa yang ingin kamu tanyakan?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Logika Jawaban
    if "vectorstore" in st.session_state:
        # Cari potongan teks relevan
        docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Kirim ke Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        full_prompt = f"Gunakan info ini untuk menjawab: {context}\n\nPertanyaan: {prompt}"
        
        response = model.generate_content(full_prompt)
        answer = response.text
    else:
        answer = "Silakan upload dan proses dokumen PDF terlebih dahulu di sidebar."

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
