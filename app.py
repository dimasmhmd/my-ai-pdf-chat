import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. KONFIGURASI HALAMAN & API ---
st.set_page_config(page_title="AI PDF Chatbot 2026", layout="centered")
st.title("📄 PDF AI Explorer")
st.markdown("Tanya jawab dengan dokumen Anda menggunakan RAG & Gemini 1.5 Flash.")

# Mengambil API Key dari Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⚠️ API Key 'GEMINI_API_KEY' tidak ditemukan di Secrets!")
    st.stop()

# --- 2. FUNGSI INTI (PROSES PDF) ---
def process_pdf(uploaded_file):
    try:
        # Simpan file sementara agar bisa dibaca PyPDFLoader
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        # Split teks menjadi potongan kecil (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(pages)
        
        # Inisialisasi Model Embedding (Gratis dari HuggingFace)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Simpan ke Vector Store (ChromaDB) di dalam RAM saja
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses PDF: {e}")
        return None

# --- 3. LOGIKA CHAT & SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar untuk upload
with st.sidebar:
    st.header("📁 Menu Dokumen")
    pdf_file = st.file_uploader("Upload file PDF", type="pdf")
    if st.button("🚀 Proses & Pelajari"):
        if pdf_file:
            with st.spinner("Sedang menganalisis dokumen..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Analisis selesai! Silakan bertanya.")
        else:
            st.warning("Pilih file PDF terlebih dahulu.")

# Menampilkan Riwayat Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Pertanyaan User
if prompt := st.chat_input("Apa yang ingin Anda ketahui?"):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respon AI
    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                # 1. Cari potongan teks paling relevan (Retrieval)
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 2. Kirim ke Gemini (Generation)
                model = genai.GenerativeModel('gemini-1.5-flash')
                full_prompt = f"""
                Anda adalah asisten AI yang ahli. Gunakan potongan teks di bawah ini sebagai sumber informasi utama untuk menjawab pertanyaan. 
                Jika jawaban tidak ada dalam teks, katakan bahwa Anda tidak mengetahuinya berdasarkan dokumen tersebut.

                SUMBER INFORMASI:
                {context}

                PERTANYAAN: 
                {prompt}
                """
                
                response = model.generate_content(full_prompt)
                full_response = response.text
                st.markdown(full_response)
        
        # Simpan ke riwayat
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.info("Silakan unggah dan klik 'Proses' di sidebar untuk mulai bertanya.")
