import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI PDF Chatbot 2026",
    page_icon="📄",
    layout="wide"
)

# --- 2. PENGATURAN API KEY ---
# Mengambil API Key dari Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key 'GEMINI_API_KEY' tidak ditemukan! Silakan atur di Settings > Secrets.")
    st.stop()

# --- 3. FUNGSI PEMPROSESAN PDF (RAG) ---
def process_pdf(uploaded_file):
    try:
        # Simpan file sementara agar bisa dibaca loader
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF menggunakan PyPDF
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        # Split teks menjadi potongan kecil (Chunking)
        # Agar AI tidak overload dan pencarian lebih akurat
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(pages)
        
        # Inisialisasi Embeddings (Menggunakan HuggingFace agar lebih stabil)
        # Model ini mengubah teks menjadi koordinat angka (vektor)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Simpan ke Vector Store (ChromaDB) di dalam RAM
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="pdf_chat_db"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Gagal memproses PDF: {e}")
        return None

# --- 4. ANTARMUKA PENGGUNA (UI) ---
st.title("📄 AI PDF Assistant")
st.markdown("---")

# Inisialisasi riwayat pesan di Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar untuk area kontrol
with st.sidebar:
    st.header("Upload Dokumen")
    uploaded_pdf = st.file_uploader("Pilih file PDF Anda", type="pdf")
    
    if st.button("🚀 Proses Dokumen"):
        if uploaded_pdf:
            with st.spinner("Sedang menganalisis isi PDF..."):
                st.session_state.vectorstore = process_pdf(uploaded_pdf)
                st.success("Dokumen siap! Silakan bertanya.")
        else:
            st.warning("Silakan unggah file PDF dulu.")
    
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

# Menampilkan Riwayat Chat (Gaya Bubble Chat)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LOGIKA TANYA JAWAB ---
if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen ini..."):
    # Tampilkan input user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cek apakah dokumen sudah diproses
    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban dalam dokumen..."):
                # 1. Cari potongan teks yang paling relevan (Similarity Search)
                docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 2. Kirim Instruksi ke Gemini (Generation)
                try:
                    model = genai.GenerativeModel('models/gemini-1.5-flash')
                except:
                    # Backup jika v1beta minta nama tanpa prefix
                    model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Prompt Engineering: Memberi konteks agar AI tidak halusinasi
                full_prompt = f"""
                Anda adalah asisten AI yang bertugas membantu menjawab pertanyaan berdasarkan DOKUMEN yang diberikan.
                
                ATURAN:
                1. Gunakan HANYA informasi dari DOKUMEN di bawah ini.
                2. Jika jawaban tidak ada di dalam dokumen, katakan: "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang Anda unggah."
                3. Jawablah dengan sopan dan jelas.

                DOKUMEN:
                {context}

                PERTANYAAN USER: 
                {prompt}
                """
                
                try:
                    response = model.generate_content(full_prompt)
                    answer = response.text
                    st.markdown(answer)
                    
                    # Simpan jawaban assistant ke riwayat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error dari Gemini API: {e}")
    else:
        st.info("💡 Tips: Silakan upload dan klik 'Proses Dokumen' di sidebar terlebih dahulu.")
