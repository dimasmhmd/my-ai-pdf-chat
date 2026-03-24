import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. CONFIG ---
st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")
st.title("📄 PDF AI Explorer (Stable)")

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Masukkan GEMINI_API_KEY di Secrets!")
    st.stop()

# --- 2. FUNGSI RAG (TANPA GOOGLE EMBEDDING) ---
def process_pdf(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        
        # SOLUSI: Pakai HuggingFace (Gratis & Lokal di Server)
        # Ini tidak akan pernah kena Error 404 API Google
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="pdf_store"
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
    if st.button("🚀 Proses Dokumen"):
        if pdf_file:
            with st.spinner("Menganalisis (Mohon tunggu sebentar)..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Analisis Selesai!")
        else:
            st.warning("Pilih file dulu.")

# Tampilkan Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya isi PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                full_prompt = f"Gunakan info ini: {context}\n\nJawab pertanyaan: {prompt}"
                
                # --- FIX 404: Mencoba beberapa variasi nama model ---
                success = False
                # Daftar kemungkinan nama model yang diterima di berbagai region
                model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                
                for m_name in model_names:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=m_name, 
                            temperature=0,
                            # Memaksa menggunakan API versi 1 (Stable) bukan v1beta
                            client_options={"api_version": "v1"} 
                        )
                        response = llm.invoke(full_prompt)
                        answer = response.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        success = True
                        break # Berhenti jika berhasil
                    except Exception as e:
                        continue # Coba model berikutnya jika gagal
                
                if not success:
                    st.error("Gagal terhubung ke Google AI. Kemungkinan besar region server Streamlit ini tidak didukung. Coba buat API Key baru atau gunakan VPN saat deploy ulang.")
    else:
        st.info("Upload PDF dulu.")
