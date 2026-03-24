import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- FIX: Inisialisasi API ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    # LangChain membutuhkan variabel environment ini secara eksplisit
    os.environ["GOOGLE_API_KEY"] = api_key 
    genai.configure(api_key=api_key)
else:
    st.error("API Key tidak ditemukan di Streamlit Secrets!")
    st.stop()

def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    # --- FIX: Gunakan model embedding yang tepat ---
    # Tambahkan task_type agar model tahu ini untuk mencari dokumen (Retrieval)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # Versi terbaru & paling stabil di 2026
        google_api_key=api_key
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="pdf_chat"
    )
    return vectorstore

# --- FUNGSI RAG ---
def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    # Ganti ini agar lebih stabil
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
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
