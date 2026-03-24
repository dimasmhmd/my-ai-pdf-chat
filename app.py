import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- 1. KONFIGURASI ---
st.set_page_config(page_title="AI PDF Chatbot 2026", page_icon="📄")
st.title("📄 AI PDF Explorer (Stable Version)")

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Masukkan GEMINI_API_KEY di Secrets!")
    st.stop()

# --- 2. FUNGSI RAG ---
def process_pdf(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(pages)
        
        # Gunakan HuggingFace untuk embedding agar tidak kena limit API Google di tahap ini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
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
    if st.button("Proses"):
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
            with st.spinner("Berpikir..."):
                # Cari Konteks
                docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Panggil Gemini via LangChain (Lebih Stabil)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                
                full_prompt = f"Gunakan teks berikut untuk menjawab: {context}\n\nPertanyaan: {prompt}"
                
                try:
                    response = llm.invoke(full_prompt)
                    answer = response.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error Akhir: {e}")
    else:
        st.info("Upload PDF dulu.")
