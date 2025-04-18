import os
import streamlit as st
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# 🧠 Streamlit instellingen
st.set_page_config(page_title="Swap Je Lease Assistent", page_icon="🚗")
st.title("🚗 Swap Je Lease Assistent")
st.markdown("Stel hier je vraag over het aanbieden of overnemen van een leaseauto.")

# 🔐 API key ophalen uit omgeving
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe als Streamlit secret (OPENAI_API_KEY).")
    st.stop()

# 📦 Zip uitpakken als vectorstore map nog niet bestaat
if not os.path.exists("faiss_klantvragen_db") and os.path.exists("faiss_klantvragen_db.zip"):
    with zipfile.ZipFile("faiss_klantvragen_db.zip", "r") as zip_ref:
        zip_ref.extractall("faiss_klantvragen_db")

# 🧠 Laad vectorstore
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        "faiss_klantvragen_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

# 🔄 Initialiseer LLM en QA-chain
vectorstore = load_vectorstore(openai_api_key)
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 💬 Vraag van gebruiker
vraag = st.text_input("💬 Typ je vraag:")

if vraag:
    with st.spinner("🧠 Bezig met zoeken..."):
        antwoord = qa_chain.run(vraag)
        st.success(antwoord)

# Subtiele afsluiter
st.caption("Swap Je Lease Klantenservice")
