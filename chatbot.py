import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# 🔐 Haal OpenAI API key uit omgevingsvariabele
openai_api_key = os.getenv("OPENAI_API_KEY")

# ❌ Stop als er geen sleutel is
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe als omgevingsvariabele (OPENAI_API_KEY).")
    st.stop()

# 🧠 Streamlit-pagina instellingen
st.set_page_config(page_title="Swap Je Lease Klantenservice", page_icon="🚗")
st.title("🚗 Swap Je Lease Klantenservice")
st.markdown("Stel hier je vraag over het aanbieden of overnemen van een leaseauto.")

# 🔄 Vectorstore laden
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        "faiss_klantvragen_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

# 🧠 Laad vectorstore en AI-model
vectorstore = load_vectorstore(openai_api_key)
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 💬 Vraag van gebruiker
vraag = st.text_input("💬 Typ je vraag:")

# 🧠 Genereer antwoord
if vraag:
    with st.spinner("🔎 Bezig met zoeken..."):
        antwoord = qa_chain.run(vraag)
        st.success(antwoord)

# 📎 Subtiele afsluiter
st.caption("Swap Je Lease Klantenservice")
