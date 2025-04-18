import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🌐 Pagina-instellingen
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")

# 💅 Quicksand SemiBold + aangepaste styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@600&display=swap" rel="stylesheet">
<style>
    * {
        font-family: 'Quicksand', sans-serif;
    }
    .title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #005F9E;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 1rem;

    .stTextInput > div > input {
        font-size: 16px;
        padding: 0.5rem;
        border: 1px solid #ccc;
        border-radius: 6px;
        outline: none;
        transition: box-shadow 0.2s ease-in-out;
    }

    .stTextInput > div > input:focus {
        border: 1px solid #005F9E;
        box-shadow: 0 0 0 2px rgba(0, 95, 158, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 🧾 Titel & subtitel
st.markdown("<div class='title'>🚗 Stel je vraag aan onze Swap Assistent!</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Direct antwoord op al je vragen. Helder en zonder gedoe.</div>", unsafe_allow_html=True)

# 🔑 OpenAI API key ophalen
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe via Settings > Secrets.")
    st.stop()

# 📦 FAISS database uitpakken indien nodig
zip_path = "faiss_klantvragen_db.zip"
extract_path = "faiss_klantvragen_db"
if not os.path.exists(extract_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
    else:
        st.error("❌ Databasebestand 'faiss_klantvragen_db.zip' niet gevonden.")
        st.stop()

# 📚 Vectorstore laden
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(extract_path, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore(openai_api_key)

# 📄 Prompt template
prompt = PromptTemplate.from_template("""
Je bent de AI-assistent van Swap Je Lease. Help gebruikers helder, vriendelijk en kort met vragen over leaseoverdracht.
Gebruik geen moeilijke woorden en spreek de gebruiker aan met 'je'.
Geef indien nodig concrete stappen of een voorbeeld.

Context: {context}
Vraag: {question}
""")

# 🤖 OpenAI LLM koppelen
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 💬 Vraaginvoer
vraag = st.text_input("Wat wil je weten?", placeholder="Bijv. Hoe kan ik mijn leaseauto aanbieden?")
if vraag:
    with st.spinner("Even kijken..."):
        resultaat = qa_chain.invoke({"query": vraag})
        st.success(resultaat["result"])
