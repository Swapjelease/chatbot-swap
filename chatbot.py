import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🌟 Pagina-instellingen
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")

# 📄 Fonts & stijl
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@600&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Quicksand', sans-serif; }
        .title {
            font-size: clamp(1.4rem, 2.2vw, 2rem);
            font-weight: 600;
            color: #005F9E;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #444;
            margin-bottom: 1.5rem;
        }
        .stTextInput > div > input {
            font-size: 16px;
            padding: 0.6rem;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# 📅 Header
st.markdown("""
    <div class="title">🚗 Stel je vraag aan onze Swap Assistent!
    </div>
    <div class="subtitle">Direct antwoord op al je vragen. Helder en zonder gedoe.</div>
""", unsafe_allow_html=True)

# 🔐 API-sleutel
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe via 'Settings > Secrets'.")
    st.stop()

# 📆 FAISS zip uitpakken
zip_path = "faiss_klantvragen_db.zip"
extract_path = "faiss_klantvragen_db"
if not os.path.exists(extract_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
    else:
        st.error("❌ Zipbestand 'faiss_klantvragen_db.zip' niet gevonden.")
        st.stop()

# 🧠 Vectorstore laden
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(extract_path, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore(openai_api_key)

# ✨ Prompt
prompt = PromptTemplate.from_template("""
Je bent de AI-assistent van Swap Je Lease. Help gebruikers helder, vriendelijk en kort met vragen over leaseoverdracht.
Gebruik geen moeilijke woorden en spreek de gebruiker aan met 'je'.
Geef indien nodig concrete stappen of een voorbeeld.

Context: {context}
Vraag: {question}
""")

# 🧥 LLM + QA chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 💬 Vraag stellen
vraag = st.text_input("✍️ Wat wil je weten?", placeholder="Bijv. Hoe kan ik mijn leaseauto aanbieden?")

# 📊 Voorbeelden tonen
with st.expander("💡 Voorbeelden"):
    st.markdown("""
    - Hoe kan ik mijn leasecontract overdragen?
    - Hoe werkt het om een leaseauto van iemand over te nemen?
    - Zijn er kosten verbonden aan het overdragen van mijn leasecontract??
    """)

# ✅ Resultaat tonen
if vraag:
    with st.spinner("💭 Even nadenken..."):
        try:
            resultaat = qa_chain.invoke({"query": vraag})
            st.markdown(f"""
                <div style='background-color:#f0f8ff;padding:1.2rem 1rem;border-radius:8px;
                            border-left:5px solid #005F9E;margin-top:1rem'>
                    <b>✅ Antwoord:</b><br>{resultaat["result"]}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("😔 Oeps! Er ging iets mis. Probeer het later opnieuw.")
