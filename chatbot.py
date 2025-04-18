import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain

# 🌐 Pagina instellingen
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")

# 💅 Stijl toevoegen
st.markdown("""
    <style>
        body { font-family: 'Open Sans', sans-serif; }
        .swap-header {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
        }
        .swap-title h1 {
            font-family: 'Quicksand', sans-serif;
            font-weight: bold;
            font-size: 1.5rem;
            color: #005F9E;
            margin: 0;
        }
        .swap-sub {
            font-size: 1rem;
            color: #000;
            margin-top: 0.2rem;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🧾 Titel
st.markdown("""
    <div class="swap-header">
        <div class="swap-title">
            <h1>🚗 Stel je vraag aan onze Swap Assistent!</h1>
            <div class="swap-sub">Snel antwoord over leasecontracten en het aanbieden van je auto</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 🔐 OpenAI API Key ophalen
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe bij 'Edit secrets'.")
    st.stop()

# 📂 Unzip vectorstore als het nog niet uitgepakt is
zip_path = "faiss_klantvragen_db.zip"
extract_path = "faiss_klantvragen_db"
if not os.path.exists(extract_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
    else:
        st.error("❌ Zipbestand 'faiss_klantvragen_db.zip' niet gevonden.")
        st.stop()

# 📚 Laad vectorstore
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        extract_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore(openai_api_key)

# 🤖 LLM instellen
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# ✨ Custom prompt met verplichte variable {context}
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Je bent de AI-assistent van Swap Je Lease. Help gebruikers helder, vriendelijk en kort met vragen over leaseoverdracht.
Gebruik geen moeilijke woorden en spreek de gebruiker aan met 'je'.
Geef indien nodig concrete stappen of een voorbeeld.

Context:
{context}

Vraag:
{question}
    """
)

# 🧠 RetrievalQA instellen
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 💬 Invoerveld voor vragen
vraag = st.text_input("Wat wil je weten?", placeholder="Bijv. Hoe kan ik mijn leasecontract overzetten?")
if vraag:
    with st.spinner("Even kijken..."):
        resultaat = qa_chain.invoke({"query": vraag})
        st.success(resultaat["result"])