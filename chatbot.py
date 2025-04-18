import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain

# 📄 Zet als eerste de pagina-instellingen
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")

# 🎨 Stijl
st.markdown("""
    <style>
        * {
            font-family: 'Quicksand', sans-serif;
        }
        h1 {
            font-size: 1.8rem;
            font-weight: bold;
            color: #005F9E;
        }
        .subtitle {
            font-size: 1rem;
            color: #444;
            margin-bottom: 2rem;
        }
        .stTextInput > div > input {
            font-size: 16px;
            padding: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🧾 Titel en subtitel
st.markdown("🚗 **Stel je vraag aan onze Swap Assistent!**")
st.markdown("<div class='subtitle'>Snel antwoord over leasecontracten en het aanbieden van je auto</div>", unsafe_allow_html=True)

# 🔐 OpenAI key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe via 'Settings > Secrets'.")
    st.stop()

# 📦 Zip met FAISS-vectorstore uitpakken als nodig
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

# 🤖 LLM en retrieval chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
qa_chain = RetrievalQA(combine_documents_chain=combine_docs_chain, retriever=vectorstore.as_retriever())

# 💬 Vraag
vraag = st.text_input("Wat wil je weten?", placeholder="Bijv. Hoe kan ik mijn leasecontract overzetten?")
if vraag:
    with st.spinner("Even kijken..."):
        resultaat = qa_chain.invoke({"query": vraag})
        st.success(resultaat["result"])
