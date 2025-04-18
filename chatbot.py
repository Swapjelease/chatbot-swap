import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🔍 Streamlit pagina setup
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")
st.title("🚗 Stel je vraag aan onze Swap Assistent!")
st.caption("Altijd snel een antwoord.")

# 🔐 API key ophalen
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe bij 'Edit secrets'.")
    st.stop()

# 📂 Vectorstore zip uitpakken
zip_path = "faiss_klantvragen_db.zip"
extract_path = "faiss_klantvragen_db"
if not os.path.exists(extract_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
    else:
        st.error("❌ Zipbestand 'faiss_klantvragen_db.zip' niet gevonden.")
        st.stop()

# 🧠 Laad vectorstore
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        extract_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore(openai_api_key)

# ✨ Prompt met verplicht veld 'context'
custom_prompt = PromptTemplate.from_template("""
Je bent de AI-assistent van Swap Je Lease. Help gebruikers helder, vriendelijk en kort met vragen over leaseoverdracht.
Gebruik geen moeilijke woorden en spreek de gebruiker aan met 'je'.
Geef indien nodig concrete stappen of een voorbeeld.

Context: {context}

Vraag: {question}
""")

# 🧑‍‍🧓 LLM + Retrieval koppelen
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# 💬 Vraag
vraag = st.text_input("Wat wil je weten?", placeholder="Bijv. Hoe kan ik mijn leasecontract overzetten?")
if vraag:
    with st.spinner("Even kijken..."):
        resultaat = qa_chain.invoke({"query": vraag})
        st.success(resultaat["result"])