import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

# 🔧 Pagina setup
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")

# 🎨 Stijl
st.markdown("""
    <style>
        body { font-family: 'Open Sans', sans-serif; }
        .swap-title h1 {
            font-family: 'Quicksand', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            color: #005F9E;
            margin: 0;
        }
        .swap-sub {
            font-family: 'Quicksand', sans-serif;
            font-weight: 500;
            font-size: 1rem;
            color: #000;
            margin-top: 0.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🧾 Titel
st.markdown("""
    <div class="swap-title">
        <h1>🚗 Stel je vraag aan onze Swap Assistent!</h1>
        <div class="swap-sub">Snel en simpel antwoord over leasecontracten en auto-aanbod</div>
    </div>
""", unsafe_allow_html=True)

# 🔐 API key ophalen
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe via 'Edit secrets'.")
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

# 📦 Vectorstore laden
@st.cache_resource
def load_vectorstore(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        extract_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore(openai_api_key)

# 🧠 Prompt
custom_prompt = PromptTemplate.from_template("""
Je bent de AI-assistent van Swap Je Lease. Help gebruikers helder, vriendelijk en kort met vragen over leaseoverdracht.
Gebruik geen moeilijke woorden en spreek de gebruiker aan met 'je'.
Geef indien nodig concrete stappen of een voorbeeld.

Context: {context}

Vraag: {question}
""")

# 🔗 Chains en retrieval
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=custom_prompt)

# 🔎 Slimmere multi-query retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-3.5-turbo")
)

# 💬 Vraag input
vraag = st.text_input("Wat wil je weten?", placeholder="Bijv. Hoe kan ik mijn leasecontract overzetten?")
if vraag:
    with st.spinner("Even kijken..."):
        documenten = multiquery_retriever.get_relevant_documents(vraag)
        resultaat = combine_docs_chain.invoke({"context": documenten, "question": vraag})
        st.success(resultaat)
