# chatbot.py
import os
import zipfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

# 1. Pagina setup
st.set_page_config(page_title="Swap Assistent", page_icon="🚗", layout="wide")

# 2. Stijl en fonts injecteren
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Open Sans', sans-serif;
    }
    .title-wrapper h1 {
        font-family: 'Quicksand', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #005F9E;
        margin: 0;
    }
    .subtitle {
        font-size: 1rem;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header
st.markdown("""
<div class="title-wrapper">
    <h1>🚗 Stel je vraag aan onze Swap Assistent!</h1>
    <div class="subtitle">Direct antwoord op al je vragen. Helder en zonder gedoe.</div>
</div>
""", unsafe_allow_html=True)

# 4. API key ophalen
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key ontbreekt. Voeg deze toe bij 'Edit secrets'.")
    st.stop()

# 5. Unzip vectorstore als nog niet uitgepakt
zip_path = "faiss_klantvragen_db.zip"
unzip_dir = "faiss_klantvragen_db"
if not os.path.exists(unzip_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# 6. Laad vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(
        unzip_dir, embeddings, allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# 7. Conversatie-geheugen instellen
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 8. Template voor de assistent
prompt = ChatPromptTemplate.from_messages([
    ("system", "Je bent de AI-assistent van Swap Je Lease. Antwoord vriendelijk, kort en duidelijk. Gebruik eenvoudige taal en spreek de gebruiker aan met 'je'."),
    ("human", "{question}")
])

# 9. Language Model instellen
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", api_key=openai_api_key)

# 10. Conversational QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=prompt),
    return_source_documents=True,
    verbose=False
)

# 11. Vraag invoer
vraag = st.text_input("Wat wil je weten?", placeholder="Bijv. Hoe zet ik mijn lease over?")

# 12. Verwerking van vraag
if vraag:
    with st.spinner("Even kijken..."):
        try:
            response = qa_chain.invoke({"question": vraag})
            antwoord = response["answer"]
            bronnen = response.get("source_documents", [])

            st.success(antwoord)

            # 13. Bronnen tonen als knoppen
            if bronnen:
                with st.expander("🔍 Gebruikte bronnen"):
                    for doc in bronnen:
                        st.markdown(f"- `{doc.metadata.get('source', 'onbekend')}`")

        except Exception as e:
            st.error(f"Er ging iets mis: {str(e)}")

# 14. Logging idee (optioneel uitbreidbaar met externe logging)
# st.write("[Log]", vraag)
