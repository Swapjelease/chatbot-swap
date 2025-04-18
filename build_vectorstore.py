import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import openai

# Zet je OpenAI API key via omgevingsvariabele
openai.api_key = os.getenv("OPENAI_API_KEY")

# CSV inlezen
df = pd.read_csv("Uitgebreide_Swap_Je_Lease_FAQ.csv")
df["tekst"] = "Vraag: " + df["Vraag van klant"].astype(str) + "\nAntwoord: " + df["Antwoord klantenservice"].astype(str)

# Chunken
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

chunks = []
for tekst in df["tekst"]:
    chunks.extend(split_text(tekst))

docs = [Document(page_content=chunk) for chunk in chunks]

# Embeddings + save
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_klantvragen_db")

print("✅ FAISS-vectorstore aangemaakt en opgeslagen!")
