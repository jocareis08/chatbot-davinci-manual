from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import os

app = FastAPI()

# Substitua pela sua chave da OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Carrega o manual
loader = PyPDFLoader("Manual_DaVinci Resolve_19_1.pdf")
docs = loader.load_and_split()

# Cria base vetorial
vectordb = Chroma.from_documents(docs, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), retriever=vectordb.as_retriever())

# Modelo de entrada da pergunta
class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar")
def perguntar(p: Pergunta):
    resposta = qa.run(p.pergunta)
    return {"resposta": resposta}
