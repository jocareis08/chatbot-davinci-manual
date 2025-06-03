from fastapi import FastAPI
from pydantic import BaseModel
import os

# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import PyPDFLoader

app = FastAPI()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Carregamento do manual (temporariamente desativado para o deploy funcionar)
# loader = PyPDFLoader("/app/Manual_DaVinci Resolve_19_1.pdf")
# docs = loader.load_and_split()
# vectordb = Chroma.from_documents(docs, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
# qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), retriever=vectordb.as_retriever())

class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar")
def perguntar(p: Pergunta):
    # resposta = qa.run(p.pergunta)
    resposta = "Sistema funcionando. PDF ainda não carregado."
    return {"resposta": resposta}
