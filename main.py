from fastapi import FastAPI
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

app = FastAPI(title="Rursee-Schifffahrt Chatbot API")

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY")

# Загружаем всё при старте
with open("rursee_context.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(text)
docs = [Document(page_content=chunk) for chunk in chunks]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prices_chunk = ""
for doc in docs:
    if "Fahrpreise – Einzelstrecken Schiff" in doc.page_content:
        prices_chunk = doc.page_content
        break

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=ANTHROPIC_KEY,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_template("""
Du bist ein hilfreicher Assistent der Rursee-Schifffahrt.
Beantworte die Frage NUR auf Basis des folgenden Kontexts.
Wenn Preise gefragt werden, berechne den Gesamtpreis und zeige die Rechnung.

Kontext:
{context}

Frage: {question}

Antwort:
""")

def get_context(question):
    docs_found = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs_found)
    context += "\n\n" + prices_chunk
    return context

rag_chain = (
    {"context": lambda x: get_context(x), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Модель запроса
class Question(BaseModel):
    question: str

# Endpoints
@app.get("/")
def root():
    return {"message": "Rursee-Schifffahrt Chatbot API is running!"}

@app.post("/ask")
def ask(body: Question):
    answer = rag_chain.invoke(body.question)
    return {
        "question": body.question,
        "answer": answer
    }
