from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from transformers import pipeline

# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# -----------------------------
# Load Vector Database
# -----------------------------

db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)


print("Vector database loaded")


# -----------------------------
# Retriever
# -----------------------------

retriever = db.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# -----------------------------
# Load LLM
# -----------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200
)


llm = HuggingFacePipeline(
    pipeline=generator
)


# -----------------------------
# RAG Function
# -----------------------------

def ask_question(question):

    docs = retriever.invoke(question)


    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )


    prompt = f"""
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    response = llm.invoke(prompt)


    return response