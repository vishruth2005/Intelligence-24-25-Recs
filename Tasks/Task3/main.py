from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# Initialize FastAPI app
app = FastAPI()

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define input model for the FastAPI request
class QueryRequest(BaseModel):
    query: str

# Load the PDF document
loader = PyPDFLoader(r"data/harry.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create HuggingFace embeddings wrapper
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use the embeddings in Chroma vectorstore
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

# Set the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Extract a few sentences from the PDF for stylistic reference
sample_text = docs[3].page_content[:500]  # Example: First 300 characters from the first document chunk

# System prompt with style instructions
system_prompt = (
    "You are an assistant tasked with answering questions using the following context. "
    "Ensure that your response is concise and styled in the same tone and speech pattern "
    "as the following excerpt from the provided context: \n\n"
    f"Style guide excerpt: '{sample_text}'\n\n"
    "Answer the following question using this style. If you don't know the answer, say "
    "you don't know. Limit the answer to three sentences.\n\n"
    "{context}"
)

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# Serve the HTML page
@app.get("/")
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API route for querying
@app.post("/query")
async def query_rag(request: QueryRequest):
    query = request.query
    if not query:
        return JSONResponse(content={"error": "Query cannot be empty"}, status_code=400)

    # Create the retrieval and question-answer chain
    question_answer_chain = create_stuff_documents_chain(
        llm=Ollama(model="llama2",temperature=0.2),
        prompt=prompt_template
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get the response from the RAG chain
    response = rag_chain.invoke({"input": query})

    # Return the response as JSON
    return {"answer": response.get("answer", "No answer found")}