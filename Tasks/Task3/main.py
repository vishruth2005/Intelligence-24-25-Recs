from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Initialize FastAPI app
app = FastAPI()

# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define input model for the FastAPI request
class QueryRequest(BaseModel):
    query: str

# Custom function to split documents by chapters and then into smaller chunks
def split_by_chapters_and_chunks(text, chunk_size=200):
    # Regex pattern for chapters
    chapter_pattern = re.compile(r'(Chapter \d+|CHAPTER \d+)', re.IGNORECASE)
    chapters = re.split(chapter_pattern, text)
    
    # Create the RecursiveCharacterTextSplitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)  # Adjust overlap if needed
    
    # Split each chapter into smaller chunks
    documents = []
    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i]
        chapter_content = chapters[i + 1]
        
        # Split chapter content into smaller chunks
        chunks = text_splitter.split_text(chapter_content)
        
        # Create a Document object for each chunk and assign chapter metadata
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"chapter": chapter_title.strip()}))
    
    return documents

# Load the PDF document
loader = PyPDFLoader(r"data/harry.pdf")
data = loader.load()

# Split the loaded PDF data into chapters
docs = []
for doc in data:
    docs.extend(split_by_chapters_and_chunks(doc.page_content))

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create HuggingFace embeddings wrapper
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use the embeddings in Chroma vectorstore
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

# Set the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Extract a few sentences from the PDF for stylistic reference
sample_text = docs[3].page_content[:500]  # Example: First 500 characters from the fourth document chunk

# System prompt with style instructions
system_prompt = (
    "You are an assistant tasked with answering questions using the following context. "
    "Ensure that your response is concise and styled in the same tone and speech pattern "
    "as the following excerpt from the provided context: \n\n"
    f"Style guide excerpt: '{sample_text}'\n\n"
    "Answer the following question using this style. If you don't know the answer, say "
    "you don't know. Limit the answer to ten sentences.\n\n"
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
        llm=Ollama(model="llama3", temperature=0.3),
        prompt=prompt_template
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get the response from the RAG chain
    response = rag_chain.invoke({"input": query})

    # Get context and metadata from retrieved documents
    retrieved_docs = retriever.get_relevant_documents(query)
    context_and_metadata = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]

    # Return the response and metadata as JSON
    return {
        "answer": response.get("answer", "No answer found"),
        "context_and_metadata": context_and_metadata
    }