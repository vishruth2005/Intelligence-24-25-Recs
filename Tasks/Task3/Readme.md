# Hogwarts Q&A: Harry Potter and the Prisoner of Azkaban

## Description

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions about "Harry Potter and the Prisoner of Azkaban". It uses natural language processing and machine learning techniques to provide contextually accurate responses to queries about characters, spells, locations, and magical events from the book.

## Features

- RAG-based question-answering system
- Web interface for easy interaction
- Stylized responses matching the tone of the book
- Retrieval of relevant context and metadata

## Project Structure

```
Task3/
│
├── myenv/
├── data/
│   └── harry.pdf  # The source PDF of "Harry Potter and the Prisoner of Azkaban"
│
├── templates/
│   └── index.html  # Frontend HTML template
│
├── main.py  # Main application file
├── requirements.txt  # Project dependencies
└── README.md  # This file
```

## Prerequisites

- Python 3.8+
- FastAPI
- Uvicorn (for serving the FastAPI application)
- LangChain
- Sentence Transformers
- ChromaDB
- PyPDF2
- Ollama (with llama3 model installed)

## Setup

1. Clone the repository:

   ```
   git clone <repository-url>
   cd Intelligence-24-25-Recs
   cd Tasks
   cd Task3
   ```

2. Activate the virtual environment:

   ```
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   ```

   If this doesn't work delete the myenv folder and:

   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   pip install pypdf
   ```

4. Ensure you have the Ollama server running with the llama3 model available.

## Running the Application

1. Start the FastAPI server:

   ```
   uvicorn main:app --reload
   ```

2. Open a web browser and navigate to `http://localhost:8000` to access the Hogwarts Q&A interface.

## Usage

1. Enter your question about "Harry Potter and the Prisoner of Azkaban" in the text input field.
2. Click the "Ask" button or press Enter to submit your query.
3. The system will process your question and display the answer along with relevant context.

## Technical Details

- The application uses a custom text splitting function to divide the book into chapters and smaller chunks.
- Embeddings are generated using the 'all-MiniLM-L6-v2' model from Sentence Transformers.
- ChromaDB is used as the vector store for efficient similarity searches.
- The LLM used for generating responses is Ollama with the llama3 model.
- The frontend is built with HTML, CSS, and JavaScript, styled to match the magical theme of Harry Potter.

## Acknowledgments

- J.K. Rowling for the magical world of Harry Potter
- The open-source community for the amazing tools and libraries used in this project
