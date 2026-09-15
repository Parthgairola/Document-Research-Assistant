# Document Research Assistant

A RAG-based AI Document Research Assistant that allows users to provide a research paper in PDF format and ask questions about its content.

## Tech Stack

- Python
- LangChain
- Hugging Face
- FAISS
- PyPDF
- RAG

## How It Works

PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → Prompt → LLM → Answer

## Example

The user provides a research paper and asks:

> What are the different types of GANs?

The assistant retrieves relevant information from the document and generates an answer using the LLM.

For questions unrelated to the document:

> When is NASA going to space next?

The assistant responds:

> I couldn't find the answer in the provided document.

## Project Structure

The project contains a `documents` folder for research papers, `main.py` containing the application code, `requirements.txt` for dependencies, and `.gitignore` for excluded files.

```text
DOCUMENT_RESEARCH_ASSISTANT/
│
├── document_research_assistant/
│   ├── documents/
│   │   └── research_paper.pdf
│   ├── main.py
│   ├── README.md
│   └── .gitignore
│
├── .env
└── requirements.txt