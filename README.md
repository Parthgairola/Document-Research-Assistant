# AI-Powered Research Assistant

A basic RAG-based AI Research Assistant that allows users to upload a research paper and ask questions about its content.

## Features

- Loads research papers in PDF format
- Splits the document into smaller chunks
- Creates vector embeddings using Hugging Face
- Stores embeddings in FAISS
- Retrieves relevant chunks based on the user's question
- Generates answers using an LLM
- Provides page numbers for the retrieved information
- Refuses to answer when information is not available in the paper

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

The assistant can answer:

> What are the different types of GANs?

It can also handle questions unrelated to the research paper:

> When is NASA going to space next?

In such cases, it responds that the answer cannot be found in the provided research paper.

## Project Structure

```text
AI_RESEARCH_ASSISTANT/
│
├── documents/
│   └── research_paper.pdf
│
├── main.py
├── requirements.txt
└── README.md