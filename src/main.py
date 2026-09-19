from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
load_dotenv()

class DocumentResearchAssistant:
    def __init__(self):

        # Initialize Model
        self.model = ChatGroq(model="openai/gpt-oss-20b" , max_tokens=4096)

        # Initialize Embedding Model
        self.embedding_model = CohereEmbeddings(model = "embed-v4.0")

        # Initialize ChromaDB
        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )

    # Ingestion
    def ingestion(self, PDF_PATH: str):

        # Load PDF
        data = PyPDFLoader(PDF_PATH)
        documents = data.load()

        # Split PDF
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        # Add chunks to Chromadb
        self.vector_store.add_documents(chunks)

    # Retrieval
    def retrieval(self, query: str):

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        documents = self.retriever.invoke(query)

        # For RAGAS : List of chunks
        contexts = [doc.page_content for doc in documents]

        # For LLM : Combined string
        context = "\n\n".join(contexts)

        prompt = ChatPromptTemplate([
            ("system", f""" You are a document research assistant.
            Answer the user's query using ONLY the information provided in the retrieved document context.
            Rules:
            1. Do not use outside knowledge.
            2. If the answer is not present in the context, say: "I couldn't find the answer in the provided document."
            3. Do not make up or assume information.
            4. Give clear, concise, and accurate answers.
            5. When possible, mention the relevant section or page from the context.
            """),

            ("human", f"query : {query} , context : {context}")
        ])

        final_prompt = prompt.invoke({})
        answer = self.model.invoke(final_prompt)

        return query, answer.content, contexts


assistant = DocumentResearchAssistant()