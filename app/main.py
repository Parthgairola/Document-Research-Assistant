from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace,  HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
load_dotenv()

class DocumentResearchAssistant:
    def __init__(self):

        # Initialize Model
        llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-4B-Instruct-2507")
        self.model = ChatHuggingFace(llm=llm)

        # Initialize Embedding Model
        self.embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    # Ingestion
    def ingestion(self,PDF_PATH:str):

        # Load PDF
        data = PyPDFLoader(PDF_PATH)
        documents = data.load()

        # Split PDF
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
        chunks = splitter.split_documents(documents) 

        # Store Chunks in vector store
        self.vector_store = FAISS.from_documents(chunks,self.embedding_model)

    # Retrieval
    def retrieval(self,query:str):

        self.retriever = self.vector_store.as_retriever(search_type = "similarity",search_kwargs = {"k":4})

        context = self.retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in context)

        prompt = ChatPromptTemplate([
            ("system",f""" You are a document research assistant.
            Answer the user's query using ONLY the information provided in the retrieved document context.
            Rules:
            1. Do not use outside knowledge.
            2. If the answer is not present in the context, say: "I couldn't find the answer in the provided document."
            3. Do not make up or assume information.
            4. Give clear, concise, and accurate answers.
            5. When possible, mention the relevant section or page from the context.
            """
             ),

            ("human",f"query : {query} , context : {context}")
        ])
        
        final_prompt= prompt.invoke({})
        answer = self.model.invoke(final_prompt)

        return query , context , answer
        

assistant = DocumentResearchAssistant()

assistant.ingestion("../documents/Generative_Adversarial_Networks.pdf")

assistant.retrieval("Name 3 Types of Gans ")








    





        

        

            