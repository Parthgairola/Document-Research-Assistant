from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# Import Environment Variables
load_dotenv()


# Load LLM Model
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-4B-Instruct-2507",temperature=0)
model = ChatHuggingFace(llm=llm)
print("\nModel Initialized Successfully!")



# Load Embedding Model to create vector embeddings of the chunks
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("\nEmbedding Model Initialized Successfully!")



# Load PDF
data = PyPDFLoader("documents/Generative_Adversarial_Networks.pdf")
docs = data.load()
print("\nDocument loaded Successfully!")



# Split Document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
chunks = splitter.split_documents(docs)
print("\nDocument Splitting Successfull!")




# Store chunks inside "FAISS" vectore database 
vector_store = FAISS.from_documents(documents=chunks,embedding=embedding_model)
print("\nVector Database Created Successfully")



# Intializing retriver
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
print("\nRetriver Initialized")

# Headline

print("\n===== AI-Powered Research Assistant =====\n")

# User Query
while True:
    
    query = input("Ask a question (or type 'exit'): ")
    if query.lower()=="exit":
        break



    # Retrieve relevant chunks based on user query
    context = retriever.invoke(query)
    print("Retrieval Successfull !")


    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system" """You are an AI Research Assistant. Answer the user's questions using only the provided research paper context.

        Rules:

        Give short, clear, and direct answers.
        Do not use information outside the provided context.
        If the answer is not present in the context, say: "I couldn't find the answer in the provided research paper."
        When answering, mention the page number(s) where the information was found.
        Do not make up or assume information."""),

        ("human","{query}\n\n  Context: {context}" )
    ])



    # Map Placeholders to the actual variables
    final_prompt = prompt.invoke({
        "query":query,
        "context":context
    })



    # Invoke LLM Model and generate response
    print("Generating Answer....")
    reponse = model.invoke(final_prompt)
    print(reponse.content)








