from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace,  HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness , AnswerCorrectness , AnswerRelevancy , AnswerSimilarity, ContextPrecision , ContextRecall


# Load Environment Variables
load_dotenv()

class DocumentResearchAssistant:
    def __init__(self):

        # Initialize Model
        llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-4B-Instruct-2507" , max_new_tokens=2048 , temperature=0.1)
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

        # Returns Document Objects
        self.documents = self.retriever.invoke(query)

        # For RAGAS : RAGAS needs List of chunks
        contexts = [doc.page_content for doc in self.documents] 

        # For LLM : LLM needs combined string
        context = "\n\n".join(contexts)

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
        response = self.model.invoke(final_prompt)

        answer = response.content

        return query , answer , contexts
        

assistant = DocumentResearchAssistant()

assistant.ingestion("../documents/Generative_Adversarial_Networks.pdf")

query, answer, contexts = assistant.retrieval("What are the two networks in a GAN, and what does each network do?")


# EVALUATE RAG APPLICATION USING RAGAS

# State Ground Truth or reference answer

reference_answer = """
A GAN consists of two networks: the generator and the discriminator.
The generator creates fake data intended to resemble real data, while
the discriminator distinguishes between real data and generated fake data.
"""


# Give RAGAS query , answer , contexts , reference answer
sample = SingleTurnSample(
    user_input=query,
    response= answer,
    retrieved_contexts=contexts,
    reference=reference_answer
)

# Initialize LLM for Evaluation
evaluator_model = LangchainLLMWrapper(assistant.model)
evaluator_embedding_model= LangchainEmbeddingsWrapper(assistant.embedding_model)


# Intialize Answer Similarity class
answer_similarity = AnswerSimilarity(embeddings=evaluator_embedding_model)


# Define Metrics
faithfulness_metric = Faithfulness(llm = evaluator_model)
answer_correctness_metric = AnswerCorrectness(llm=evaluator_model,answer_similarity=answer_similarity)
answer_relevancy_metric = AnswerRelevancy(llm = evaluator_model,embeddings=evaluator_embedding_model)
context_precision_metric = ContextPrecision(llm = evaluator_model)
context_recall_metric = ContextRecall(llm = evaluator_model)


#Generate Scores
faithfulness_score = faithfulness_metric.single_turn_score(sample)
answer_correctness_score = answer_correctness_metric.single_turn_score(sample)
answer_relevancy_score = answer_relevancy_metric.single_turn_score(sample)
context_precision_score = context_precision_metric .single_turn_score(sample)
context_recall_score = context_recall_metric.single_turn_score(sample)


# Scores
print("\n========== RAGAS EVALUATION ==========")

print("\nQuestion:")
print(query)

print("\nAnswer:")
print(answer)

print("\nScores:")
print("Faithfulness:", faithfulness_score)
print("Answer Correctness:", answer_correctness_score)
print("Answer Relevancy:", answer_relevancy_score)
print("Context Precision:", context_precision_score)
print("Context Recall:", context_recall_score)

print("\n======================================")