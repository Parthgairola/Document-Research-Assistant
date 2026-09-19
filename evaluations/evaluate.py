from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness , AnswerCorrectness , AnswerSimilarity, ContextPrecision , ContextRecall

# Import DocumentResearchAssistant class for using its functions
from src.main import DocumentResearchAssistant
assistant = DocumentResearchAssistant()

# PDF Ingestion
assistant.ingestion("documents/Generative_Adversarial_Networks.pdf")

# Answer retrieval 
query, answer, contexts = assistant.retrieval(
    "What are the two networks in a GAN, and what does each network do?"
)

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

context_precision_metric = ContextPrecision(llm = evaluator_model)
context_recall_metric = ContextRecall(llm = evaluator_model)


#Generate Scores
faithfulness_score = faithfulness_metric.single_turn_score(sample)
answer_correctness_score = answer_correctness_metric.single_turn_score(sample)

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

print("Context Precision:", context_precision_score)
print("Context Recall:", context_recall_score)

print("\n======================================")