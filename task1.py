from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import numpy as np

# Create a FastAPI application
app = FastAPI()
pdfpath = "Annual-Report-2024-14-32.pdf"

load_dotenv()  # will search for .env file in local folder and load variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Cohere_API_token = os.getenv("COHERE_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def vectordb(pdfpath):
    """
    Loaded the pdf and make vector db stored in chromadb

    """

    loader = PyPDFLoader(pdfpath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_splits = text_splitter.split_documents(documents)
    chunks = text_splitter.split_documents(documents)
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory="./chroma_db"
    )
    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})

    return retriever_vectordb, text_splits


def hybridretrieval_reranking(retriever_vectordb, text_splits):
    """
    In this method, Cohererank and keyword retriever is used that will search on keyword basis and re rank the documents

    """
    # Hybrid Retrieval
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vectordb, keyword_retriever], weights=[0.5, 0.5]
    )

    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compression_retriever


def query_expansion(query):
    """
    Used OpenAI model for query expansion

    """

    query_expansion_template = PromptTemplate(
        input_variables=["query"],
        template="""You are a search query expansion expert. Your task is to see if there is any improvement required
            in the qiven query. You can make the query in a more understable way
            Include relevant synonyms and related terms to improve retrieval.
            Return only the expanded query without any explanations or additional text.
            If you feel that the query is perfect then you can return the query as it is

            Original query: {query}

            Expanded query:""",
    )
    prompt = query_expansion_template.format(query=query)
    # model for query expansion
    query_expansion_model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.8,  # Lower temperature for more focused query expansion
    )
    expanded_query = query_expansion_model.predict(prompt)
    return expanded_query


def calculate_weighted_confidence_score(llm_results_metadata):
    """
    Calculate the weighted confidence score from the given LLM results metadata containing log probabilities.

    """
    # Extract log probabilities
    log_probs = [
        item["logprob"] for item in llm_results_metadata["logprobs"]["content"]
    ]

    # Convert log probabilities to probabilities
    probabilities = np.exp(log_probs)

    # Get indices of top 5 probabilities
    sorted_indices = np.argsort(log_probs)[-5:]

    # Calculate joint probability for all tokens
    joint_probability_all = np.prod(probabilities)

    # Calculate joint probability for top 5 tokens
    top_probabilities = [probabilities[i] for i in sorted_indices]
    joint_probability_top_5 = np.prod(top_probabilities)

    # Weighted confidence score (70% from top 5, 30% from all tokens)
    confidence_score = round(
        (0.7 * joint_probability_top_5 + 0.3 * joint_probability_all) * 100, 2
    )

    return confidence_score


# process query method for user
def process_query(user_query):

    retriever_vectordb, text_splits = vectordb(pdfpath)
    compression_retriever = hybridretrieval_reranking(retriever_vectordb, text_splits)

    query = query_expansion(user_query)

    compressed_docs = compression_retriever.get_relevant_documents(query)

    response_generation_template = PromptTemplate(
        input_variables=["query", "context"],
        template="""You are a helpful AI assistant. Use the following context to answer the user's query.
            Be clear, concise, and accurate. I don't have answer to this question if the user's query is not in the context

            Context:
            {context}

            User Query: {query}

            Response:""",
    )

    prompt = response_generation_template.format(
        query=user_query, context=compressed_docs
    )
    response_model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,  # Higher temperature for more creative responses
        logprobs=True,
    )
    response = response_model.invoke(prompt)

    llm_results_metadata = response.response_metadata

    # finding confidence score based on meta data
    confidence_score = calculate_weighted_confidence_score(llm_results_metadata)

    return response.content, int(confidence_score), compressed_docs


# Define a route at the root web address ("/")
@app.get("/")
def read_root():

    query1 = "What is PSX?"
    output1, confidence_score1, compressed_docs1 = process_query(query1)

    query2 = "What is the rate of Gold?"
    output2, confidence_score2, compressed_docs2 = process_query(query2)

    query3 = "What are PSX developments?"
    output3, confidence_score3, compressed_docs3 = process_query(query3)

    query4 = "What are two exchange traded funds that PSX introduced?"
    output4, confidence_score4, compressed_docs4 = process_query(query4)

    query5 = "What are equity listing?"
    output5, confidence_score5, compressed_docs5 = process_query(query5)

    return {
        "User Query 1": query1,
        "Output ": output1,
        "confidence_score1": confidence_score1,
        "citation1": compressed_docs1,
        "User Query 2": query2,
        "Output 2": output2,
        "confidence_score2": confidence_score2,
        "citation2": compressed_docs2,
        "User Query 3": query3,
        "Output 3": output3,
        "confidence_score3": confidence_score3,
        "citation3": compressed_docs3,
        "User Query 4": query4,
        "Output 4": output4,
        "confidence_score4": confidence_score4,
        "citation4": compressed_docs4,
        "User Query 5": query5,
        "Output 5": output5,
        "confidence_score5": confidence_score5,
        "citation5": compressed_docs5,
    }
