#
# This one is to be used with Streamlit
#

import streamlit as st

# for caching
from langchain.storage import LocalFileStore

# possible vector stores
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

# Using OCI vectordb vector store
import oracledb
# Luigi's OracleVectorStore wrapper for LangChain
from oracle_vector_db_lc import OracleVectorStore

from tqdm import tqdm
import array
import os
import ast

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


from langchain_community.embeddings import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain_community.llms import Cohere

import oci

# from genai_langchain_integration.langchain_oci import OCIGenAI
from langchain_community.llms import OCIGenAI
from aqua_cllm import * # Custom LLM langchain class created by RG
from ampA1_cllm import * # Custom LLM langchain class created by RG
# from genai_langchain_integration.langchain_oci_embeddings import OCIGenAIEmbeddings
from langchain_community.embeddings import OCIGenAIEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

my_conv_memory = ConversationBufferMemory()
# my_conv_memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     input_key='question', output_key='answer',
#     return_messages=True,
#     verbose = True
# )

# config for the RAG
from config_rag_oci import *
from config_private import *

# from rag_vectordb import create_cached_embedder

CONFIG_PROFILE = "DEFAULT"


#
# load_oci_config(): load the OCI security config
#
def load_oci_config():
    # read OCI config to connect to OCI with API key
    oci_config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # check the config to access to api keys
    if DEBUG:
        print()
        print("OCI Config:")
        print(oci_config)
        print()

    return oci_config

#
# Load the embedding model
#
def create_cached_embedder():
    print("Initializing Embeddings model...")

    # Introduced to cache embeddings and make it faster
    fs = LocalFileStore(f"./vector-cache/{VECTOR_FOLDER}/")

    if EMBED_TYPE == "COHERE":
        print("Loading Cohere Embeddings Model...")
        embed_model = CohereEmbeddings(
            model=EMBED_COHERE_MODEL_NAME, cohere_api_key=COHERE_API_KEY
        )
    elif EMBED_TYPE == "LOCAL":
        print(f"Loading HF Embeddings Model: {EMBED_HF_MODEL_NAME}")

        model_kwargs = {"device": "cpu"}
        # changed to True for BAAI, to use cosine similarity
        encode_kwargs = {"normalize_embeddings": True}

        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_HF_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif EMBED_TYPE == "OCI_COHERE_EMBED":
        print(f"Loading OCI GenAI Embeddings Model: {EMBED_OCI_COHERE_MODEL_NAME}")

        embed_model = OCIGenAIEmbeddings(
            model_id=EMBED_OCI_COHERE_MODEL_NAME, 
            service_endpoint=OCI_GENAI_ENDPOINT,
            compartment_id=COMPARTMENT_ID
        )

    # the cache for embeddings
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        # embed_model, fs, namespace=embed_model.model_name # HUGGING FACE
        # embed_model, fs, namespace=embed_model.model # COHERE
        embed_model, fs, namespace=embed_model.model_id # OCI GEN AI COHERE
    )

    return cached_embedder


#
# create retrievere with optional reranker
#
def create_retriever(vectorstore):
    if ADD_RERANKER == False:
        # no reranking
        print("No reranking...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    else:
        # to add reranking
        print("Adding reranking to QA chain...")

        compressor = CohereRerank(cohere_api_key=COHERE_API_KEY)

        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    return retriever

#
# Build LLM
#
def build_llm(llm_type):
    print(f"Using {llm_type} llm...")

    if llm_type == "OCI":

        print("Using OCI experimental integration with LangChain...")
        
        llm = OCIGenAI(
            auth_profile="DEFAULT",
            model_id=LLM_NAME,
            compartment_id=COMPARTMENT_ID,
            service_endpoint=OCI_GENAI_ENDPOINT,
            #temperature=TEMPERATURE # old endpoint
            model_kwargs={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}#, "stop": ["populous"]} # new endpoint
        )

    elif llm_type == "COHERE":
        llm = Cohere(
            model="command",  # using large model and not nightly
            cohere_api_key=COHERE_API_KEY,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

    elif llm_type == "OCI_DS_AQUA":
        config = oci.config.from_file("~/.oci/config") # replace with the location of your oci config file
        llm = DS_AQUA_LLM(
                temperature=TEMPERATURE, 
                config=config, 
                # compartment_id=compartment_id, 
                service_endpoint=OCI_AQUA_ENDPOINT,        
                max_tokens = MAX_TOKENS, 
                top_k = top_k, 
                top_p = 1
                )
        print(f'Config used for AQUA LLM:\n{llm}')

    elif llm_type == "OCI_AmpereA1":
         llm = OCI_AmpereA1_LLM(
                temperature=TEMPERATURE, 
                # compartment_id=compartment_id, 
                # service_endpoint=OCI_AQUA_ENDPOINT,        
                max_tokens = MAX_TOKENS, 
                top_k = top_k, 
                top_p = 1
                )

    return llm


#
# Initialize_rag_chain
#
# to run it only once
@st.cache_resource
def initialize_rag_chain():
    # Initialize RAG
    
    # 1. Load embeddings model
    embedder = create_cached_embedder()
    
    # 2. restore vectordb
    if VECTOR_STORE_NAME == "CHROMA":
        # restore persistant chromadb
        vectorstore = Chroma(persist_directory=f"./chroma_db/{VECTOR_FOLDER}", embedding_function=embedder)
    elif VECTOR_STORE_NAME == "ORCL_VECTORDB":
        # restore oracle vectordb
        # OracleVectorStore is custom class that inherits from langchain_core.vectorstores
        vectorstore = OracleVectorStore(embedding_function=embedder.embed_query, verbose=True)
    else:
        print(f'Please check the vector store to be used as part of configuration')
        exit()

    # 3. Create a retriever
    # added optionally a reranker
    retriever = create_retriever(vectorstore)
    print(f'Retriever object: {retriever}')
    
    # 4. Build the LLM
    llm = build_llm(LLM_TYPE)
    
    # 5. define the prompt (for now hard coded...)
    # rag_prompt = hub.pull("rlm/rag-prompt")
    
    # template = """Answer the question based only on the following context:
    # {context}
    # Use three sentences maximum and keep the answer as concise as possible.
    # Do not end your answer with a question.
    # Question: {question}
    # """
    
    # template = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the context below, say "I don't know". Do not end your answer with a question.
    # Context: {context}
    # Question: {question}
    # """

    template = """Answer the question by using only the provided context below. If the answer is not contained within the context below, say "I don't know". Only give answer. Do not end the answer by asking another question.
    Context: {context}
    Question: {question}
    """
    
    rag_prompt = ChatPromptTemplate.from_template(template)
    # print output here
    
    # 6. build the entire RAG chain
    print("Building rag_chain...")
    global my_conv_memory
    my_conv_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question', output_key='answer',
        return_messages=True,
        verbose = True
    )

    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        memory=my_conv_memory,
        return_source_documents=True,
        return_generated_question=True,
        rephrase_question=False, 
        combine_docs_chain_kwargs={'prompt': rag_prompt}
    )
    
    print("\nInit RAG complete.")
    
    return qa

# 
# to format output response
# 

def print_response(my_dict):
    desired_order = ['question', 'generated_question', 'answer', 'chat_history', 'source_documents']
    for key in desired_order:
        print(f"\n\n{key}:\n")
        # print(f"    {type(my_dict[key])}")
        if isinstance(my_dict[key], list): 
            for item in my_dict[key]:
                print(f'{item}\n')
        else:
            print(f"    {my_dict[key]}")

#
# def: get_answer  from LLM
#
def get_answer(rag_chain, question):
    response = rag_chain.invoke(question)

    if DEBUG:
        print(f"Question: {question}")
        print("The response:")
        print(response)
        print()
    
    print_response(response)

    return response
    # return response["answer"]

# 
# Reset Conversation Buffer Memory
# 
def clear_conv_memory():
    global my_conv_memory
    my_conv_memory.clear()
