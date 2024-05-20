#
# This one is to be used with Streamlit
#

import streamlit as st

# for pdf post processing
import re

# modified to load from Pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
import shutil
import ast

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# removed OpenAI, using Cohere embeddings
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
# from genai_langchain_integration.langchain_oci_embeddings import OCIGenAIEmbeddings
from langchain_community.embeddings import OCIGenAIEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import sys

# config for the RAG
from config_rag_oci import *
from config_private import *

CONFIG_PROFILE = "DEFAULT"

# 
# Capturing names of pdf files to be used
# 

# book_id = []
book_name = []

# 
# Capturing pdf files name from a particular directory
# 
for filename in os.listdir(f"./documents/{VECTOR_FOLDER}/"):
     if filename.lower().endswith((".pdf", ".txt")):
        book_name.append(filename)

print(f'Docs being processed:')
for i, book in enumerate(book_name):
    print(f'({i+1}) {book}')


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
# Add extra metadata
# 
def add_metadata(doc):
    # Get the directory separator for the platform
    # separator = os.path.sep
    separator = '/'
    # print(separator)
    start = doc.metadata["source"].rfind(separator)
    # print(start)
    # end = doc.metadata["source"].find('.pdf')
    # print(end)
    # print(f'Doc name: {doc.metadata["source"][start+1:end]}')
    doc.metadata["name"] = doc.metadata["source"][start+1:]
    key = "page"
    if key not in doc.metadata:
        # print(f"{key} does not exist in the document metadata")
        doc.metadata[key] = 0
    # print(doc.metadata)

#
# load all documents inside a particular directory
#
#
def load_all_docs():
    all_docs = []
    docs_dir = f"./documents/{VECTOR_FOLDER}/"
    for filename in os.listdir(docs_dir):
        # Loading pdf docs
        if filename.lower().endswith(".pdf"):
            print(f"Loading document: {filename}...")
            loader = PyPDFLoader(f'{docs_dir}{filename}')
        # load txt files
        elif filename.lower().endswith(".txt"):
            print(f"Loading file: {filename}...")
            loader = TextLoader(f'{docs_dir}{filename}')
        else:
            print(f'No loader defined for this file: {filename}')
            continue

        # loader split in pages
        pages = loader.load()
        
        # looping to add extra metadata to each page
        for page in pages:
            if DEBUG:
                print("===========================")
                print(type(page))
                print(type(page.page_content))
                print(page.page_content)
                print(type(page.metadata))
                print(page.metadata)
                print("===========================")
            # Add extra metadata to each page
            add_metadata(page)

        all_docs.extend(pages)

    print(f"\nLoaded {len(all_docs)} docs...\n")
    print(f'##### All docs ####:\n')
    # [print(f'{doc}\n') for doc in all_docs]
    return all_docs


#
# do some post processing on text
#
def post_process(splits):
    for split in splits:
        # replace newline with blank
        split.page_content = split.page_content.replace("\n", " ")
        split.page_content = re.sub("[^a-zA-Z0-9 \n\.]", " ", split.page_content)
        split.page_content = re.sub(r"\.{2,}", ".", split.page_content)
        # remove duplicate blank
        split.page_content = " ".join(split.page_content.split())

    return splits


#
# Split pages in chunk
#
def split_in_chunks(all_pages):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["(?<=\. )", "\n"],
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len
    )

    splits = text_splitter.split_documents(all_pages)
    print(f"Splitted the document in {len(splits)} chunks...")
    
    # some post processing on text
    splits = post_process(splits)
    if DEBUG:
        print('==========================================')
        for item in splits:
            print(f'\n {item}')
        print('==========================================')

    non_empty_splits = []
    for i, item in enumerate(splits):
        if dict(item)['page_content'] != '':
            non_empty_splits.append(item)
        
    print(f"Number of non-empty chunks: {len(non_empty_splits)}")

    return non_empty_splits


#
# Load the embedding model
#
def create_cached_embedder():
    print("Initializing Embeddings model...")

    # Introduced to cache embeddings and make it faster
    vector_folder_path = f"./vector-cache/{VECTOR_FOLDER}/" 
    if os.path.isdir(vector_folder_path):
        shutil.rmtree(vector_folder_path) 
        print(f'Directory called {VECTOR_FOLDER} has been deleted from vector-cache folder')
    else:
        print(f'A new directory called {VECTOR_FOLDER} will be created under vector-cache folder')

    fs = LocalFileStore(vector_folder_path)

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
            compartment_id=COMPARTMENT_ID, 
            truncate = 'NONE'
        )

    # the cache for embeddings
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        # embed_model, fs, namespace=embed_model.model_name # HUGGING FACE
        # embed_model, fs, namespace=embed_model.model # COHERE
        embed_model, fs, namespace=embed_model.model_id # OCI GEN AI COHERE
    )

    return cached_embedder


# 
# required for oracle vectordb
# 
def initialize_vectordb_tables(cursor):
    # Drop tables
    table = f'{VECTOR_FOLDER}_CHUNKS'
    # print(f'table: {table}')
    query = f"""
    begin
        execute immediate 'drop table {table}';
        exception when others then if sqlcode <> -942 then raise; end if;
    end;"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    table = f'{VECTOR_FOLDER}_BOOKS'
    # print(f'table: {table}')
    query = f"""
    begin
        execute immediate 'drop table {table}';
        exception when others then if sqlcode <> -942 then raise; end if;
    end;"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    table = f'{VECTOR_FOLDER}_VECTORS'
    # print(f'table: {table}')
    query = f"""
    begin
        execute immediate 'drop table {table}';
        exception when others then if sqlcode <> -942 then raise; end if;
    end;"""
    # print(f'query: {query}')
    cursor.execute(query)

    print(f"\nAll {VECTOR_FOLDER} tables in DATABASE SCHEMA {DB_USER}: ")
    query = f"""SELECT table_name FROM all_tables WHERE owner = '{DB_USER}' and table_name like '{VECTOR_FOLDER}%'"""
    # print(f'all tables: {query}')
    cursor.execute(query)
    
    for row in cursor:
         print(row)
    
    # create tables
    query = f"""
    create table {VECTOR_FOLDER}_VECTORS (
        id VARCHAR2(64) NOT NULL,
        VEC VECTOR(1024, FLOAT64),
        primary key (id))"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    query = f"""
    create table {VECTOR_FOLDER}_BOOKS (
        ID NUMBER NOT NULL,
        NAME VARCHAR2(100) NOT NULL,
        PRIMARY KEY (ID)  )"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    query = f"""
    create table {VECTOR_FOLDER}_CHUNKS 
        (ID VARCHAR2(64) NOT NULL,
        CHUNK CLOB,
        PAGE_NUM VARCHAR2(10),
        BOOK_ID NUMBER,
        PRIMARY KEY ("ID"),
        CONSTRAINT fk_{VECTOR_FOLDER}_book
                FOREIGN KEY (BOOK_ID)
                REFERENCES {VECTOR_FOLDER}_BOOKS (ID)
        )"""
    # print(f'query: {query}')
    cursor.execute(query)
    
    print("vectordb tables initialized...")


# 
# required for oracle vectordb
# with this function every book added to DB is registered with a unique id
# 
def register_book(book_name, connection):
       
    with connection.cursor() as cursor:
                
        # get the new key
        query = f"SELECT MAX(ID) FROM {VECTOR_FOLDER}_BOOKS"
        cursor.execute(query)

        # Fetch the result
        row = cursor.fetchone()

        if row[0] is not None:
            new_key = row[0] + 1
        else:
            new_key = 1

        # insert the record for the book
        query = f"INSERT INTO {VECTOR_FOLDER}_BOOKS (ID, NAME) VALUES (:1, :2)"

        # Execute the query with your values
        cursor.execute(query, [new_key, book_name])

    return new_key


# 
# required for oracle vectordb
# 
def save_chunks_in_db(document_splits, book_id, book_name, connection):
    tot_errors = 0
    
    chunk_id = [] 
    chunk_text = [] 
    
    document_splits_str = [str(item) for item in document_splits]
    
    with connection.cursor() as cursor:
        print("Saving texts to DB...")
        cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

        for i, chunk in enumerate(document_splits_str):
            # chunk_id = i+1
            chunk_id.append(i+1)

            chunk_text_start = chunk.find("page_content=")
            chunk_metadata_start = chunk.find("metadata=")

            chunk_content = chunk[chunk_text_start+13:chunk_metadata_start]
            chunk_text.append(chunk_content)
            
            chunk_metadata=chunk[chunk_metadata_start+9:]
            chunk_metadata = ast.literal_eval(chunk_metadata)

            chunk_book_id = book_id[book_name.index(chunk_metadata["name"])]
            chunk_page_num = int(chunk_metadata["page"])+1
            
            try:
                query = f"insert into {VECTOR_FOLDER}_CHUNKS (ID, CHUNK, PAGE_NUM, BOOK_ID) values (:1, :2, :3, :4)"
                cursor.execute(query, [i+1, chunk_content, chunk_page_num, chunk_book_id])
            except Exception as e:
                print("Error in save chunks...")
                print(e)
                tot_errors += 1
            
        print(f'No. of chunk ids created inside get_chunk_content(): {len(chunk_id)}')
        return chunk_id, chunk_text


# 
# required for oracle vectordb
# 
def save_embeddings_in_db(embeddings, chunks_id, connection):
    tot_errors = 0

    with connection.cursor() as cursor:
        print("Saving embeddings to DB...")

        for id, vector in zip(chunks_id, embeddings):        
            # 'f' single precision 'd' double precision
            if EMBEDDINGS_BITS == 64:
                input_array = array.array("d", vector)
            else:
                # 32 bits
                input_array = array.array("f", vector)

            try:
                # insert single embedding
                query = f"insert into {VECTOR_FOLDER}_VECTORS values (:1, :2)"
                cursor.execute(query, [id, input_array])
            except Exception as e:
                print("Error in save embeddings...")
                print(e)
                tot_errors += 1

    print(f"Tot. errors in save_embeddings: {tot_errors}")


#
# create vector store
#
def create_vector_store(store_type, document_splits, embedder):
    print(f"Indexing: using {store_type} as Vector Store...")

    if store_type == "CHROMA":
        # modified to cache

        # in-memory chromadb
        # vectorstore = Chroma.from_documents(
        #     documents=document_splits, embedding=embedder
        # )

        # persistant chromadb
        # vectorstore = Chroma.from_documents(
        #     documents=document_splits, 
        #     embedding=embedder, 
        #     persist_directory=f"./chroma_db/{VECTOR_FOLDER}" 
        # )

        # OCI GenAI Cohere Embedding supports a size of [1:96] as input array
        # Below code takes into consideration input array size limit
        vectordb_path = f"./chroma_db/{VECTOR_FOLDER}/" 
        if os.path.isdir(vectordb_path):
            shutil.rmtree(vectordb_path) 
            print(f'Directory called {VECTOR_FOLDER} has been deleted from chroma-db folder')
        else:
            print(f'A new directory called {VECTOR_FOLDER} will be created under chroma-db folder')
        
        vectorstore = Chroma(
                        persist_directory=vectordb_path,
                        embedding_function=embedder)
        
        ids_list = [str(pos+1) for pos, s in enumerate(document_splits)]
        print(f'No. of document splits: {len(ids_list)}')
        
        input_array_size=96
        start=0
        while start < len(document_splits):
            try:
                vectorstore.add_documents(
                ids = ids_list[start:start+input_array_size],
                documents = document_splits[start:start+input_array_size]
                )
                start+=input_array_size
            except Exception as error:
                print(f'\nERROR OCCURRED WHILE CREATING VECTOR DB:\n {error} ')
                print(f'\nStart = {start}, End = {start+input_array_size}')
                for i, item in enumerate(document_splits[start:start+input_array_size]):
                    print(f'ID# : {ids_list[start+i-1]}')
                    print(f'Document: {item}')
                # break
                sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred

    elif store_type == "FAISS":
        # modified to cache
        vectorstore = FAISS.from_documents(
            documents=document_splits, embedding=embedder
        )
    
    elif store_type == "ORCL_VECTORDB":
        # connect to db
        print("Connecting to Oracle DB...")

        DSN = DB_HOST_IP + "/" + DB_SERVICE

        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            print("Successfully connected to Oracle Database...")
            
            initialize_vectordb_tables(connection.cursor())
            
            # determine book_id and save in table BOOKS
            print("Registering docs to vectordb table BOOKS...")
            book_id = [register_book(book, connection) for book in book_name]
            # book_id = register_book(book_name, connection)
            print(f"Completed registering docs inside oracle vectordb")

            chunk_id, chunk_text = save_chunks_in_db(document_splits, book_id, book_name, connection)
            print(f"Completed savings chunks inside oracle vectordb")

            input_array_size=96
            start=0
            embeddings = []
            while start < len(document_splits):
                try:
                    chunk_embeddings = embedder.embed_documents(chunk_text[start:start+input_array_size])
                    embeddings.extend(chunk_embeddings)
                    start+=input_array_size
                except Exception as error:
                    print(f'\nERROR OCCURRED WHILE CREATING VECTOR DB:\n {error} ')
                    print(f'\nStart = {start}, End = {start+input_array_size}')
                    for i, item in enumerate(document_splits[start:start+input_array_size]):
                        print(f'ID# : {ids_list[start+i-1]}')
                        print(f'Document: {item}')
                    # break
                    sys.exit(1) # to exit the program with a non-zero exit code, indicating that an error occurred
            print(f'No. of embeddings: {len(embeddings)}')

            # store embeddings
            # here we save in DB
            save_embeddings_in_db(embeddings, chunk_id, connection)
            print(f"Completed savings embeddings inside oracle vectordb")

            # a txn is a book
            connection.commit()

    # return not required in case of persistant chromadb or oracle vectordb
    # return vectorstore


#
# create vector db
#
# to run it only once
# @st.cache_resource
def set_up_vectordb():

    # 1. Load a list of documents
    all_pages = load_all_docs()
    # all_pages
    
    # 2. Split pages in chunks
    document_splits = split_in_chunks(all_pages)
    
    # 3. Load embeddings model
    embedder = create_cached_embedder()
    
    # 4. Create a Vectore Store and store embeddings
    # for in-memory chromadb
    # vectorstore = create_vector_store(VECTOR_STORE_NAME, document_splits, embedder)
    
    # for persistant chromadb & OCI vectordb
    create_vector_store(VECTOR_STORE_NAME, document_splits, embedder)

set_up_vectordb()