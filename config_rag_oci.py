# configurations for the RAG

# to enable debugging info..
DEBUG = False

# 
# To create vector cache and persistant vector db in unique folder as per a use case or customer
# 
# VECTOR_FOLDER = "yesbank"
# VECTOR_FOLDER = "cdc"
# VECTOR_FOLDER = "coe_internal"
# VECTOR_FOLDER = "coe_demo"
# VECTOR_FOLDER = "hindalco"
VECTOR_FOLDER = "ncert"


# to divide docs in chunks
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50


#
# Vector Store (Chrome or FAISS)
#
# VECTOR_STORE_NAME = "FAISS"
VECTOR_STORE_NAME = "CHROMA"
# VECTOR_STORE_NAME = "ORCL_VECTORDB"

# type of Embedding Model. The choice has been parametrized
# Local means HF
# EMBED_TYPE = "LOCAL"
# see: https://huggingface.co/spaces/mteb/leaderboard
# see also: https://github.com/FlagOpen/FlagEmbedding
# base seems to work better than small
# EMBED_HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# EMBED_HF_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_HF_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Cohere means the embed model from Cohere site API
# EMBED_TYPE = "COHERE"
EMBED_COHERE_MODEL_NAME = "embed-english-v3.0"

EMBED_TYPE = "OCI_COHERE_EMBED"
# cohere.embed-english-light-v2.0, cohere.embed-english-light-v3.0, cohere.embed-english-v3.0
EMBED_OCI_COHERE_MODEL_NAME = "cohere.embed-english-v3.0"

# number of docs to return from Retriever
top_k = 3

# to add Cohere reranker to the QA chain
ADD_RERANKER = False

#
# LLM Config
#
# LLM_TYPE = "COHERE"
# LLM_TYPE = "OCI"
# LLM_TYPE = "OCI_DS_AQUA"
LLM_TYPE = "OCI_AmpereA1"

# 
# type of LLM Model. The choice has been parametrized
# cohere.command, cohere.command-light, meta.llama-2-70b-chat
LLM_NAME = "cohere.command"

# max tokens returned from LLM for single query
MAX_TOKENS = 1500
# to avoid "creativity"
TEMPERATURE = 0

#
# OCI GenAI configs
#
TIMEOUT = 30

# bits used to store embeddings
# possible values: 32 or 64
# must be aligned with the create_tables.sql used
EMBEDDINGS_BITS = 64

# ID generation: LLINDEX, HASH, BOOK_PAGE_NUM
# define the method to generate ID
ID_GEN_METHOD = "HASH"