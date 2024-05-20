"""
File name: oracle_vector_db_lc.py
Author: Luigi Saetta
Date created: 2024-01-17
Date last modified: 2024-01-20
Python Version: 3.9

Description:
    This module provides the class to integrate Oracle
    DB Vector Store 
    in LangChain

Inspired by:
    

Usage:
    Import this module into other scripts to use its functions. 
    Example:
        from oracle_vector_db_lc import OracleVectorStore
        v_store = OracleVectorStore(embedding_function=embed_model.embed_query,
                            verbose=True)

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""
from __future__ import annotations
# allows using postponed evaluation of type annotations
# type annotations are a way to provide hints about the types of variables and function return values
# so an object type can be used i.e. during new class creation before being defined later in the module 
# introduced in python 3.7, default behaviour starting Python 3.10

import time
import array

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance

import oracledb
import logging

# load configs from here
from config_private import *

# But for now we don't need to compute the id.. it is set in the driving
# code when the doc list is created
from config_rag_oci import *

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


#
# supporting functions
#
def oracle_query(
    embed_query: List[float], top_k: int = 3, verbose=False
) -> List[Document]:
    """
    Executes a query against an Oracle database to find the top_k closest vectors to the given embedding.

    History:
        23/12/2023: modified to return some metadata (book_name, page_num)
    Args:
        embed_query (List[float]): A list of floats representing the query vector embedding.
        top_k (int, optional): The number of closest vectors to retrieve. Defaults to 2.
        verbose (bool, optional): If set to True, additional information about the query and execution time will be printed. Defaults to False.

    Returns:
        VectorStoreQueryResult: Object containing the query results, including nodes, similarities, and ids.
    """
    tStart = time.time()

    # build the DSN from data taken from config.py
    DSN = DB_HOST_IP + "/" + DB_SERVICE

    try:
        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                # 'f' single precision 'd' double precision
                if EMBEDDINGS_BITS == 64:
                    array_query = array.array("d", embed_query)
                else:
                    # 32 bits
                    array_query = array.array("f", embed_query)

                # changed select adding books (29/12/2023)
                select = f"""select V.id, C.CHUNK, C.PAGE_NUM, 
                            ROUND(VECTOR_DISTANCE(V.VEC, :1, DOT), 3) as d,
                            B.NAME 
                            from {VECTOR_FOLDER}_VECTORS V, {VECTOR_FOLDER}_CHUNKS C, {VECTOR_FOLDER}_BOOKS B
                            where C.ID = V.ID and
                            C.BOOK_ID = B.ID
                            order by d
                            FETCH FIRST {top_k} ROWS ONLY"""

                if verbose:
                    logging.info(f"select: {select}")

                cursor.execute(select, [array_query])

                rows = cursor.fetchall()

                result_docs = []
                node_ids = []
                similarities = []

                # prepare output
                for row in rows:
                    clob_pointer = row[1]
                    full_clob_data = clob_pointer.read()

                    # 29/12: added book_name to metadata
                    result_docs.append(
                        # pack in the expected format
                        Document(
                            page_content=full_clob_data,
                            metadata={"file_name": row[4], "page_label": row[2]},
                        )
                    )
                    # not used, for now
                    node_ids.append(row[0])
                    similarities.append(row[3])

    except Exception as e:
        logging.error(f"Error occurred in oracle_query: {e}")

        return None

    tEla = time.time() - tStart

    if verbose:
        logging.info(f"Query duration: {round(tEla, 1)} sec.")

    return result_docs


#
# OracleVectorStore
#
class OracleVectorStore(VectorStore): # inherits from langchain_core.vectorstores
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain" # class attribute 
    # Class attributes are shared among all instances of the class. They are accessed using the class name (OracleVectorStore) rather than an instance of the class.
    # Prefixing a variable or attribute name with an underscore (_) in Python is a convention that suggests that the variable or attribute is intended for internal use within the class or module
    # It serves as a signal to other developers that the variable or attribute is not part of the public API and should be treated as implementation details. 

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[Any] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.verbose = verbose

        self._embedding_function = embedding_function

        self.override_relevance_score_fn = relevance_score_fn

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        raise NotImplementedError("add_texts method must be implemented...")

    @property # used to create a read-only property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function
    # The use of a property here allows the user to access the _embedding_function attribute as if it were a property of the class. 
    # For example, if obj is an instance of the class, you can access the embeddings using obj.embeddings instead of obj.embeddings().


    #
    # similarity_search
    #
    def similarity_search(
        self, query: str, k: int = 3, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""

        if self.verbose:
            logging.info(f"top_k: {k}")
            logging.info("")

        # 1. embed the query
        embed_query = self._embedding_function(query)

        # 2. invoke oracle_query, return List[Document]
        result_docs = oracle_query(
            embed_query=embed_query, top_k=k, verbose=self.verbose
        )

        return result_docs
    
    # A class method is a method that is bound to the class and not the instance of the class. 
    # It takes the class itself as its first parameter (often named cls), and it can be called on the class rather than on an instance of the class.
    @classmethod
    def from_texts(
        cls: Type[OracleVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> OracleVectorStore:
        """Return VectorStore initialized from texts and embeddings."""
        raise NotImplementedError("from_texts method must be implemented...")