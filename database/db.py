import sys
sys.path.append("..")
import os
import json
import chromadb as cdb
from chromadb.config import Settings
from typing import Union
import hashlib
import uuid
import pandas as pd
from database.embed import bgeEmbeddings

def generate_uuid(content: Union[str, bytes]) -> str:
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid

class ChromaVS:
    def generate_embedding(self, text: str) -> list[float]:
        return self.embed_model.embed_documents([text])[0]
    def __init__(self,config):
        self.embed_model = bgeEmbeddings("/home/tj/MBA_PAPER/One_Month_Paper/model/bge-large-en-v1.5")
        path = config.get("path",".")
        status_client = config.get("client","persistent")
        collection_metadata = config.get("collection_metadata", None)

        self.n_results_sql = config.get("n_results_sql", config.get("n_results", 3))
        self.n_results_documentation = config.get("n_results_documentation", config.get("n_results", 3))
        self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 3))
        if status_client == "persistent":
            self.client = cdb.PersistentClient(
                path=path,
                 settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = cdb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )

        self.documentation_collection = self.client.get_or_create_collection(
            name="documentation",

            metadata=collection_metadata,
        )
        self.ddl_collection = self.client.get_or_create_collection(
            name="ddl",
            metadata=collection_metadata,
        )
        self.sql_collection = self.client.get_or_create_collection(
            name="sql",
            metadata=collection_metadata,
        )
    
    

    def add_documentation(self, document: str, **kwargs) -> str:
        id = generate_uuid(document)
        self.documentation_collection.add(
            documents=document,
            embeddings=self.generate_embedding(document),
            ids=id,
        )
        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = generate_uuid(ddl)
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
        )
        return id
    
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = generate_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
        )

        return id
    
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get()

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get()

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df
    
    def remove_training_data(self, id: str,type: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id)
            return True
        elif type == "ddl":
            self.ddl_collection.delete(ids=id)
            return True
        elif type == "doc":
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False
        
    def remove_collection(self, collection_name: str) -> bool:
        if collection_name == "sql":
            self.client.delete_collection(name="sql")
            self.sql_collection = self.client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "ddl":
            self.client.delete_collection(name="ddl")
            self.ddl_collection = self.client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "documentation":
            self.client.delete_collection(name="documentation")
            self.documentation_collection = self.client.get_or_create_collection(
                name="documentation", embedding_function=self.embedding_function
            )
            return True
        else:
            return False
    
    @staticmethod
    def _extract_documents(query_results) -> list:
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents
        
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        return ChromaVS._extract_documents(
            self.sql_collection.query(
                # query_texts=[question],
                query_embeddings=[self.generate_embedding(question)],
                n_results=self.n_results_sql,
            )
        )

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return ChromaVS._extract_documents(
            self.ddl_collection.query(
                # query_texts=[question],
                query_embeddings=[self.generate_embedding(question)],
                n_results=self.n_results_ddl,
            )
        )

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return ChromaVS._extract_documents(
            self.documentation_collection.query(
                # query_texts=[question],
                query_embeddings=[self.generate_embedding(question)],
                n_results=self.n_results_documentation,
            )
        )
    
config = {
    "client": "persistent",
    "path": "/media/tj/zhijia-main/Langchain_shen/py_data_cdb",
}
chroma_vs = ChromaVS(config)
if __name__ == "__main__":
    config = {
        "client": "persistent",
        "path": "/media/tj/zhijia-main/Langchain_shen/py_data_cdb",
    }
    chroma_vs = ChromaVS(config)
    print(len(chroma_vs.generate_embedding("hello,world!")))