from langchain_community.vectorstores import Chroma
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_vectorstore():
    return Chroma(
        collection_name="split_parents",
        embedding_function=FastEmbedEmbeddings(),
        persist_directory="./split_parent_docs",
    )


def get_docstore():
    return create_kv_docstore(LocalFileStore("./parent_docs"))


def get_parent_document_retriever():
    return ParentDocumentRetriever(
        vectorstore=get_vectorstore(),
        docstore=get_docstore(),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
    )
