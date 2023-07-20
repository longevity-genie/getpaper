#!/usr/bin/env python3
import os
from enum import Enum
from pathlib import Path

import click
from click import Context
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, VertexAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma, VectorStore, Qdrant
from loguru import logger
from pycomfort.files import *
from qdrant_client import QdrantClient

from getpaper.config import load_environment_keys, LOG_LEVELS, LogLevel, configure_logger
from getpaper.splitting import OpenAISplitter, SourceTextSplitter, papers_to_documents

class VectorDatabase(Enum):
    Chroma = "Chroma"
    Qdrant = "Qdrant"


VECTOR_DATABASES: list[str] = [db.value for db in VectorDatabase]


def resolve_embeddings(embeddings_name: str, model_path: Optional[Union[Path, str]] = None) -> Embeddings:
    if embeddings_name == "openai":
        return OpenAIEmbeddings()
    elif embeddings_name == "llama":
        if model_path is None:
            logger.error(f"for llama embeddings for {model_path} model")
        return LlamaCppEmbeddings(model_path = str(model_path))
    elif embeddings_name == "vertexai":
        return VertexAIEmbeddings()
    else:
        logger.warning(f"{embeddings_name} is not yet supported by CLI, using default openai embeddings instead")
        return OpenAIEmbeddings()


def db_with_documents(db: VectorStore, documents: list[Document],
                      splitter: TextSplitter,
                      id_field: Optional[str] = None):
    """
    Function to add documents to a Chroma database.

    Args:
        db (Chroma): The database to add the documents to.
        documents (list[Document]): The list of documents to add.
        splitter (TextSplitter): The TextSplitter to use for splitting the documents.
        debug (bool): If True, print debug information. Default is False.
        id_field (Optional[str]): If provided, use this field from the document metadata as the ID. Default is None.

    Returns:
        Chroma: The updated database.
    """
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata[id_field] for doc in docs] if id_field is not None else None
    for doc in documents:
        logger.trace(f"ADD TEXT: {doc.page_content}")
        logger.trace(f"ADD METADATA {doc.metadata}")
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return db


def db_with_documents(db: VectorStore, documents: list[Document],
                      splitter: TextSplitter,
                      id_field: Optional[str] = None):
    """
    Function to add documents to a Chroma database.

    Args:
        db (Chroma or Qdrant): The database to add the documents to.
        documents (list[Document]): The list of documents to add.
        splitter (TextSplitter): The TextSplitter to use for splitting the documents.
        debug (bool): If True, print debug information. Default is False.
        id_field (Optional[str]): If provided, use this field from the document metadata as the ID. Default is None.

    Returns:
        Chroma: The updated database.
    """
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata[id_field] for doc in docs] if id_field is not None else None
    for doc in documents:
        logger.trace(f"ADD TEXT: {doc.page_content}")
        logger.trace(f"ADD METADATA {doc.metadata}")
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return db


def init_qdrant(collection_name: str, path_or_url: str,  embedding_function: Optional[Embeddings], api_key: Optional[str] = None, distance_func: str = "Cosine", prefer_grpc: bool = False):
    is_url = "ttp:" in path_or_url or "ttps:" in path_or_url
    path: Optional[str] = None if is_url else path_or_url
    url: Optional[str] = path_or_url if is_url else None
    logger.info(f"initializing quadrant database at {path_or_url}")
    client: QdrantClient = QdrantClient(
        url=url,
        port=6333,
        grpc_port=6334,
        prefer_grpc=is_url if prefer_grpc is None else prefer_grpc,
        api_key=api_key,
        path=path
    )
    from qdrant_client.http import models as rest
    #client.recreate_collection(collection_name)
    # Just do a single quick embedding to get vector size
    partial_embeddings = embedding_function.embed_documents("probe")
    vector_size = len(partial_embeddings[0])
    distance_func = distance_func.upper()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance[distance_func],
        )
    )
    return Qdrant(client, collection_name=collection_name, embeddings=embedding_function)


def write_remote_db(url: str,
                    collection_name: str,
                    documents: list[Document],
                    splitter: TextSplitter,
                    id_field: Optional[str] = None,
                    embeddings: Optional[Embeddings] = None,
                    database: VectorDatabase = VectorDatabase.Qdrant,
                    key: Optional[str] = None, prefer_grpc: bool = False):
    if database == VectorDatabase.Qdrant:
        logger.info(f"writing a collection {collection_name} of {len(documents)} documents to quadrant db at {url}")
        db = init_qdrant(collection_name, path_or_url=url, embedding_function=embeddings, api_key=key, prefer_grpc=prefer_grpc)
        db_updated = db_with_documents(db, documents, splitter,  id_field)
        return db_updated
    else:
        raise Exception(f"Remote Chroma is not yet supported by this script!")
    pass


def write_local_db(persist_directory: Path,
                   collection_name: str,
                   documents: list[Document],
                   splitter: TextSplitter,
                   id_field: Optional[str] = None,
                   embeddings: Optional[Embeddings] = None,
                   database: VectorDatabase = VectorDatabase.Chroma,
                   prefer_grpc: bool = False
                   ):
    """
    Writes the provided documents to a database.

    Args:
        persist_directory (Path): The directory where the database should be saved.
        collection_name (str): The name of the collection in the database.
        documents (list[Document]): The list of documents to be added to the database.
        splitter: TextSplitter
        chunk_size (int): The size of the text chunks to split the documents into. Default is 6000.
        debug (bool): If True, print debug information. Default is False.
        id_field (Optional[str]): The name of the field to use as the document ID. Default is None.
        embeddings (Optional[Embeddings]): The embeddings to use. If not provided, defaults to OpenAIEmbeddings.

    Returns:
        Path: The directory where the database was saved.
    """

    # Create the directory where the database will be saved, if it doesn't already exist
    where = persist_directory / collection_name
    where.mkdir(exist_ok=True, parents=True)

    # If no embeddings were provided, default to OpenAIEmbeddings
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    # Create a Chroma database with the specified collection name and embeddings, and save it in the specified directory
    if database == VectorDatabase.Qdrant:
        db = init_qdrant(collection_name, str(where), embedding_function=embeddings,  prefer_grpc = prefer_grpc)
    else:
        db = Chroma(collection_name=collection_name, persist_directory=str(where), embedding_function=embeddings)
    #db = init_db(database, collection_name=collection_name, path_or_url=str(where), embedding_function=embeddings)
    # Add the documents to the database
    db_updated = db_with_documents(db, documents, splitter,  id_field)

    # Persist the changes to the database
    db_updated.persist()

    # Return the directory where the database was saved
    return where


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


def index_selected_papers(papers_folder: Path,
                          collection: str,
                          splitter: SourceTextSplitter,
                          embeddings: str,
                          include_meta: bool,
                          folder: Optional[Union[Path,str]] = None,
                          url: Optional[str] = None,
                          key: Optional[str] = None,
                          database: VectorDatabase = VectorDatabase.Chroma.value,
                          model: Optional[Union[Path, str]] = None,
                          prefer_grpc: Optional[bool] = None
                          ):
    openai_key = load_environment_keys() #for openai key
    embeddings_function = resolve_embeddings(embeddings, model)
    logger.info(f"embeddings are {embeddings}")
    documents = papers_to_documents(papers_folder, include_meta=include_meta)
    if folder is not None:
        index = Path(folder) if isinstance(folder, str) else folder
        index.mkdir(exist_ok=True)
        where = index / f"{embeddings}_{splitter.chunk_size}_chunk"
        where.mkdir(exist_ok=True, parents=True)
        logger.info(f"writing index of papers to {where}")
        return write_local_db(where, collection, documents, splitter, embeddings=embeddings_function, prefer_grpc = prefer_grpc, database=database)
    elif url is not None:
        return write_remote_db(url, collection, documents, splitter, embeddings=embeddings_function, database=database, key=key, prefer_grpc = prefer_grpc)
    else:
        raise Exception("neither folder nor url are set")
        pass


@app.command("index_papers")
@click.option('--papers', type=click.Path(exists=True), help="papers folder to index")
@click.option('--collection', default='papers', help='papers collection name')
@click.option('--folder', type=click.Path(), default=None, help="folder to put chroma indexes to")
@click.option('--url', type=click.STRING, default=None, help="alternatively you can provide url, for example http://localhost:6333 for qdrant")
@click.option('--key', type=click.STRING, default=None, help="your api key if you are using cloud vector store")
@click.option('--splitter_name', type=click.Choice(["openai", "recursive"]), default="openai", help='which splitter to choose for the text splitting')
@click.option('--chunk_size', type=click.INT, default=3000, help='size of the chunk for splitting (characters for recursive spliiter and tokens for openai one)')
@click.option('--embeddings', type=click.Choice(["openai", "llama", "vertexai"]), default="openai",
              help='size of the chunk for splitting')
@click.option("--model", type=click.Path(), default=None, help="path to the model (required for embeddings)")
@click.option('--include_meta', type=click.BOOL, default=True, help="if metadata is included")
@click.option('--database', type=click.Choice(VECTOR_DATABASES, case_sensitive=False), default=VectorDatabase.Chroma.value, help = "which store to take")
@click.option('--prefer_grpc', type=click.BOOL, default = None)
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_papers_command(papers: str, collection: str, folder: str, url: str, key: str, splitter_name: str, chunk_size: int, embeddings: str, model: Optional[str], include_meta: bool, database: str, prefer_grpc: Optional[bool], log_level: str) -> Path:
    configure_logger(log_level)
    load_environment_keys(usecwd=True)
    papers_folder = Path(papers)
    assert not (folder is None and url is None and key is None), "either database folder or database url or api_key should be provided!"
    if splitter_name == "openai":
        # Create a RecursiveSplitterWithSource to split the documents into chunks of the specified size
        splitter = SourceTextSplitter(chunk_size=chunk_size)
    elif splitter_name == "recursive":
        splitter = OpenAISplitter(tokens=chunk_size)
    else:
        logger.warning(f"{splitter_name} is not supported, using openai tiktoken based splitter instead")
        splitter = OpenAISplitter(tokens=chunk_size)
    if embeddings == "llama" and model is None:
        model = os.getenv("LLAMA_MODEL")

    return index_selected_papers(papers_folder, collection, splitter, embeddings, include_meta, folder, url,
                                 database=VectorDatabase[database],
                                 key=key,
                                 model=model,
                                 prefer_grpc=prefer_grpc
                                 )


if __name__ == '__main__':
    app()