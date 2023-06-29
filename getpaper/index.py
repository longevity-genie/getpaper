#!/usr/bin/env python3

from pathlib import Path

import click
from click import Context
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, VertexAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma
from loguru import logger
from pycomfort.files import *

from getpaper.config import load_environment_keys, LOG_LEVELS, LogLevel, configure_logger
from getpaper.splitting import OpenAISplitter, SourceTextSplitter, papers_to_documents


def resolve_embeddings(embeddings_name: str) -> Embeddings:
    if embeddings_name == "openai":
        return OpenAIEmbeddings()
    elif embeddings_name == "lambda":
        return LlamaCppEmbeddings()
    elif embeddings_name == "vertexai":
        return VertexAIEmbeddings()
    else:
        logger.warning(f"{embeddings_name} is not yet supported by CLI, using default openai embeddings instead")
        return OpenAIEmbeddings()


def db_with_documents(db: Chroma, documents: list[Document],
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


def write_db(persist_directory: Path,
             collection_name: str,
             documents: list[Document],
             splitter: TextSplitter,
             id_field: Optional[str] = None,
             embeddings: Optional[Embeddings] = None
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
    db = Chroma(collection_name=collection_name, persist_directory=str(where), embedding_function=embeddings)

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


@logger.catch
def index_papers(papers_folder: Path, index: Path,
                 collection: str,
                 splitter: SourceTextSplitter,
                 embeddings: str,
                 include_meta: bool
                 ) -> Path:
    index.mkdir(exist_ok=True)
    openai_key = load_environment_keys()
    embeddings_function = resolve_embeddings(embeddings)
    logger.info(f"embeddings are {embeddings}")
    where = index / f"{embeddings}_{splitter.chunk_size}_chunk"
    where.mkdir(exist_ok=True, parents=True)
    logger.info(f"writing index of papers to {where}")
    documents = papers_to_documents(papers_folder, include_meta=include_meta)
    return write_db(where, collection, documents, splitter, embeddings=embeddings_function)


@app.command("index_papers")
@click.option('--papers', type=click.Path(exists=True), help="papers folder to index")
@click.option('--folder', type=click.Path(), help="folder to put chroma indexes to")
@click.option('--collection', default='papers', help='papers collection name')
@click.option('--splitter_name', type=click.Choice(["openai", "recursive"]), default="openai", help='which splitter to choose for the text splitting')
@click.option('--chunk_size', type=click.INT, default=3000, help='size of the chunk for splitting (characters for recursive spliiter and tokens for openai one)')
@click.option('--embeddings', type=click.Choice(["openai", "lambda", "vertexai"]), default="openai",
              help='size of the chunk for splitting')
@click.option('--include_meta', type=click.BOOL, default=True, help="if metadata is included")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_papers_command(papers: str, folder: str, collection: str, splitter_name: str, chunk_size: int, embeddings: str, include_meta: bool, log_level: str) -> Path:
    configure_logger(log_level)
    index = Path(folder)
    papers_folder = Path(papers)

    if splitter_name == "recursive":
        # Create a RecursiveSplitterWithSource to split the documents into chunks of the specified size
        splitter = SourceTextSplitter(chunk_size=chunk_size)
    elif splitter_name == "recursive":
        splitter = OpenAISplitter(tokens=chunk_size)
    else:
        logger.warning(f"{splitter_name} is not supported, using openai tiktoken based splitter instead")
        splitter = OpenAISplitter(tokens=chunk_size)
    return index_papers(papers_folder, index, collection, splitter, embeddings, include_meta)


if __name__ == '__main__':
    app()