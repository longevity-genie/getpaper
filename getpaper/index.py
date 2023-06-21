#!/usr/bin/env python3

from pathlib import Path
import dotenv
from dotenv import load_dotenv
import os
import click
from click import Context
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, VertexAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma
from pycomfort.files import *

from getpaper.parse import papers_to_documents
from getpaper.splitter import RecursiveSplitterWithSource


def load_environment_keys(debug: bool = True):
    e = dotenv.find_dotenv()
    if debug:
        print(f"environment found at {e}")
    has_env: bool = load_dotenv(e, verbose=True, override=True)
    if not has_env:
        print("Did not found environment file, using system OpenAI key (if exists)")
    openai_key = os.getenv('OPENAI_API_KEY')
    return openai_key

def resolve_embeddings(embeddings_name: str) -> Embeddings:
    if embeddings_name == "openai":
        return OpenAIEmbeddings()
    elif embeddings_name == "lambda":
        return LlamaCppEmbeddings()
    elif embeddings_name == "vertexai":
        return VertexAIEmbeddings()
    else:
        print(f"{embeddings_name} is not yet supported by CLI, using default openai embeddings instead")
        return OpenAIEmbeddings()

def db_with_documents(db: Chroma, documents: list[Document],
                      splitter: TextSplitter,
                      debug: bool = False,
                      id_field: Optional[str] = None):
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata[id_field] for doc in docs] if id_field is not None else None
    if debug:
        for doc in documents:
            print(f"ADD TEXT: {doc.page_content}")
            print(f"ADD METADATA {doc.metadata}")
    db.add_texts(texts=texts, metadatas=metadatas, ids = ids)
    return db


def write_db(persist_directory: Path,
             collection_name: str,
             documents: list[Document],
             chunk_size: int = 6000,
             debug: bool = False,
             id_field: Optional[str] = None,
             embeddings: Optional[Embeddings] = None):
    where = persist_directory / collection_name
    where.mkdir(exist_ok=True, parents=True)
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    db = Chroma(collection_name=collection_name, persist_directory=str(where), embedding_function=embeddings)
    splitter = RecursiveSplitterWithSource(chunk_size=chunk_size)
    splitter._chunk_size = chunk_size
    db_updated = db_with_documents(db, documents, splitter, debug, id_field)
    db_updated.persist()
    return where

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


@app.command("index_papers")
@click.option('--papers', type=click.Path(exists=True), help="papers folder to index")
@click.option('--folder', type=click.Path(), help="folder to put chroma indexes to")
@click.option('--collection', default='papers', help='papers collection name')
@click.option('--chunk_size', type=click.INT, default=6000, help='size of the chunk for splitting')
@click.option('--embeddings', type=click.Choice(["openai", "lambda", "vertexai"]), default="openai", help='size of the chunk for splitting')
@click.option('--base', default='.', help='base folder')
def index_papers(papers: str, folder: str, collection: str, chunk_size: int, embeddings: str,  base: str):
    index = Path(folder)
    papers_folder = Path(papers)
    index.mkdir(exist_ok=True)
    openai_key = load_environment_keys()
    embeddings_function = resolve_embeddings(embeddings)
    print(f"embeddings are {embeddings}")
    where = index  / f"{embeddings}_{chunk_size}_chunk"
    where.mkdir(exist_ok=True, parents=True)
    print(f"writing index of papers to {where}")
    documents = papers_to_documents(papers_folder)
    return write_db(where, collection, documents, chunk_size, embeddings = embeddings_function)