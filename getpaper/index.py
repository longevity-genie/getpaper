#!/usr/bin/env python3

from pathlib import Path

import click
from click import Context

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

#TODO: finish indexing
"""
@app.command("index_papers")
@click.option('--collection', default='papers', help='papers collection name')
@click.option('--chunk_size', type=click.INT, default=6000, help='size of the chunk for splitting')
@click.option('--embeddings', type=click.Choice(["openai", "lambda", "vertexai"]), default="openai", help='size of the chunk for splitting')
@click.option('--base', default='.', help='base folder')
def index_papers(collection: str, chunk_size: int, embeddings: str,  base: str):
    locations = Locations(Path(base))
    openai_key = load_environment_keys()
    embeddings_function = resolve_embeddings(embeddings)
    print(f"embeddings are {embeddings}")
    where = locations.index / f"{embeddings}_{chunk_size}_chunk"
    where.mkdir(exist_ok=True, parents=True)
    print(f"writing index of papers to {where}")
    documents = papers_to_documents(locations.papers)
    return write_db(where, collection, documents, chunk_size, embeddings = embeddings_function)
"""