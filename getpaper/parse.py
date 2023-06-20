#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import click
import requests
from click import Context
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from pycomfort.files import traverse

def parse_paper(paper: Path, folder: Optional[Path] = None, mode: str = "single"):
    """
    Parses the paper using Unstructured paper parser
    :param paper:
    :param folder:
    :param mode: can be single or paged
    :return:
    """
    loader = UnstructuredPDFLoader(str(paper), mode=mode)
    where = paper.parent if folder is None else folder
    docs: list[Document] = loader.load()
    if len(docs) ==1:
        f = where / f"{paper.stem}.txt"
        print(f"writing {f}")
        f.write_text(docs[0].page_content)
        return [f]
    else:
        acc = []
        for i, doc in enumerate(docs):
            f = (where / f"{paper.stem}_{i}.txt")
            print(f"writing {f}")
            f.write_text(doc.page_content)
            acc.append(f)
        return acc

def parse_papers(parse_folder: Path, mode: str = "single", destination: Optional[Path] = None):
    papers: list[Path] = traverse(parse_folder, lambda p: "pdf" in p.suffix)
    print(f"indexing {len(papers)} papers")
    acc = []
    for i, paper in enumerate(papers):
        where = destination if destination is not None else paper.parent
        print(f"adding paper {i} out of {len(papers)}, will be saved to {where}")
        acc = acc + parse_paper(paper, where, mode)
    print("papers parsing finished!")
    return acc


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

@app.command("parse_paper")
@click.option('--paper', type=click.Path(exists=True), help="paper pdf to parse")
@click.option('--mode', type=click.Choice(["single", "paged"]), default="single", help="paper mode to be used")
@click.option('--destination', type=click.STRING, default=".", help="destination folder")
def parse_paper_command(paper: str, mode: str, destination: str):
    paper_file = Path(paper)
    destination_folder = Path(destination)
    print(f"parsing paper {paper} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_paper(paper_file, None, mode)

@app.command("parse_folder")
@click.option('--folder', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--mode', type=click.Choice(["single", "paged"]), default="single", help="paper mode to be used")
@click.option('--destination', type=click.STRING, default=None, help="destination folder")
def parse_paper_command(folder: str, mode: str, destination: str):
    parse_folder = Path(folder)
    destination_folder = Path(destination) if destination is not None else None
    print(f"parsing paper {folder} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_papers(parse_folder, mode, destination_folder)


if __name__ == '__main__':
    app()