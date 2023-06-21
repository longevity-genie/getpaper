#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, List

import click
from click import Context
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from pycomfort.files import traverse


def parse_paper(paper: Path, folder: Optional[Path] = None,
                mode: str = "single", strategy: str = "auto",
                pdf_infer_table_structure: bool = True,
                include_page_breaks: bool = False
                ):
    """
    Parses the paper using Unstructured paper parser
    :param paper:
    :param folder:
    :param mode: can be single or paged
    :return:
    """
    bin_file = open(str(paper), "rb")
    loader = UnstructuredPDFLoader(file_path=None, file = bin_file,  mode=mode,
                                   pdf_infer_table_structure=pdf_infer_table_structure,
                                   strategy = strategy,
                                    include_page_breaks = include_page_breaks)
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

def parse_papers(parse_folder: Path, destination: Optional[Path] = None,
                 mode: str = "single", strategy: str = "auto",
                 pdf_infer_table_structure: bool = True,
                 include_page_breaks: bool = False):
    papers: list[Path] = traverse(parse_folder, lambda p: "pdf" in p.suffix)
    print(f"indexing {len(papers)} papers")
    acc = []
    for i, paper in enumerate(papers):
        where = destination if destination is not None else paper.parent
        print(f"adding paper {paper} which is {i} out of {len(papers)}. It will be saved to {where}")
        acc = acc + parse_paper(paper, where, mode, strategy, pdf_infer_table_structure, include_page_breaks)
    print("papers parsing finished!")
    return acc

def papers_to_documents(folder: Path, suffix: str = ""):
    txt = traverse(folder, lambda p: "txt" in p.suffix)
    texts = [t for t in txt if suffix in t.name] if suffix != "" else txt
    docs: List[Document] = []
    for t in texts:
        doi = f"http://doi.org/{t.parent.name}/{t.stem}"
        with open(t, 'r', encoding="utf-8") as file:
            text = file.read()
            if len(text)<10:
                print("TOO SHORT TEXT")
            else:
                doc = Document(
                    page_content = text,
                    metadata={"source": doi}
                )
                docs.append(doc)
    return docs

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

@app.command("parse_paper")
@click.option('--paper', type=click.Path(exists=True), help="paper pdf to parse")
@click.option('--destination', type=click.STRING, default=".", help="destination folder")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single", help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="auto", help="parsing strategy to be used, auto by default")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included")
def parse_paper_command(paper: str, destination: str, mode: str, strategy: str, infer_tables: bool, include_page_breaks: bool):
    paper_file = Path(paper)
    destination_folder = Path(destination)
    print(f"parsing paper {paper} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_paper(paper_file, None, mode, strategy, infer_tables, include_page_breaks)

@app.command("parse_folder")
@click.option('--folder', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--destination', type=click.STRING, default=None, help="destination folder")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single", help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="auto", help="parsing strategy to be used, auto by default")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included")
def parse_paper_command(folder: str,destination: str, mode: str, strategy: str, infer_tables: bool, include_page_breaks: bool):
    parse_folder = Path(folder)
    destination_folder = Path(destination) if destination is not None else None
    print(f"parsing paper {folder} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_papers(parse_folder, destination_folder, mode, strategy, infer_tables, include_page_breaks)


if __name__ == '__main__':
    app()