#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import click
import requests
from click import Context
from pycomfort.files import with_ext
from pynction import Try
from scidownl import scihub_download

def doi_from_pubmed(pubmed_id: str):
    """
    resolves doi by pubmedid
    :param pubmed_id:
    :return:
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pubmed_id,
        "retmode": "xml"
    }
    response = requests.get(base_url, params=params)
    #response.raise_for_status()
    root = ET.fromstring(response.content)
    article = root.find(".//PubmedArticle")
    if article is None:
        raise ValueError("PubMed ID not found")
    doi_element = article.find(".//ArticleId[@IdType='doi']")
    if doi_element is None:
        raise ValueError("DOI not found for this PubMed ID")
    return doi_element.text

def try_doi_from_pubmed(pubmed: str) -> Try[str]:
    """
    doi_from_pubmed wrapped in try
    :param pubmed:
    :return:
    """
    return Try.of(lambda: doi_from_pubmed(pubmed))

def try_download(doi: str, papers: Path, skip_if_exist: bool = True, name: Optional[str] = None) -> Try[Path]:
    doi_url = f"https://doi.org/{doi}"
    paper = (papers / f"{doi}.pdf").absolute().resolve() if name is None else (papers / f"{name.replace(',pdf', '')}.pdf").absolute().resolve()
    if skip_if_exist and paper.exists():
        print(f"Paper {paper} for {doi} already exists!")
        return paper
    return Try.of(lambda: scihub_download(doi_url, paper_type="doi", out=str(paper))).map(lambda _: paper)

def download_pubmed(pubmed: str, papers: Path, skip_if_exist: bool = True, name: Optional[str] = None):
    try_resolve = try_doi_from_pubmed(pubmed)
    return try_resolve.flat_map(lambda doi: try_download(doi, papers, skip_if_exist, name))


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

@app.command("download_doi")
@click.option('--doi', type=click.STRING, help="download doi")
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--skip_existing', type=click.BOOL, default=True, help="if it should skip downloading if the paper exists")
@click.option('--name', type=click.STRING, default=None, help="custom name, used doi of none")
def download_doi_command(doi: str, folder: str, skip_existing: bool, name: Optional[str]):
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    return try_download(doi, where, skip_existing, name)

@app.command("download_pubmed")
@click.option('--pubmed', type=click.STRING, help="download doi")
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--skip_existing', type=click.BOOL, default=True, help="if it should skip downloading if the paper exists")
@click.option('--name', type=click.STRING, default=None, help="custom name, uses doi if none and pubmed id if pmid")
def download_pubmed_command(pubmed: str, folder: str, skip_existing: bool, name: Optional[str]):
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    custom_name = pubmed if name == "pmid" or name == "PMID" else name
    return download_pubmed(pubmed, where, skip_existing, custom_name)



if __name__ == '__main__':
    app()