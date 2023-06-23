#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List

import click
import requests
from click import Context
from functional import seq
from pynction import Try
from scidownl import scihub_download
import asyncio
from concurrent.futures import ThreadPoolExecutor, Executor
from pathlib import Path

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def doi_from_pubmed(pubmed_id: str):
    """
    resolves doi by pubmed id
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


def try_download(doi: str, destination: Path, skip_if_exist: bool = True, name: Optional[str] = None) -> Try[Path]:
    """
    downloads the paper by doi
    :param doi:
    :param destination: where to put the results
    :param skip_if_exist:
    :param name:
    :return: Try monad with the result
    """
    doi_url = f"https://doi.org/{doi}"
    paper = (destination / f"{doi}.pdf").absolute().resolve() if name is None else (destination / f"{name.replace(',pdf', '')}.pdf").absolute().resolve()
    if skip_if_exist and paper.exists():
        print(f"Paper {paper} for {doi} already exists!")
        return Try.of(lambda: paper)
    return Try.of(lambda: scihub_download(doi_url, paper_type="doi", out=str(paper))).map(lambda _: paper)

def download_pubmed(pubmed: str, destination: Path, skip_if_exist: bool = True, name: Optional[str] = None):
    """
    downloads paper by its pubmed id
    :param pubmed: pubmed id
    :param destination: where to store the result
    :param skip_if_exist:
    :param name:
    :return:
    """
    try_resolve = try_doi_from_pubmed(pubmed)
    return try_resolve.flat_map(lambda doi: try_download(doi, destination, skip_if_exist, name))


async def try_download_async(executor: Executor, doi: str, destination: Path, skip_if_exist: bool = True, name: Optional[str] = None) -> (str, Path):
    """
    Asynchronously download a paper using its DOI.

    Args:
        executor (Executor): The ThreadPoolExecutor to run blocking IO in.
        doi (str): The DOI of the paper to download.
        destination (Path): The directory where the downloaded paper should be stored.
        skip_if_exist (bool): If True, skip the download if the paper already exists in the destination. Default is True.
        name (Optional[str]): The name of the file to save the paper as. If not provided, use the DOI. Default is None.

    Returns:
        (str, Path): A tuple containing the DOI of the paper and the path to the downloaded paper.
    """

    # Construct the URL for the paper using the DOI
    doi_url = f"https://doi.org/{doi}"
    paper = (destination / f"{doi}.pdf").absolute().resolve() if name is None else (destination / f"{name.replace(',pdf', '')}.pdf").absolute().resolve()

    # If the paper already exists and we are skipping existing papers, return the DOI and path
    if skip_if_exist and paper.exists():
        print(f"Paper {paper} for {doi} already exists!")
        return doi, paper

    # Get a reference to the current event loop
    loop = asyncio.get_event_loop()

    def blocking_io():
        # Download the paper
        # This is a blocking IO operation, so it's run in the executor
        scihub_download(doi_url, paper_type="doi", out=str(paper))
        return paper  # Explicitly return the path of the downloaded file

    _ = await loop.run_in_executor(executor, blocking_io)

    # Return the DOI and path to the downloaded paper
    return doi, Path


def download_papers(dois: List[str], destination: Path, threads: int) -> (OrderedDict[str, Path], List[str]):
    """
    :param dois: List of DOIs of the papers to download
    :param destination: Directory where to put the downloaded papers
    :param threads: Maximum number of concurrent downloads
    :return: tuple with OrderedDict of succeeded results and list of failed dois)
    """
    # Create a ThreadPoolExecutor with desired number of threads
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Create a coroutine for each download
        coroutines = [try_download_async(executor, doi, destination) for doi in dois]

        # Get the current event loop, run the downloads, and wait for all of them to finish
        loop = asyncio.get_event_loop()
        downloaded: List[(str, Path)] = loop.run_until_complete(asyncio.gather(*coroutines))
    partitions: List[List[(str, Path)]] = seq(downloaded).partition(lambda kv: isinstance(kv[1], Path)).to_list()
    return OrderedDict(partitions[0]), [kv[0] for kv in partitions[1]]



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
def download_doi_command(doi: str, folder: str, skip_existing: bool = True, name: Optional[str] = None) -> Try:
    print(f"downloading {doi} to {folder}")
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

@app.command("download_papers")
@click.option('--dois', multiple=True)
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--threads', '-t', type=int, default=5, help='Number of threads (default: 5)')
def download_papers_command(dois: List[str], folder: str, threads: int):
    """Downloads papers with the given DOIs to the specified destination."""
    if not dois:
        dois = []
    # Call the actual function with the provided arguments
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    return download_papers(dois, where, threads)


if __name__ == '__main__':
    app()