#!/usr/bin/env python3

import asyncio
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
import click
import requests
from click import Context
from functional import seq
from loguru import logger
from pynction import Try
from scidownl import scihub_download
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
import sys

from pycomfort.config import LOG_LEVELS, configure_logger, LogLevel

DownloadedPaper = (str, Optional[Path], Optional[Path]) #type synonim for doi, Path, Path of the downloaded paper


def _pdf_path_for_doi(doi: str, folder: Path, name: Optional[str] = None, create_parent: bool = True) -> Path:
    result = (folder / f"{doi}.pdf").absolute().resolve() if name is None else (folder / f"{name.replace('.pdf', '')}.pdf").absolute().resolve()
    if create_parent:
        if not result.parent.exists():
            result.parent.mkdir(exist_ok=True, parents=True)
    return result
def schihub_doi(doi: str, paper: Path, meta: Optional[Path] = None) -> (str, Optional[Path], Optional[Path]):
    doi_url = f"https://doi.org/{doi}"
    scihub_download(doi_url, paper_type="doi", out=str(paper))
    if paper.exists():
        logger.info(f"downloaded {doi_url} to {paper}")
    return doi, paper, meta

#@logger.catch(reraise=False)
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


#@logger.catch(reraise=False)
def download_semantic_scholar(paper_id: str, download: Optional[Path] = None, metadata: Optional[Path] = None, raise_on_no_pdf: bool = False) -> [str, str, str]:
    sch = SemanticScholar()
    paper: Paper = sch.get_paper(paper_id)
    if metadata is not None:
        json_data = json.dumps(paper.raw_data)
        metadata.touch(exist_ok=True)
        metadata.write_text(json_data)
        logger.info(f"metadata for {paper_id} successfully written to {metadata}")
    if download is not None:
        if paper.openAccessPdf is not None and "url" in paper.openAccessPdf:
            url = paper.openAccessPdf["url"]
            response = requests.get(url)
            response.raise_for_status()
            download.touch(exist_ok=True)
            download.write_bytes(response.content)
            logger.info(f"downloading open-access pdf for {paper_id} from {url} successfully downloaded to {download}")
        else:
            message = f"could not find open-access pdf for {paper_id}"
            if raise_on_no_pdf:
                raise Exception(message)
            else:
                logger.info(message)
                return paper_id, download, metadata
    return paper_id, download, metadata


def try_download(doi: str,
                 destination: Path,
                 skip_if_exist: bool = True,
                 name: Optional[str] = None,
                 ) -> Try:
    """
    downloads the paper by doi
    :param doi:
    :param destination: where to put the results
    :param skip_if_exist:
    :param name:
    :return: Try monad with the result
    """
    paper = _pdf_path_for_doi(doi, destination, name, True)
    meta = paper.parent / paper.name.replace(".pdf", "_meta.json")
    if skip_if_exist and paper.exists():
        if not meta.exists():
            logger.info(f"paper {paper} pdf already exists, however metadata {meta} does not, trying to download only metadata!")
            return Try.of(lambda: download_semantic_scholar(doi, None, meta))
        else:
            logger.info(f"paper {paper} already exists, skipping!")
    return Try.of(lambda: download_semantic_scholar(doi, paper, meta, raise_on_no_pdf=True)).catch(lambda _: schihub_doi(doi, paper, meta))


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


async def download_async(executor: Executor,
                             doi: str, destination: Path,
                             skip_if_exist: bool = True,
                             name: Optional[str] = None) -> DownloadedPaper:
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

    paper = _pdf_path_for_doi(doi, destination, name, True)
    meta = paper.parent / paper.name.replace(".pdf", "_meta.json")

    # Get a reference to the current event loop
    loop = asyncio.get_event_loop()

    _ = await loop.run_in_executor(executor, lambda: try_download(doi, destination, skip_if_exist, name))

    # Return the DOI and path to the downloaded paper
    return doi, Path, meta


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
        coroutines = [download_async(executor, doi, destination) for doi in dois]

        # Get the current event loop, run the downloads, and wait for all of them to finish
        loop = asyncio.get_event_loop()
        downloaded: List[DownloadedPaper] = loop.run_until_complete(asyncio.gather(*coroutines))
    partitions: List[List[DownloadedPaper]] = seq(downloaded).partition(lambda kv: isinstance(kv[1], Path)).to_list()
    return OrderedDict(partitions[0]), [kv[0] for kv in partitions[1]]


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


@app.command("download_semantic_scholar")
@click.option('--doi', type=click.STRING, help="download doi or other paper id")
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--skip_existing', type=click.BOOL, default=True, help="if it should skip downloading if the paper exists")
@click.option('--name', type=click.STRING, default=None, help="custom name, used doi of none")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG, help="logging level")
def download_semantic_scholar_command(doi: str, folder: str, skip_existing: bool = True, name: Optional[str] = None, log_level: str = LogLevel.DEBUG.value):
    configure_logger(log_level)
    logger.info(f"downloading {doi} to {folder}")
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    paper = _pdf_path_for_doi(doi, where, name, True)
    meta = paper.parent / paper.name.replace(".pdf", "_meta.json")
    if skip_existing and paper.exists():
        if not meta.exists():
            logger.info(f"paper {paper} pdf already exists, however metadata {meta} does not, trying to download only metadata!")
            return download_semantic_scholar(doi, paper, meta)
        else:
            logger.info(f"paper {paper} already exists, skipping!")
            return doi, paper, meta
    else:
        return download_semantic_scholar(doi, paper, meta)

@app.command("download_doi")
@click.option('--doi', type=click.STRING, help="download doi")
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--skip_existing', type=click.BOOL, default=True, help="if it should skip downloading if the paper exists")
@click.option('--name', type=click.STRING, default=None, help="custom name, used doi of none")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG, help="logging level")
def download_doi_command(doi: str, folder: str, skip_existing: bool = True, name: Optional[str] = None, log_level: str = "NONE") -> Try:
    configure_logger(log_level)
    logger.debug(f"downloading {doi} to {folder}")
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


def get_access(
        sch: SemanticScholar,
        paper_ids: List[str],
        fields: Optional[List[str]] = None
) -> List[dict]:
    if fields is None:
        fields = ["externalIds", "isOpenAccess", "openAccessPdf", "title", "year"]

    url = f'{sch.api_url}/paper/batch'

    fields = ','.join(fields)
    parameters = f'&fields={fields}'
    payload = { "ids": paper_ids }

    data = sch._requester.get_data(url, parameters, sch.auth_header, payload)
    return [item for item in data if item is not None]


def check_access(dois: List[str]) ->(List[Paper], List[Paper]):
    sch = SemanticScholar()
    papers_list = get_access(sch, dois)
    papers = seq(papers_list)
    result = papers.partition(lambda p: "openAccessPdf" in p and p["openAccessPdf"] is not None and "url" in p["openAccessPdf"])
    opened = result[0].to_list()
    closed = result[1].to_list()
    failed = len(dois) - len(papers_list)
    logger.info(f"{len(result[0].to_list())} papers out of {len(dois)}{' out of which ' + str(failed) + ' were not found' if failed >0 else ''}")
    return opened, closed

@app.command("access")
@click.option('--dois', multiple=True)
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default="debug", help="logging level")
def check_access_command(dois: List[str],  log_level: str):
    configure_logger(log_level)
    return check_access(dois)

@app.command("download_papers")
@click.option('--dois', multiple=True)
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--threads', '-t', type=int, default=5, help='Number of threads (default: 5)')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default="debug", help="logging level")
def download_papers_command(dois: List[str], folder: str, threads: int, log_level: str):
    """Downloads papers with the given DOIs to the specified destination."""
    configure_logger(log_level)
    if not dois:
        dois = []
    # Call the actual function with the provided arguments
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    return download_papers(dois, where, threads)


if __name__ == '__main__':
    app()