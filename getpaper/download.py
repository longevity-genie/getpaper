#!/usr/bin/env python3

from __future__ import annotations
import asyncio
import json
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
import click
import loguru
import requests
from click import Context
from functional import seq
from pynction import Try
from scidownl import scihub_download
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from dataclasses import dataclass
import sys

from pycomfort.config import LOG_LEVELS, configure_logger, LogLevel
from unpywall import Unpywall
from unpywall.utils import UnpywallCredentials

#DownloadedPaper = (str, Optional[Path], Optional[Path]) #type synonim for doi, Path, Path of the downloaded paper
@dataclass
class PaperDownload:
    id: str
    pdf: Optional[Path]
    metadata: Optional[Path]
    parsed: Optional[list[Path]] = None
    url: Optional[str] = None

    def with_pdf(self, pdf: Path):
        self.pdf = pdf
        return self

    def with_url(self, url: str):
        self.url = url
        return self

    def with_parsed(self, parsed: Optional[list[Path]]):
        self.parsed = parsed
        return self

def _pdf_path_for_doi(doi: str, folder: Path, name: Optional[str] = None, create_parent: bool = True) -> Path:
    result = (folder / f"{doi}.pdf").absolute().resolve() if name is None else (folder / f"{name.replace('.pdf', '')}.pdf").abewsw1wswqssolute().resolve()
    if create_parent:
        if not result.parent.exists():
            result.parent.mkdir(exist_ok=True, parents=True)
    return result

def schihub_doi(doi: str, paper: Path, meta: Optional[Path] = None, logger: Optional["loguru.Logger"] = None) -> PaperDownload:
    """
    If you use scihub you should know that it can resolve both openaccess and paywalled articles.
    If you download paywalled articles it can be illegal in some of the countries.
    We are not responsible for how you are using the software, so do it at your own risk.
    :param doi: papers doi
    :param paper:
    :param meta:
    :return:
    """
    if logger is None:
        logger = loguru.logger
    doi_url = f"https://doi.org/{doi}"
    scihub_download(doi_url, paper_type="doi", out=str(paper))
    if paper.exists():
        logger.info(f"downloaded {doi_url} to {paper}")
    return PaperDownload(doi, paper, meta)


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


def resolve_semantic_scholar(paper_id: str,
                              metadata: Optional[Path] = None,
                              logger: Optional["loguru.Logger"] = None) -> PaperDownload:
    if logger is None:
        logger = loguru.logger
    sch = SemanticScholar()
    paper: Paper = sch.get_paper(paper_id)
    if metadata is not None:
        json_data = json.dumps(paper.raw_data)
        metadata.touch(exist_ok=True)
        metadata.write_text(json_data)
        logger.info(f"metadata for {paper_id} successfully written to {metadata}")
    if paper.openAccessPdf is not None and "url" in paper.openAccessPdf:
        url = paper.openAccessPdf["url"]
    else:
        url = None
    return PaperDownload(paper_id, None, metadata, None, url)

#@logger.catch(reraise=False)
def download_semantic_scholar(paper_id: str,
                              download,
                              metadata: Optional[Path] = None,
                              raise_on_no_pdf: bool = False,
                              headers: Optional[dict] = None,
                              logger: Optional["loguru.Logger"] = None) -> PaperDownload:
    if logger is None:
        logger = loguru.logger
    sch = SemanticScholar()
    paper: Paper = sch.get_paper(paper_id)
    if metadata is not None:
        json_data = json.dumps(paper.raw_data)
        metadata.touch(exist_ok=True)
        metadata.write_text(json_data)
        logger.info(f"metadata for {paper_id} successfully written to {metadata}")
    if paper.openAccessPdf is not None and "url" in paper.openAccessPdf:
        url = paper.openAccessPdf["url"]
        if download is not None:
            simple_download(download, headers, logger, paper_id, url)
        else:
            message = f"could not find open-access pdf for {paper_id}"
            if raise_on_no_pdf:
                raise Exception(message)
            else:
                logger.info(message)
                return PaperDownload(paper_id, download, metadata, url = url)
    return PaperDownload(paper_id, download, metadata)


def simple_download(url: str, download: Path, headers: Optional[dict] = None, logger: Optional["loguru.Logger"] = None):
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    download.touch(exist_ok=True)
    download.write_bytes(response.content)
    logger.info(f"downloading open-access pdf for {url} successfully downloaded to {download}")
    return download


def try_download(doi: str,
                 destination: Path,
                 skip_if_exist: bool = True,
                 name: Optional[str] = None,
                 scihub_on_fail: bool = False,
                 unpaywall_email: Optional[str] = None,
                 selenium_on_fail: bool = False,
                 selenium_headless: bool=True, selenium_min_wait: int=12, selenium_max_wait: int=60,
                 logger: Optional["loguru.Logger"] = None
                 ) -> Try[PaperDownload]:
    """
    :param doi:
    :param destination:
    :param skip_if_exist:
    :param name:
    :param scihub_on_fail: use sci-hub on fail. Note: use it at your own risk. If the paper you download is not open-access in some countries and cases sci-hub use will not be legal
    :param unpaywall_email:
    :param selenium_on_fail:
    :param selenium_headless:
    :param selenium_min_wait:
    :param selenium_max_wait:
    :param logger:
    :return:
    """
    if logger is None:
        logger = loguru.logger
    # for example from https://doi.org/10.1016/j.stemcr.2022.09.009
    doi = doi.replace("https://doi.org/", "")
    paper = _pdf_path_for_doi(doi, destination, name, True)
    meta = paper.parent / paper.name.replace(".pdf", "_meta.json")
    if skip_if_exist and paper.exists():
        if not meta.exists():
            logger.info(f"paper {paper} pdf already exists, however metadata {meta} does not, trying to download only metadata!")
            return Try.of(lambda: download_semantic_scholar(doi, None, meta))
        else:
            logger.info(f"paper {paper} already exists, skipping!")
            return Try.of(lambda: PaperDownload(doi, paper, meta))
    p: PaperDownload = resolve_semantic_scholar(doi, meta)
    if unpaywall_email is not None:
        unpaywall_email = os.getenv("UNPAYWALL_MAIL")
    if unpaywall_email is not None and p.url is None:
        UnpywallCredentials()
        p.url = Unpywall.get_pdf_link(doi)
    try_simple = Try.of(lambda: p.with_pdf(simple_download(p.url, logger = logger)))
    if selenium_on_fail:
        from getpaper.selenium_download import download_pdf_selenium
        before_last = try_simple.catch(lambda ex: p.with_pdf(download_pdf_selenium(p.url, destination, selenium_headless, selenium_min_wait, selenium_max_wait, final_path=paper, logger=logger)))
    else:
        before_last = try_simple
    before_last.on_failure(lambda e: logger.error(e))
    return before_last.catch(lambda _: schihub_doi(doi, paper, meta)) if scihub_on_fail else before_last


def download_pubmed(pubmed: str, destination: Path, skip_if_exist: bool = True, name: Optional[str] = None, scihub_on_fail: bool = False):
    """
    downloads paper by its pubmed id
    :param pubmed: pubmed id
    :param destination: where to store the result
    :param skip_if_exist:
    :param name:
    :param scihub_on_fail: if SciHub should be used as back up resolver. False by default. For paywalled articles it can be illegal in some of the countries, so use it at your own risk.
    :return:
    """
    try_resolve = try_doi_from_pubmed(pubmed)
    return try_resolve.flat_map(lambda doi: try_download(doi, destination, skip_if_exist, name, scihub_on_fail))


async def download_async(executor: Executor,
                             doi: str, destination: Path,
                             skip_if_exist: bool = True,
                             name: Optional[str] = None,
                             scihub_on_fail: bool = False,
                             logger: Optional["loguru.Logger"] = None
                         ) -> PaperDownload:
    """
    Asynchronously download a paper using its DOI.

    Args:
        executor (Executor): The ThreadPoolExecutor to run blocking IO in.
        doi (str): The DOI of the paper to download.
        destination (Path): The directory where the downloaded paper should be stored.
        skip_if_exist (bool): If True, skip the download if the paper already exists in the destination. Default is True.
        name (Optional[str]): The name of the file to save the paper as. If not provided, use the DOI. Default is None.
        param scihub_on_fail: if SciHub should be used as back up resolver. False by default. For paywalled articles it can be illegal in some of the countries, so use it at your own risk.


    Returns:
        (str, Path): A tuple containing the DOI of the paper and the path to the downloaded paper.
    """
    if logger is None:
        logger = loguru.logger

    paper = _pdf_path_for_doi(doi, destination, name, True)
    meta = paper.parent / paper.name.replace(".pdf", "_meta.json")

    # Get a reference to the current event loop
    loop = asyncio.get_event_loop()

    _ = await loop.run_in_executor(executor, lambda: try_download(doi, destination, skip_if_exist, name, scihub_on_fail, logger))

    # Return the DOI and path to the downloaded paper
    return PaperDownload(doi, paper, meta)


def download_papers(dois: List[str], destination: Path, threads: int, scihub_on_fail: bool = False) -> (OrderedDict[str, PaperDownload], List[str]):
    """
    :param dois: List of DOIs of the papers to download
    :param destination: Directory where to put the downloaded papers
    :param threads: Maximum number of concurrent downloads
    :param scihub_on_fail: if SciHub should be used as back up resolver. False by default. For paywalled articles it can be illegal in some of the countries, so use it at your own risk.
    :return: tuple with OrderedDict of succeeded results and list of failed dois)
    """
    # Create a ThreadPoolExecutor with desired number of threads
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Create a coroutine for each download
        coroutines = [download_async(executor, doi, destination, scihub_on_fail=scihub_on_fail) for doi in dois]
        #TODO: can be problematic
        # Get the current event loop, run the downloads, and wait for all of them to finish
        loop = asyncio.get_event_loop()
        downloaded: List[PaperDownload] = loop.run_until_complete(asyncio.gather(*coroutines))
    partitions: List[List[PaperDownload]] = seq(downloaded).partition(lambda kv: isinstance(kv.pdf, Path)).to_list()
    return OrderedDict([(d.id, d) for d in partitions[0]]), [kv.pdf for kv in partitions[1]]


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
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def download_semantic_scholar_command(doi: str, folder: str, skip_existing: bool = True, name: Optional[str] = None, log_level: str = LogLevel.DEBUG.value):
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
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
@click.option('--doi', required=True, type=click.STRING, help="download doi")
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--skip_existing', type=click.BOOL, default=True, help="if it should skip downloading if the paper exists")
@click.option('--name', type=click.STRING, default=None, help="custom name, used doi of none")
@click.option('--selenium_on_fail', type=click.BOOL, default=False, help="use selenium for cases when it fails")
@click.option('--scihub_on_fail', type=click.BOOL, default=False, help="if schihub should be used as backup resolver. Use it at your own risk and responsibility (false by default)")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def download_doi_command(doi: str, folder: str, skip_existing: bool = True, name: Optional[str] = None, selenium_on_fail: bool = False, scihub_on_fail: bool = False, log_level: str = "NONE") -> Try:
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
    logger.debug(f"downloading {doi} to {folder}")
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    result = try_download(doi, where, skip_existing, name, selenium_on_fail=selenium_on_fail, scihub_on_fail=scihub_on_fail)
    result.on_success(lambda p: logger.info(f"successfully downloaded {p.id} from {p.url} to {p.pdf}"))
    return result


@app.command("download_pubmed")
@click.option('--pubmed', type=click.STRING, help="download doi")
@click.option('--folder', type=click.Path(), default=".", help="where to download the paper")
@click.option('--skip_existing', type=click.BOOL, default=True, help="if it should skip downloading if the paper exists")
@click.option('--scihub_on_fail', type=click.BOOL, default=False, help="if schihub should be used as backup resolver. Use it at your own risk and responsibility (false by default)")
@click.option('--name', type=click.STRING, default=None, help="custom name, uses doi if none and pubmed id if pmid")
def download_pubmed_command(pubmed: str, folder: str, skip_existing: bool, name: Optional[str], scihub_on_fail: bool = False):
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    custom_name = pubmed if name == "pmid" or name == "PMID" else name
    return download_pubmed(pubmed, where, skip_existing, custom_name, scihub_on_fail=scihub_on_fail)


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


def check_access(dois: List[str], logger: Optional["loguru.Logger"] = None) ->(List[Paper], List[Paper]):
    if logger is None:
        logger = loguru.logger
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
@click.option('--scihub_on_fail', type=click.BOOL, default=False, help="if schihub should be used as backup resolver. Use it at your own risk and responsibility (false by default)")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default="debug", help="logging level")
def download_papers_command(dois: List[str], folder: str, threads: int, scihub_on_fail: bool, log_level: str):
    """Downloads papers with the given DOIs to the specified destination."""
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
    if not dois:
        dois = []
    # Call the actual function with the provided arguments
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    return download_papers(dois, where, threads, scihub_on_fail)


if __name__ == '__main__':
    app()