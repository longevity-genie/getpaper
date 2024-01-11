#!/usr/bin/env python3
from __future__ import annotations
import sys
from enum import Enum
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, List
import click
import loguru
import pynction.monads.try_monad
import tiktoken
from click import Context
from functional import seq
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader, PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader
from langchain.schema import Document
from pycomfort.files import traverse, files
from pynction import Try
import loguru
from getpaper.download import try_download, PaperDownload

from pycomfort.config import configure_logger, LOG_LEVELS, LogLevel


class PDFParser(Enum):
    unstructured = "unstructured"
    pdf_miner = "pdf_miner"
    py_pdf = "py_pdf"
    py_mu_pdf = "py_mu_pdf"
    pdfplumber = "pdfplumber"


def num_tokens_openai(string: str, model: str, price_per_1k: float = 0.0001) -> (int, float):
    """Returns the number of tokens for a model"""
    encoding = tiktoken.encoding_for_model(model)
    n_tokens = len(encoding.encode(string))
    return n_tokens, n_tokens / 1000.0 * price_per_1k


"""
    Model	Input	Output
8K context	$0.03 / 1K tokens	$0.06 / 1K tokens
32K context	$0.06 / 1K tokens	$0.12 / 1K tokens
"""

openai_prices_per_thousand = {
    "gpt-4-32k": None,  # 32,768 tokens
    "gpt-4": 0.03,  # 8,192 tokens
    "gpt-3.5-turbo-16k": None,  # 16,384 tokens,
    "gpt-3.5-turbo": None  # 4,096 tokens
}


def clean_text(text: str) -> str:
    from unstructured.cleaners.core import clean, group_broken_paragraphs, replace_unicode_quotes
    return clean(group_broken_paragraphs(replace_unicode_quotes(text)))


def parse_paper(paper: Path, folder: Optional[Path] = None,
                parser: PDFParser = PDFParser.py_mu_pdf,
                recreate_parent: bool = False,
                subfolder: bool = True,
                do_not_reparse: bool = True,
                cleaning: bool = False,
                mode: str = "single", strategy: str = "auto",
                pdf_infer_table_structure: bool = True,
                include_page_breaks: bool = False,
                logger: Optional["loguru.Logger"] = None
                ) -> List[Path]:
    """
    :param paper:
    :param folder:
    :param parser:
    :param recreate_parent:
    :param cleaning:
    :param mode: unstructured specific
    :param strategy:  unstructured specific
    :param pdf_infer_table_structure:  unstructured specific
    :param include_page_breaks:  unstructured specific
    :return:
    """
    if logger is None:
        logger = loguru.logger
    if parser is None or parser == PDFParser.unstructured:
        loader = init_unstructured_loader(open(str(paper), "rb"), include_page_breaks, mode, pdf_infer_table_structure,
                                          strategy)
    elif parser == PDFParser.pdf_miner:
        loader = PDFMinerLoader(str(paper))
    elif parser == PDFParser.py_mu_pdf:
        loader = PyMuPDFLoader(str(paper))
    elif parser == PDFParser.py_pdf:
        loader = PyPDFLoader(str(paper))
    elif parser == PDFParser.pdfplumber:
        loader = PDFPlumberLoader(str(paper))
    elif parser == PDFParser.unstructured:
        loader = init_unstructured_loader(open(str(paper), "rb"), include_page_breaks, mode, pdf_infer_table_structure,
                                          strategy)
    else:
        loader = PDFPlumberLoader(str(paper))
    where = paper.parent if folder is None else folder / paper.parent.name if recreate_parent else folder
    where.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = loader.load()
    upd_where: Path = (where / paper.stem) if subfolder else where
    upd_where.mkdir(exist_ok=True)
    if upd_where.exists() and subfolder and do_not_reparse and files(upd_where).len() > 0:
        logger.info(f"avoiding re-parsing, providing result {upd_where}")
        return [upd_where]
    if len(docs) == 1:
        return [upd_where] if subfolder else [write_parsed(docs[0], paper, upd_where, cleaning)]
    else:
        acc = []
        for i, doc in enumerate(docs):
            f = write_parsed(doc, paper, upd_where, cleaning, i)
            acc.append(f)
        return [upd_where] if subfolder else acc


def write_parsed(doc: Document, paper: Path, where: Path, cleaning: bool = False, i: int = -1,
                 logger: Optional["loguru.Logger"] = None):
    if logger is None:
        logger = loguru.logger
    name = f"{paper.stem}_{i}.txt" if i >= 0 else f"{paper.stem}.txt"
    f = (where / name)
    logger.trace(f"writing {f}")
    text = clean_text(doc.page_content) if cleaning else doc.page_content
    f.write_text(text)
    return f


def init_unstructured_loader(bin_file, include_page_breaks, mode, pdf_infer_table_structure, strategy: str):
    return UnstructuredPDFLoader(file_path=None, file=bin_file, mode=mode,
                                 pdf_infer_table_structure=pdf_infer_table_structure,
                                 strategy=strategy, include_page_breaks=include_page_breaks)


def try_parse_paper(paper: Path, folder: Optional[Path] = None,
                    parser: PDFParser = PDFParser.unstructured,
                    recreate_parent: bool = False,
                    subfolder: bool = True,
                    do_not_reparse: bool = True,
                    cleaning: bool = False,
                    mode: str = "single", strategy: str = "auto",
                    pdf_infer_table_structure: bool = True,
                    include_page_breaks: bool = False,
                    logger: Optional["loguru.Logger"] = None
                    ) -> Try[List[Path]]:
    return Try.of(
        lambda: parse_paper(paper, folder, parser, recreate_parent, cleaning, subfolder, do_not_reparse, mode, strategy,
                            pdf_infer_table_structure, include_page_breaks, logger=logger))


def parse_papers(parse_folder: Path, destination: Optional[Path] = None,
                 parser: PDFParser = PDFParser.py_mu_pdf,
                 recreate_parent: bool = False, cores: Optional[int] = None,
                 subfolder: bool = True,
                 do_not_reparse: bool = True,
                 cleaning: bool = False,
                 mode: str = "single", strategy: str = "auto",
                 pdf_infer_table_structure: bool = True,
                 include_page_breaks: bool = False,
                 logger: Optional["loguru.Logger"] = None
                 ):
    """
    Function to parse multiple papers using multiple cores.
    The function employs multiprocessing to speed up the process.

    Args:
        parse_folder (Path): The folder where the papers (PDF files) are located.
        destination (Optional[Path]): The destination folder where parsed papers will be saved.
                                      If not provided, the parsed papers are saved in their original folder.
        parser (PDFParser): which pdf parsing library to choose
        mode (str): The mode to parse the papers. Default is "single".
        strategy (str): The strategy to parse the papers. Default is "auto".
        pdf_infer_table_structure (bool): If True, attempts to infer table structure in PDFs. Default is True.
        include_page_breaks (bool): If True, includes page breaks in the parsed output. Default is False.
        recreate_parent (bool): If True, recreates the parent directory structure in the destination folder. Default is False.
        cores (Optional[int]): The number of cores to use. If not provided, uses all available cores.

    Returns:
        list[Path], list[Failure]: A list of paths to the parsed papers and a list of Failtures
    """
    if logger is None:
        logger = loguru.logger
    papers: list[Path] = traverse(parse_folder, lambda p: "pdf" in p.suffix)
    logger.info(f"indexing {len(papers)} papers")
    parsed = []
    errors = []

    cores = cpu_count() if cores is None else min(cpu_count(), cores)

    with Pool(cores) as p:
        parse_func = partial(try_parse_paper, folder=destination, parser=parser,
                             recreate_parent=recreate_parent, subfolder=subfolder, do_not_reparse=do_not_reparse,
                             cleaning=cleaning,
                             mode=mode, strategy=strategy,
                             pdf_infer_table_structure=pdf_infer_table_structure,
                             include_page_breaks=include_page_breaks, logger=logger
                             )
        results = p.map(parse_func, papers)
        for result in results:
            if isinstance(result, pynction.monads.try_monad.Success):
                parsed = parsed + result._value
            elif isinstance(result, pynction.monads.try_monad.Failure):
                errors.append(result._e)
            else:
                logger.warning(f"unpredicted type of the {result}")

    logger.info("papers parsing finished!")
    if len(errors) > 0:
        logger.warning(f"errors discovered: {errors}")
    return results, errors


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


@app.command("count_tokens")
@click.option('--path', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--model', default='gpt-3.5-turbo-16k', help='model to use, gpt-3.5-turbo-16k by default')
@click.option("--suffix", default=".txt", help="suffix in the files to evaluate, .txt by default")
@click.option("--price", type=click.FLOAT, default=0.0001, help="price for 1K tokens")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value,
              help="logging level")
def count_tokens_command(path: Path, model: str, suffix: str, price: float, log_level: str):
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
    where = Path(path)
    if where.is_dir():
        papers: list[Path] = traverse(where, lambda p: suffix in p.name)
        tokens_price = seq(papers).map(lambda p: num_tokens_openai(p.read_text(encoding="utf-8"), model, price))
        num = tokens_price.map(lambda r: r[0])
        money = tokens_price.map(lambda r: r[1])
        num_sum = num.sum()
        money_sum = money.sum()
        num_avg = num.average()
        money_avg = money.average()
        num_max = num.max()
        money_max = money.max()
        logger.info(f"Checked {len(papers)} papers. TOTAL TOKENS = {num_sum} , COST = {money_sum}")
        logger.info(
            f"PER PAPER: \n average tokens {num_avg} , cost {money_avg}\n max tokens = {num_max} , max cost = {money_max}")
        return num, money
    else:
        content = path.read_text(encoding="utf-8")
        return num_tokens_openai(content, model)


def try_download_and_parse(doi: str,
                           destination: Path,
                           selenium_on_fail: bool = False,
                           scihub_on_fail: bool = False,
                           parser: PDFParser = PDFParser.py_mu_pdf.value,
                           subfolder: bool = True,
                           do_not_reparse: bool = True,
                           cleaning: bool = False,
                           mode: str = "single",
                           strategy: str = "fast",
                           infer_tables: bool = True,
                           include_page_breaks: bool = False,
                           recreate_parent: bool = True,
                           selenium_headless: bool=True, selenium_min_wait: int=15, selenium_max_wait: int=60,
                           unpaywall_email: Optional[str] = None,
                           logger: Optional["loguru.Logger"] = None) -> Try[PaperDownload]:
    """
    :param doi: paper's doi
    :param destination:
    :param selenium_on_fail: if use selenium on download failure
    :param scihub_on_fail:
    :param parser:
    :param subfolder:
    :param do_not_reparse:
    :param cleaning:
    :param mode:
    :param strategy:
    :param infer_tables:
    :param include_page_breaks:
    :param recreate_parent:
    :param selenium_headless:
    :param selenium_min_wait:
    :param selenium_max_wait:
    :param logger:
    :return:
    """
    if logger is None:
        logger = loguru.logger
    result: Try[PaperDownload] = try_download(doi, destination, skip_if_exist=True,
                                              scihub_on_fail=scihub_on_fail,
                                              selenium_on_fail=selenium_on_fail, selenium_headless=selenium_headless,
                                              selenium_min_wait=selenium_min_wait, selenium_max_wait=selenium_max_wait,
                                              unpaywall_email=unpaywall_email,
                                              logger=logger)
    result.on_failure(lambda ex: logger.error(f"Could not resolve the paper {doi} with the following error {str(ex)}"))
    return result.map(lambda r: r.with_parsed(parse_paper(r.pdf, None, parser, recreate_parent,
                                                          subfolder, do_not_reparse, cleaning,
                                                          mode, strategy, infer_tables, include_page_breaks)))


@app.command("download_and_parse")
@click.option('--doi', type=click.STRING, required=True, help="doi of the paper to parse")
@click.option('--folder', type=click.STRING, default=".", help="destination folder")
@click.option('--selenium_on_fail', type=click.BOOL, default=False, help="use selenium for cases when it fails")
@click.option('--scihub_on_fail', type=click.BOOL, default=False,
              help="if schihub should be used as backup resolver. Use it at your own risk and responsibility (false by default)")
@click.option('--parser', type=click.Choice([loader.value for loader in PDFParser]), default=PDFParser.py_mu_pdf.value,
              help="pdf parser to choose from, unstructured by default")
@click.option('--subfolder', type=click.BOOL, default=True, help="if it should create a folder per paper")
@click.option('--do_not_reparse', type=click.BOOL, default=True, help="if we should avoid reparsing")
@click.option('--cleaning', type=click.BOOL, default=False, help="if we should use basic cleaning for the text")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single",
              help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="fast",
              help="parsing strategy to be used, auto by default, unstructured parser specific")
@click.option('--infer_tables', type=click.BOOL, default=True,
              help="if the table structure should be inferred, unstructured parser specific")
@click.option('--include_page_breaks', type=click.BOOL, default=False,
              help="if page breaks should be included, unstructured parser specific")
@click.option('--recreate_parent', type=click.BOOL, default=False,
              help="if parent folder should be recreated in the new destination")
@click.option('--selenium_headless', type=click.BOOL, default=True, help="run selenium in the headless mode")
@click.option('--selenium_min_wait', type=click.INT, default=15, help="Min selenium timeout")
@click.option('--selenium_max_wait', type=click.INT, default=60, help="Max selenium timeout")
@click.option('--unpaywall_email', type=click.STRING, default="antonkulaga@gmail.com", help="Unpaywall email to use for pdf resolution")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value,
              help="logging level")
def download_and_parse_command(doi: str, folder: str, selenium_on_fail: bool, scihub_on_fail: bool, parser: str,
                               subfolder: bool, do_not_reparse: bool,
                               cleaning: bool, mode: str, strategy: str, infer_tables: bool,
                               include_page_breaks: bool, recreate_parent: bool,
                               selenium_headless: bool, selenium_min_wait: int, selenium_max_wait: int,
                               unpaywall_email: Optional[str],
                               log_level: str):
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
    results = try_download_and_parse(doi,
                                     Path(folder),
                                     selenium_on_fail,
                                     scihub_on_fail,
                                     PDFParser[parser],
                                     subfolder,
                                     do_not_reparse,
                                     cleaning,
                                     mode,
                                     strategy,
                                     infer_tables,
                                     include_page_breaks,
                                     recreate_parent,
                                     selenium_headless,
                                     selenium_min_wait,
                                     selenium_max_wait,
                                     unpaywall_email,
                                     logger)
    results.on_success(lambda r: logger.info(f"Results of {doi} parsing to {r.parsed}:"))
    results.on_failure(lambda e: logger.error(str(e)))
    return results


@app.command("parse_paper")
@click.option('--paper', type=click.Path(exists=True), help="paper pdf to parse")
@click.option('--destination', type=click.STRING, default=".", help="destination folder")
@click.option('--parser', type=click.Choice([loader.value for loader in PDFParser]),
              default=PDFParser.unstructured.value, help="pdf parser to choose from, unstructured by default")
@click.option('--recreate_parent', type=click.BOOL, default=False,
              help="if parent folder should be recreated in the new destination")
@click.option('--subfolder', type=click.BOOL, default=True, help="if it should create a folder per paper")
@click.option('--do_not_reparse', type=click.BOOL, default=True, help="if we should avoid reparsing")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single",
              help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="fast",
              help="parsing strategy to be used, auto by default, unstructured parser specific")
@click.option('--infer_tables', type=click.BOOL, default=True,
              help="if the table structure should be inferred, unstructured parser specific")
@click.option('--include_page_breaks', type=click.BOOL, default=False,
              help="if page breaks should be included, unstructured parser specific")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value,
              help="logging level")
def parse_paper_command(paper: str, destination: str, parser: str, recreate_parent: bool, subfolder: bool,
                        do_not_reparse: bool, mode: str, strategy: str, infer_tables: bool, include_page_breaks: bool,
                        log_level: str):
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
    paper_file = Path(paper)
    destination_folder = Path(destination)
    logger.info(
        f"parsing paper {paper} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_paper(paper_file, None, PDFParser[parser], recreate_parent, True, subfolder, do_not_reparse, mode,
                       strategy, infer_tables, include_page_breaks)


@app.command("parse_folder")
@click.option('--folder', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--destination', type=click.STRING, default=None, help="destination folder")
@click.option('--parser', type=click.Choice([loader.value for loader in PDFParser]),
              default=PDFParser.unstructured.value, help="pdf parser to choose from, unstructured by default")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single",
              help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="fast",
              help="parsing strategy to be used, auto by default")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included")
@click.option('--cores', '-t', type=int, default=None, help='Number of cores to use')
@click.option('--recreate_parent', type=click.BOOL, default=False,
              help="if parent folder should be recreated in the new destination")
@click.option('--cleaning', type=click.BOOL, default=True, help="if we should use basic cleaning for the text")
@click.option('--subfolder', type=click.BOOL, default=True, help="if it should create a folder per paper")
@click.option('--do_not_reparse', type=click.BOOL, default=True, help="if we should avoid reparsing")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value,
              help="logging level")
def parse_folder_command(folder: str, destination: str, parser: str, mode: str, strategy: str, infer_tables: bool,
                         include_page_breaks: bool, cores: Optional[int],
                         recreate_parent: bool, cleaning: bool,
                         subfolder: bool, do_not_reparse: bool, log_level: str):
    from loguru import logger
    if log_level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=log_level.upper())
    parse_folder = Path(folder)
    destination_folder = Path(destination) if destination is not None else None
    logger.info(
        f"parsing paper {folder} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_papers(parse_folder, destination_folder, PDFParser[parser], recreate_parent, cores, subfolder,
                        do_not_reparse, cleaning, mode, strategy, infer_tables, include_page_breaks, logger)


if __name__ == '__main__':
    app()
