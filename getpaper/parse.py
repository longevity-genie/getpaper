#!/usr/bin/env python3
from enum import Enum
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, List

import click
import pynction.monads.try_monad
import tiktoken
from click import Context
from functional import seq
from langchain.document_loaders import UnstructuredPDFLoader, PDFMinerLoader, PyPDFLoader, PyMuPDFLoader
from langchain.schema import Document
from pycomfort.files import traverse
from pynction import Try
from loguru import logger
import sys

def _configure_logger(log_level: str):
    # yes, it is duplicate but it is nice to avoid cross-module dependencies here
    if log_level.upper() != "NONE":
        logger.add(sys.stdout, level=log_level.upper())

class PDFParser(Enum):
    unstructured = "unstructured"
    pdf_miner = "pdf_miner"
    py_pdf = "py_pdf"
    py_mu_pdf = "py_mu_pdf"


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
    "gpt-4-32k": None, #32,768 tokens
    "gpt-4": 0.03, #8,192 tokens
    "gpt-3.5-turbo-16k": None, #16,384 tokens,
    "gpt-3.5-turbo": None #4,096 tokens
}

def clean_text(text: str) -> str:
    from unstructured.cleaners.core import clean, group_broken_paragraphs, replace_unicode_quotes
    return clean(group_broken_paragraphs(replace_unicode_quotes(text)))


def parse_paper(paper: Path, folder: Optional[Path] = None, parser: PDFParser = PDFParser.unstructured,
                mode: str = "single", strategy: str = "auto",
                pdf_infer_table_structure: bool = True,
                include_page_breaks: bool = False,
                recreate_parent: bool = False,
                cleaning: bool = True
                ) -> List[Path]:
    """
    Parses the paper using Unstructured paper parser
    :param paper:
    :param folder:
    :param mode: can be single or paged
    :param recreate_parent: can be useful if we grouped papers by subfolders (for example for dois)
    :return:
    """
    if parser is None or parser == PDFParser.unstructured:
        loader = init_unstructured_loader(open(str(paper), "rb"), include_page_breaks, mode, pdf_infer_table_structure, strategy)
    elif parser == PDFParser.pdf_miner:
        loader = PDFMinerLoader(str(paper))
    elif parser == PDFParser.py_mu_pdf:
        loader = PyMuPDFLoader(str(paper))
    elif parser == PDFParser.py_pdf:
        loader = PyPDFLoader(str(paper))
    else:
        loader = init_unstructured_loader(open(str(paper), "rb"), include_page_breaks, mode, pdf_infer_table_structure, strategy)
    where = paper.parent if folder is None else folder / paper.parent.name if recreate_parent else folder
    where.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = loader.load()


    if len(docs) ==1:
        name = f"{paper.stem}.txt"
        f = where / name
        logger.info(f"writing {f}")
        text = clean_text(docs[0].page_content) if cleaning else docs[0].page_content
        f.write_text(text)
        return [f]
    else:
        acc = []
        for i, doc in enumerate(docs):
            name = f"{paper.stem}_{i}.txt"
            f = (where / name)
            logger.info(f"writing {f}")
            text = clean_text(doc.page_content) if cleaning else doc.page_content
            f.write_text(text)
            acc.append(f)
        return acc


def init_unstructured_loader(bin_file, include_page_breaks, mode, pdf_infer_table_structure, strategy: str):
    return UnstructuredPDFLoader(file_path=None, file=bin_file, mode=mode,
                                 pdf_infer_table_structure=pdf_infer_table_structure,
                                 strategy=strategy, include_page_breaks=include_page_breaks)


def try_parse_paper(paper: Path, folder: Optional[Path] = None,
                    parser: PDFParser = PDFParser.unstructured,
                    mode: str = "single", strategy: str = "auto",
                    pdf_infer_table_structure: bool = True,
                    include_page_breaks: bool = False, recreate_parent: bool = False, cleaning: bool = True) -> Try[List[Path]]:
    return Try.of(lambda: parse_paper(paper, folder, parser, mode, strategy, pdf_infer_table_structure, include_page_breaks, recreate_parent, cleaning))


def parse_papers(parse_folder: Path, destination: Optional[Path] = None,
                 parser: PDFParser = PDFParser.unstructured,
                 mode: str = "single", strategy: str = "auto",
                 pdf_infer_table_structure: bool = True,
                 include_page_breaks: bool = False, recreate_parent: bool = False, cores: Optional[int] = None, cleaning: bool = True):
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
    papers: list[Path] = traverse(parse_folder, lambda p: "pdf" in p.suffix)
    logger.info(f"indexing {len(papers)} papers")
    parsed = []
    errors = []

    cores = cpu_count() if cores is None else min(cpu_count(), cores)
    with Pool(cores) as p:
        parse_func = partial(try_parse_paper, folder=destination, parser=parser, mode=mode, strategy=strategy,
                             pdf_infer_table_structure=pdf_infer_table_structure,
                             include_page_breaks=include_page_breaks, recreate_parent = recreate_parent, cleaning = cleaning)
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
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

@app.command("count_tokens")
@click.option('--path', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--model', default='gpt-3.5-turbo-16k', help='model to use, gpt-3.5-turbo-16k by default')
@click.option("--suffix", default=".txt", help="suffix in the files to evaluate, .txt by default")
@click.option("--price", type=click.FLOAT, default=0.0001, help = "price for 1K tokens")
@click.option('--log_level', type=click.Choice(["NONE", "DEBUG", "INFO", "ERROR", "WARNING", "DEBUG", "TRACE"], case_sensitive=False), default="debug", help="logging level")
def count_tokens_command(path: Path, model: str, suffix: str, price: float, log_level: str):
    _configure_logger(log_level)
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
        logger.info(f"PER PAPER: \n average tokens {num_avg} , cost {money_avg}\n max tokens = {num_max} , max cost = {money_max}")
        return num, money
    else:
        content = path.read_text(encoding="utf-8")
        return num_tokens_openai(content, model)

@app.command("parse_paper")
@click.option('--paper', type=click.Path(exists=True), help="paper pdf to parse")
@click.option('--destination', type=click.STRING, default=".", help="destination folder")
@click.option('--parser', type=click.Choice([loader.value for loader in PDFParser]), default=PDFParser.unstructured.value, help="pdf parser to choose from, unstructured by default")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single", help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="fast", help="parsing strategy to be used, auto by default, unstructured parser specific")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred, unstructured parser specific")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included, unstructured parser specific")
@click.option('--recreate_parent', type=click.BOOL, default=False, help="if parent folder should be recreated in the new destination")
@click.option('--log_level', type=click.Choice(["NONE", "DEBUG", "INFO", "ERROR", "WARNING", "DEBUG", "TRACE"], case_sensitive=False), default="debug", help="logging level")
def parse_paper_command(paper: str, destination: str, parser: str, mode: str, strategy: str, infer_tables: bool, include_page_breaks: bool, recreate_parent: bool, log_level: str):
    _configure_logger(log_level)
    paper_file = Path(paper)
    destination_folder = Path(destination)
    logger.info(f"parsing paper {paper} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_paper(paper_file, None, PDFParser[parser], mode, strategy, infer_tables, include_page_breaks, recreate_parent)


@app.command("parse_folder")
@click.option('--folder', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--destination', type=click.STRING, default=None, help="destination folder")
@click.option('--parser', type=click.Choice([loader.value for loader in PDFParser]), default=PDFParser.unstructured.value, help="pdf parser to choose from, unstructured by default")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single", help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="fast", help="parsing strategy to be used, auto by default")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included")
@click.option('--cores', '-t', type=int, default=None, help='Number of cores to use')
@click.option('--recreate_parent', type=click.BOOL, default=False, help="if parent folder should be recreated in the new destination")
@click.option('--cleaning', type=click.BOOL, default=True, help="if we should use basic cleaning for the text")
@click.option('--log_level', type=click.Choice(["NONE", "DEBUG", "INFO", "ERROR", "WARNING", "DEBUG", "TRACE"], case_sensitive=False), default="debug", help="logging level")
def parse_folder_command(folder: str, destination: str, parser: str, mode: str, strategy: str, infer_tables: bool,
                         include_page_breaks: bool, cores: Optional[int], recreate_parent: bool, cleaning: bool, log_level: str):
    _configure_logger(log_level)
    parse_folder = Path(folder)
    destination_folder = Path(destination) if destination is not None else None
    logger.info(f"parsing paper {folder} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_papers(parse_folder, destination_folder, PDFParser[parser], mode, strategy, infer_tables, include_page_breaks, recreate_parent, cores = cores, cleaning=cleaning)


if __name__ == '__main__':
    app()