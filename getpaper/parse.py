#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, List, Dict

import click
import pynction.monads.try_monad
import tiktoken
from click import Context
from functional import seq
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from pycomfort.files import traverse
from multiprocessing import Pool, cpu_count
from functools import partial
from deprecated import deprecated
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pynction import Try


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


def parse_paper(paper: Path, folder: Optional[Path] = None,
                mode: str = "single", strategy: str = "auto",
                pdf_infer_table_structure: bool = True,
                include_page_breaks: bool = False, recreate_parent: bool = False
                ) -> List[Path]:
    """
    Parses the paper using Unstructured paper parser
    :param paper:
    :param folder:
    :param mode: can be single or paged
    :param recreate_parent: can be useful if we grouped papers by subfolders (for example for dois)
    :return:
    """
    bin_file = open(str(paper), "rb")
    loader = UnstructuredPDFLoader(file_path=None, file = bin_file,  mode=mode,
                                   pdf_infer_table_structure=pdf_infer_table_structure,
                                   strategy = strategy, include_page_breaks = include_page_breaks)
    where = paper.parent if folder is None else folder / paper.parent.name if recreate_parent else folder
    where.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = loader.load()
    if len(docs) ==1:
        name = f"{paper.stem}.txt"
        f = where / name
        print(f"writing {f}")
        f.write_text(docs[0].page_content)
        return [f]
    else:
        acc = []
        for i, doc in enumerate(docs):
            name = f"{paper.stem}_{i}.txt"
            f = (where / name)
            print(f"writing {f}")
            f.write_text(doc.page_content)
            acc.append(f)
        return acc

def try_parse_paper(paper: Path, folder: Optional[Path] = None,
                    mode: str = "single", strategy: str = "auto",
                    pdf_infer_table_structure: bool = True,
                    include_page_breaks: bool = False, recreate_parent: bool = False) -> Try[List[Path]]:
    return Try.of(lambda: parse_paper(paper, folder, mode, strategy, pdf_infer_table_structure, include_page_breaks, recreate_parent))


def parse_papers(parse_folder: Path, destination: Optional[Path] = None,
                 mode: str = "single", strategy: str = "auto",
                 pdf_infer_table_structure: bool = True,
                 include_page_breaks: bool = False, recreate_parent: bool = False, cores: Optional[int] = None):
    """
    Function to parse multiple papers using multiple cores.
    The function employs multiprocessing to speed up the process.

    Args:
        parse_folder (Path): The folder where the papers (PDF files) are located.
        destination (Optional[Path]): The destination folder where parsed papers will be saved.
                                      If not provided, the parsed papers are saved in their original folder.
        mode (str): The mode to parse the papers. Default is "single".
        strategy (str): The strategy to parse the papers. Default is "auto".
        pdf_infer_table_structure (bool): If True, attempts to infer table structure in PDFs. Default is True.
        include_page_breaks (bool): If True, includes page breaks in the parsed output. Default is False.
        recreate_parent (bool): If True, recreates the parent directory structure in the destination folder. Default is False.
        cores (Optional[int]): The number of cores to use. If not provided, uses all available cores.

    Returns:
        list[Path]: A list of paths to the parsed papers.
    """
    papers: list[Path] = traverse(parse_folder, lambda p: "pdf" in p.suffix)
    print(f"indexing {len(papers)} papers")
    acc = []
    errors = []

    cores = cpu_count() if cores is None else min(cpu_count(), cores)
    with Pool(cores) as p:
        parse_func = partial(try_parse_paper, folder=destination, mode=mode, strategy=strategy,
                             pdf_infer_table_structure=pdf_infer_table_structure,
                             include_page_breaks=include_page_breaks, recreate_parent = recreate_parent)
        results = p.map(parse_func, papers)
        for result in results:
            if isinstance(result, pynction.monads.try_monad.Success):
                acc = acc + result._value
            else:
                errors = errors + result

    print("papers parsing finished!")
    if len(errors) >0:
        print(f"errors discovered: {errors}")
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


@deprecated(version='0.1.6', reason="parse_papers function was rewritten with multiprocessors support")
def parse_papers_async(parse_folder: Path, destination: Optional[Path] = None,
                 mode: str = "single", strategy: str = "auto",
                 pdf_infer_table_structure: bool = True,
                 include_page_breaks: bool = False,
                 threads: int = 5):
    papers: list[Path] = traverse(parse_folder, lambda p: "pdf" in p.suffix)
    print(f"indexing {len(papers)} papers")

    async def async_parse_paper(paper):
        loop = asyncio.get_running_loop()
        where = destination if destination is not None else paper.parent
        return await loop.run_in_executor(
            executor, parse_paper, paper, where, mode, strategy, pdf_infer_table_structure, include_page_breaks
        )

    executor = ThreadPoolExecutor(max_workers=threads)

    loop = asyncio.get_event_loop()
    tasks = [async_parse_paper(paper) for paper in papers]
    parsed_papers = loop.run_until_complete(asyncio.gather(*tasks))

    print("papers parsing finished!")
    return parsed_papers


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
def count_tokens(path: Path, model: str, suffix: str, price: float):
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
        print(f"Checked {len(papers)} papers. TOTAL TOKENS = {num_sum} , COST = {money_sum}")
        print(f"PER PAPER: \n average tokens {num_avg} , cost {money_avg}\n max tokens = {num_max} , max cost = {money_max}")
        return num, money
    else:
        content = path.read_text(encoding="utf-8")
        print("checked")
        return num_tokens_openai(content, model)

@app.command("parse_paper")
@click.option('--paper', type=click.Path(exists=True), help="paper pdf to parse")
@click.option('--destination', type=click.STRING, default=".", help="destination folder")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single", help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="auto", help="parsing strategy to be used, auto by default")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included")
@click.option('--recreate_parent', type=click.BOOL, default=False, help="if parent folder should be recreated in the new destination")
def parse_paper_command(paper: str, destination: str, mode: str, strategy: str, infer_tables: bool, include_page_breaks: bool, recreate_parent: bool):
    paper_file = Path(paper)
    destination_folder = Path(destination)
    print(f"parsing paper {paper} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_paper(paper_file, None, mode, strategy, infer_tables, include_page_breaks, recreate_parent)

@app.command("parse_folder")
@click.option('--folder', type=click.Path(exists=True), help="folder to parse papers in")
@click.option('--destination', type=click.STRING, default=None, help="destination folder")
@click.option('--mode', type=click.Choice(["single", "elements", "paged"]), default="single", help="paper mode to be used")
@click.option('--strategy', type=click.Choice(["auto", "hi_res", "fast"]), default="auto", help="parsing strategy to be used, auto by default")
@click.option('--infer_tables', type=click.BOOL, default=True, help="if the table structure should be inferred")
@click.option('--include_page_breaks', type=click.BOOL, default=False, help="if page breaks should be included")
@click.option('--cores', '-t', type=int, default=None, help='Number of cores to use')
@click.option('--recreate_parent', type=click.BOOL, default=False, help="if parent folder should be recreated in the new destination")
def parse_paper_command(folder: str, destination: str, mode: str, strategy: str, infer_tables: bool, include_page_breaks: bool, cores: Optional[int], recreate_parent: bool):
    parse_folder = Path(folder)
    destination_folder = Path(destination) if destination is not None else None
    print(f"parsing paper {folder} with mode={mode} {'' if destination_folder is None else 'destination folder ' + destination}")
    return parse_papers(parse_folder, destination_folder, mode, strategy, infer_tables, include_page_breaks, recreate_parent, cores = cores)


if __name__ == '__main__':
    app()