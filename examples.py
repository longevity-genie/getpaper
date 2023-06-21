#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, List

import click
import requests
from click import Context
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from pycomfort.files import traverse
from pynction import Try


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    if ctx.invoked_subcommand is None:
        click.echo('Running the default command...')
    pass

def doi_download_parse(doi: str = "10.3390/ijms22031073", strategy: str = "auto"):
    print("example_download_and_parse_doi")
    from getpaper.download import try_download
    from getpaper.parse import parse_paper
    where = Path("./data/output/test").absolute().resolve()
    try_download: Try[Path] = try_download(doi, where)
    return try_download.run(lambda p: parse_paper(p.absolute().resolve(), strategy=strategy), lambda f: "it crashed :((((((")

@app.command("doi_download_parse")
@click.option('--doi', type=click.STRING, default = "10.3390/ijms22031073", help="download doi")
@click.option("--strategy", type=click.Choice(["auto", "hi_res", "fast"]), default = "auto", help="strategy used to convert the page")
def doi_download_parse_command(doi: str = "10.3390/ijms22031073", strategy: str = "auto"):
   return doi_download_parse(doi, strategy)


@app.command("doi_download_parse_index")
@click.option('--doi', type=click.STRING, default = "10.3390/ijms22031073", help="download doi")
@click.option("--strategy", type=click.Choice(["auto", "hi_res", "fast"]), default = "auto", help="strategy used to convert the page")
def doi_download_parse_index(doi: str, strategy: str = "auto"):
    return doi_download_parse(doi, strategy)

if __name__ == '__main__':
    app()