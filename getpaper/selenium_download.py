#!/usr/bin/env python3

from __future__ import annotations

import sys

import loguru
from click import Context

try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.common.exceptions import TimeoutException
    # Use optional_dependency in your code
except ImportError:
    # Handle the case where the optional_dependency is not installed
    print("Optional dependency is not installed. Some features may be unavailable.")

from pathlib import Path
import time
import os
import click
from typing import Optional


def download_pdf_selenium(url: str, download_dir: Path, headless: bool=True, min_wait_time: int=8, max_wait_time: int=60, final_path: Optional[Path] = None, logger: Optional["loguru.Logger"] = None) -> Path:
    if logger is None:
        logger = loguru.logger

    download_dir.mkdir(parents=True, exist_ok=True)
    absolute_download_dir = str(download_dir.resolve())

    options = FirefoxOptions()
    if headless:
        options.add_argument("--headless")

    # Set preferences to disable the built-in PDF viewer and auto-download PDFs
    options.set_preference("pdfjs.disabled", True)
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", absolute_download_dir)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")

    logger.info(f"will use selenium firefox driver to download {url}")
    driver = webdriver.Firefox(options=options)

    try:
        driver.set_page_load_timeout(min_wait_time)
        driver.get(url)
    except TimeoutException:
        logger.info(f"min download time of the {url} timed out but continuing with the selenium download script...")

    try:
        start_time = time.time()
        downloaded_file = None

        while time.time() - start_time < max_wait_time:
            pdf_files = [f for f in download_dir.iterdir() if f.is_file() and f.suffix == '.pdf']

            if pdf_files:
                new_file = max(pdf_files, key=os.path.getctime)
                if downloaded_file != new_file or os.path.getsize(new_file) > 0:
                    downloaded_file = new_file
                    break  # Exit the loop as soon as a new file is found and is not empty

            time.sleep(1)  # Polling interval

        if not downloaded_file:
            raise Exception("Download timed out or no new file detected.")
        return downloaded_file if final_path is None else downloaded_file.rename(final_path)
    except Exception as e:
        raise e
    finally:
        driver.quit()

"""
# Example usage
url = "https://www.cell.com/article/S2213671122004581/pdf"
output_directory = Path("/home/antonkulaga/sources/getpaper/data/output/test/papers")

try:
    downloaded_file_path = download_pdf_selenium(url, output_directory, headless=True)
    print(f"Downloaded file: {downloaded_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
"""

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    if ctx.invoked_subcommand is None:
        click.echo('Running the default command...')
        selenium_download_command()
    pass

@app.command("selenium_download")
@click.option('--url', type=click.STRING, required=True, help="url to download")
@click.option('--destination', type=click.Path(exists=True), default="data/output/test/papers", help="folder to parse papers in")
@click.option('--headless', type=click.BOOL, default=True, help="if to run in a headless mode")
@click.option('--min_wait', type=int, default=12, help='Min waiting time')
@click.option('--max_wait', type=int, default=60, help='Max waiting time')
@click.option('--final_path', type=click.Path(), default = None, help="final path")
def selenium_download_command(url: str, destination: str, headless: bool, min_wait: int, max_wait: int, final_path: Optional[Path]):
    logger = loguru.logger
    logger.add(sys.stdout)
    logger.info(f"download {url} to {destination} in headless={headless} mode with min_wait={min_wait} and max_wait={max_wait}")
    result = download_pdf_selenium(url, Path(destination), headless, min_wait, max_wait, final_path=final_path, logger=logger)
    loguru.logger.info(f"downloaded to {result}")
    return result

if __name__ == '__main__':
    app()