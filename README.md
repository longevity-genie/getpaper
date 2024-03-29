# getpaper
Paper downloader

WARNING: temporally broken with last update

# getting started

Install the library with:
```bash
pip install getpaper
```
If you want to edit getpaper repository consider installing it locally:
```
pip install -e .
```

On linux systems you sometimes need to check that build essentials are installed:
```bash
sudo apt install build-essential.
```
It is also recommended to use micromamba, conda, anaconda or other environments to avoid bloating system python with too many dependencies.

# Usage
## Downloading papers

After the installation you can either import the library into your python code or you can use the console scripts.

If you install from pip calling _download_ will mean calling getpaper/download.py , for _parse_ - getpaper/parse.py , for _index_ - getpaper/index.py

```bash
download download_pubmed --pubmed 22266545 --folder "data/output/test/papers" --name pmid
```
Downloads the paper with pubmed id into the folder 'papers' and uses the pubmed id as name
```bash
download download_doi --doi 10.1038/s41597-020-00710-z --folder "data/output/test/papers"
```
Downloads the paper with DOI into the folder papers, as --name is not specified doi is used as name

It is also possible to download many papers in parallel with download_papers(dois: List[str], destination: Path, threads: int) function, for example:
```python
from pathlib import Path
from typing import List
from getpaper.download import download_papers
dois: List[str] = ["10.3390/ijms22031073", "10.1038/s41597-020-00710-z", "wrong"]
destination: Path = Path("./data/output/test/papers").absolute().resolve()
threads: int = 5
results = download_papers(dois, destination, threads)
successful = results[0]
failed = results[1]
```
Here results will be OrderedDict[str, Path] with successfully downloaded doi->paper_path and List[str] with failed dois, in current example:
```
(OrderedDict([('10.3390/ijms22031073',
               PosixPath('/home/antonkulaga/sources/getpaper/notebooks/data/output/test/papers/10.3390/ijms22031073.pdf')),
              ('10.1038/s41597-020-00710-z',
               PosixPath('/home/antonkulaga/sources/getpaper/notebooks/data/output/test/papers/10.1038/s41597-020-00710-z.pdf'))]),
 ['wrong'])
```
Same function can be called from the command line:
```bash
download download_papers --dois "10.3390/ijms22031073" --dois "10.1038/s41597-020-00710-z" --dois "wrong" --folder "data/output/test/papers" --threads 5
```
You can also call download.py script directly:
```bash
python getpaper/download.py download_papers --dois "10.3390/ijms22031073" --dois "10.1038/s41597-020-00710-z" --dois "wrong" --folder "data/output/test/papers" --threads 5
```

## Parsing the papers

You can parse the downloaded papers with the unstructured library. For example if the papers are in the folder test, you can run:
```bash
getpaper/parse.py parse_folder --folder data/output/test/papers --cores 5
```
You can also switch between different PDF parsers:
```
getpaper/parse.py parse_folder --folder data/output/test/papers --parser pdf_miner --cores 5
```
You can also parse papers on a per-file basis, for example:
```bash
getpaper/parse.py parse_paper --paper data/output/test/papers/10.3390/ijms22031073.pdf
```

## Combining parsing and downloading

```bash
getpaper/parse.py download_and_parse --doi 10.1038/s41597-020-00710-z
```

## Count tokens

To evaluate how much you want to split texts and how much embeddings will cost you it is useful to compute token number:

```bash
getpaper/parse.py count_tokens --path /home/antonkulaga/sources/non-animal-models/data/inputs/datasets
```
# Examples

You can run examples.py to see usage examples

# Additional requirements

index.py has local dependencies on other modules, for this reason if you are running it inside getpaper project folder consider having it installed locally:
```bash
pip install -e .
```

Detectron2 is required for using models from the layoutparser model zoo but is not automatically installed with this package. 
For macOS and Linux, build from source with:

pip install 'git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2'

# Note

Since 0.3.0 version all indexing features were moved to indexpaper library