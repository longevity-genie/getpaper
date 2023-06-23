# getpaper
Paper downloader

# getting started

Install the library with:
```bash
pip install getpaper
```

On linux systems you sometimes need to check that build essentials are installed:
```bash
sudo apt install build-essential.
```
It is also recommended to use micromamba, conda, anaconda or other environments to avoid bloating system python with too many dependencies.

# Usage
## Downloading papers

After the installation you can either import the library into your python code or you can use the console scripts.

If you install from pip calling download will mean calling getpaper/download.py , for parse - getpaper/parse.py , for index - getpaper/index.py

```bash
download download download_pubmed --pubmed 22266545 --folder papers --name pmid
```
Downloads the paper with pubmed id into the folder 'papers' and uses the pubmed id as name
```bash
download download download_doi --doi 10.1519/JSC.0b013e318225bbae --folder papers
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

## Parsing the papers

You can parse the downloaded papers with the unstructured library. For example if the papers are in the folder test, you can run:
```bash
getpaper/parse.py parse_folder --folder data/output/test/papers --threads 5
```
You can also parse papers on a per-file basis, for example:
```bash
getpaper/parse.py parse_paper --paper data/output/test/papers/10.3390/ijms22031073.pdf
```

## Count tokens

To evaluate how much you want to split texts and how much embeddings will cost you it is useful to compute token number:

```bash
getpaper/parse.py count_tokens --path /home/antonkulaga/sources/non-animal-models/data/inputs/datasets
```

## Indexing papers

We also provide features to index the papers with openai or lambda embeddings and save them in chromadb vector store.
For openai embeddings to work you have to create .env file and specify your openai key there, see .env.template as example
For example if you have your papers inside data/output/test/papers folder, and you want to make a ChromaDB index at data/output/test/index you can do it by:
```bash
python getpaper/index.py index_papers --papers data/output/test/papers --folder data/output/test/index --collection mypapers --chunk_size 6000
```

# Examples

You can run examples.py to see usage examples

# Additional requirements

Detectron2 is required for using models from the layoutparser model zoo but is not automatically installed with this package. 
For macOS and Linux, build from source with:

pip install 'git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2'