# getpaper
Paper downloader

# getting started

Install the library with:
```bash
pip install getpaper
```

# Usage
## Downloading papers

After the installation you can either import the library into your python code or you can use the console scripts, for example:
```bash
download download download_pubmed --pubmed 22266545 --folder papers --name pmid
```
Downloads the paper with pubmed id into the folder 'papers' and uses the pubmed id as name
```bash
download download download_doi --doi 10.1519/JSC.0b013e318225bbae --folder papers
```
Downloads the paper with DOI into the folder papers, as --name is not specified doi is used as name

## Parsing the papers

You can parse the downloaded papers with the unstructure library. For example if the papers are in the folder test, you can run:
```bash
getpaper/parse.py parse_folder --folder /home/antonkulaga/sources/getpaper/test
```
You can also parse papers on a per file basis, for example:
```bash
getpaper/parse.py parse_paper --paper /home/antonkulaga/sources/getpaper/test/22266545.pdf
```

# Additional requirements

Detectron2 is required for using models from the layoutparser model zoo but is not automatically installed with this package. 
For MacOS and Linux, build from source with:

pip install 'git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2'