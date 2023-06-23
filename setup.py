from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.7'
DESCRIPTION = 'getpaper - papers download made easy!'
LONG_DESCRIPTION = 'A package with python functions for downloading papers'

# Setting up
setup(
    name="getpaper",
    version=VERSION,
    author="antonkulaga (Anton Kulaga)",
    author_email="<antonkulaga@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pyfunctional', 'more-itertools', 'click', 'python-dotenv', 'tiktoken', 'pynction',
                      'unstructured', 'unstructured-inference', 'unstructured[local-inference]', 'unstructured.PaddleOCR',
                      'scidownl', 'langchain', 'openai', 'chromadb', 'Deprecated'],
    keywords=['python', 'utils', 'files', 'papers', 'download'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
     "console_scripts": [
         "download=getpaper.download:app",
         "parse=getpaper.parse:app",
         "index=getpaper.index:app"
     ]
    }
)
