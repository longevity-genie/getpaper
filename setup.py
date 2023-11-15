from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.3.7'
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
    install_requires=['pycomfort>=0.0.14', 'click',
                      'unstructured>=0.10.25', 'unstructured-inference>=0.7.9',
                      'unstructured[local-inference]', 'unstructured.PaddleOCR>=2.6.1.3',
                      'scidownl>=1.0.2', 'Deprecated', 'semanticscholar>=0.5.0', 'pdfminer.six', 'langchain>=0.0.333',
                      'PyMuPDF>=1.22.3', 'pdfplumber>=0.10.3'],
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
         "parse=getpaper.parse:app"
     ]
    }
)
