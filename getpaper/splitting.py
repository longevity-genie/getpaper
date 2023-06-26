from abc import ABC
from typing import List

import tiktoken
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TextSplitter
from pycomfort.files import *
from copy import deepcopy


def papers_to_documents(folder: Path, suffix: str = ""):
    txt = traverse(folder, lambda p: "txt" in p.suffix)
    texts = [t for t in txt if suffix in t.name] if suffix != "" else txt
    docs: List[Document] = []
    for t in texts:
        doc = paper_to_document(t)
        if doc is not None:
            docs.append(doc)
    return docs


def paper_to_document(paper: Path, min_tokens: int = 200) -> Optional[Document]:
    """
    Turns paper into document, assumes the folder/paper_name is DOI
    :param paper:
    :param min_tokens:
    :return:
    """
    doi = f"http://doi.org/{paper.parent.name}/{paper.stem}"
    text = paper.read_text(encoding="utf-8")
    if len(text) < min_tokens:
        print("TOO SHORT TEXT")
        return None
    else:
        return Document(
            page_content=text,
            metadata={"source": doi, "doi": doi}
        )


class SourceTextSplitter(RecursiveCharacterTextSplitter, ABC):
    """
    Class that insludes dois and paging into metadata
    """

    @property
    def chunk_size(self):
        return self._chunk_size

    def create_documents(
            self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            meta = _metadatas[i]
            source: Optional[str] = meta["source"] if "source" in meta else None
            for j, chunk in enumerate(self.split_text(text)):
                new_meta = deepcopy(meta)
                if source is not None:
                    num = str(j)
                    new_meta["source"] = source + "#" + num
                    if "doi" not in new_meta:
                        new_meta["doi"] = source
                    if "split" not in new_meta:
                        new_meta["split"] = num
                new_doc = Document(page_content=chunk, metadata=new_meta)
                documents.append(new_doc)
        return documents


class OpenAISplitter(SourceTextSplitter, ABC):

    def __init__(self, tokens: int = 2000,
                 tokens_overlap: int = 0,
                 model: str = "gpt-3.5-turbo-16k",
                 keep_separator: bool = False,
                 add_start_index: bool = False
                 ):

        def length_function(text: str) -> int:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))

        super().__init__(chunk_size=tokens, chunk_overlap=tokens_overlap,
                   length_function=length_function,
                   keep_separator=keep_separator,
                   add_start_index=add_start_index)
