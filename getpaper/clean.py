#!/usr/bin/env python3


from pathlib import Path
from typing import Optional
import click
from click import Context

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from pycomfort.config import load_environment_keys

from langchain.schema import Document

proofread_template_text: str = """
You are a proof-reader in a biomedical academic journal and your role is to proofread the text enclosed in triple quotes and separate text, authors and references sections. 
Your tasks are:
1  Correct any words that are spelled incorrectly while preserving the intended meaning of the sentence and ensuring that identifiers of gene products and other biological enteties are unchanged.
2  Split incorrectly combined words, ensuring they make sense within the sentence context.
3  If a letter, word or phrase is duplicated unnecessarily, eliminate the redundancy while maintaining the coherence of the sentence.
4  Correct fundamental grammatical errors like incorrect verb tenses, subject-verb agreement issues, and wrong prepositions.
5  Merge words that are incorrectly split across two lines or with whitespace, remove unneeded hyphens.
Remember, the goal is to improve readability and coherence, preserving the original intent and context of the text to the maximum extent possible.
{format_instructions}
The text to proofread is: ```{text}```
"""
authors_schema = ResponseSchema(name="authors", description="If author section is present, you should separate it into authors field and clean it from numeric references, otherwise put empty quotes")
processed_text_schema = ResponseSchema(name="processed_text", description="Output processed proofread text excluding authors, affiliations, tables and literature sections if they are present there")
references_schema = ResponseSchema(name="references", description="Output literature references section, if it is not found output empty string")
proofread_output_parser = StructuredOutputParser.from_response_schemas([authors_schema, processed_text_schema, references_schema])
format_instructions = proofread_output_parser.get_format_instructions()
proofread_prompt = PromptTemplate.from_template(template=proofread_template_text, output_parser=proofread_output_parser)

def make_chain():
    openai_key = load_environment_keys()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    return LLMChain(
        llm=llm,
        prompt=proofread_prompt
    )


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

def proofread(text: str, chain: Optional[LLMChain] = None, doi: Optional[str] = None):
    proofread_chain = make_chain() if chain is None else chain
    dic = proofread_chain.predict(text = text, format_instructions = format_instructions)
    if doi is not None:
        dic["doi"] = doi
    return dic

def proofread_document(doc: Document, proofread_chain: LLMChain):
    text = doc.page_content
    results = proofread_output_parser.parse(proofread_chain.predict(text = text, format_instructions = format_instructions))
    content = results["processed_text"]
    print(f"==========INPUT IS ===============\n{text}, \n--------OUTPUT IS--------------: \n{str(results)}")
    meta = doc.metadata.copy()
    meta["authors"] = results["authors"]
    meta["references"] = results["references"]
    return Document(
        page_content=content,
        metadata=meta
    )

"""
def clean_paper_as_document(paper: Path, tokens: int = 12000) -> Document:
    splitter = OpenAISplitter(tokens)
    doc = paper_to_document(paper)
    proofread_chain = make_chain()
    docs = splitter.split_documents([doc])
    new_docs = [proofread_document(d, proofread_chain) for d in docs]
    doc = new_docs[0]
    if len(new_docs) > 1:
        for d in new_docs:
            doc.page_content = doc.page_content + d.page_content
            doc.metadata["authors"] = doc.metadata["authors"] + d.metadata["authors"]
            doc.metadata["references"] = doc.metadata["references"] + d.metadata["references"]
    return doc

def clean_paper(paper: Path, tokens: int = 10000, destination: Optional[Path] = None):
    doc = clean_paper_as_document(paper, tokens)
    where = paper.parent / paper.name.replace(".txt", "_proofread.txt") if destination is None else destination
    where.write_text(doc.page_content)
    print(f"cleaned result written to {where}")
    return where
"""


@app.command("clean_paper")
@click.option('--paper', type=click.Path(exists=True), help="paper pdf to parse")
@click.option('--destination', type=click.STRING, default=".", help="destination folder")
@click.option('--recreate_parent', type=click.BOOL, default=True, help="if parent folder should be recreated in the new destination")
def clean_paper_command(paper: str, destination: str, recreate_parent: bool):
    paper_file = Path(paper)
    destination_folder = Path(destination)
    print(f"cleaning {destination_folder}")
    print("NOT IMPLEMENTED YET")
    pass


if __name__ == '__main__':
    app()