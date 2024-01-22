import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from config import get_parent_document_retriever
import uuid


def parse_sitemap() -> list[str]:
    tree = ET.parse("sitemap.xml")
    loc_values = []
    root = tree.getroot()
    root_el = "{http://www.sitemaps.org/schemas/sitemap/0.9}"  # Define the namespace

    for url_element in root.findall(f"{root_el}url"):
        loc_element = url_element.find(f"{root_el}loc")
        if loc_element is not None:
            loc_values.append(loc_element.text)
    return loc_values


def scrape_text_from_url(url: str) -> str:
    """
    This function fetches the content of the given URL and extracts all text from it.
    :param url: URL of the webpage to scrape
    :return: A string containing all the extracted text
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        }
        # Send a request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        soup.find("div", class_="dmHeaderContainer").decompose()
        soup.find("div", class_="dmFooterContainer").decompose()
        print(soup.get_text(separator="\n", strip=True))

        # Extract and return the text
        return soup.get_text(separator="\n", strip=True)
    except requests.RequestException as e:
        return f"Error fetching the webpage: {e}"


def main():
    urls = parse_sitemap()
    parent_document_retriever = get_parent_document_retriever();
    for url in urls:
        page_content = scrape_text_from_url(url)
        documents = [
            Document(
                page_content=page_content,
                metadata={"url": url},
            )
        ]
        print(f"Adding document from url: {url}")
        parent_document_retriever.add_documents(documents)
    print("Done")



if __name__ == "__main__":
    # calling main function
    main()
