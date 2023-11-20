import requests

def get_science_direct_pdf(doi, api_key):
    base_url = "https://api.elsevier.com/content/article/doi/"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/pdf"
    }

    response = requests.get(f"{base_url}{doi}", headers=headers)

    if response.status_code == 200:
        # Assuming the response is a PDF file
        with open(f"{doi.replace('/', '_')}.pdf", "wb") as f:
            f.write(response.content)
        return f"PDF saved as {doi.replace('/', '_')}.pdf"
    else:
        return "Failed to retrieve PDF: " + response.text

# Usage
your_doi = "10.1519/JSC.0b013e318225bbae"
your_api_key = "e009797d502ddbdeedfb744abd63f5d3"  # Replace with your Elsevier API key
print(get_science_direct_pdf(your_doi, your_api_key))