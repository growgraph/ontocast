"""Test API client for OntoCast.

This module provides a simple command-line client for testing the OntoCast API.
It can send requests to the API server with either a default payload, a JSON file,
or a PDF file.

The client supports:
- Custom server URLs
- JSON or PDF file uploads
- Default test payload for Apple 10-Q document
- Response formatting and display

Example:
    # Send default test payload
    python test_api.py --url http://localhost:8999

    # Send a JSON file (as multipart/form-data)
    python test_api.py --url http://localhost:8999 --file sample.json

    # Send a PDF file
    python test_api.py --url http://localhost:8999 --file document.pdf
"""

import io
import json
import pathlib

import click
import requests


@click.command()
@click.option(
    "--url",
    required=True,
    help="Base URL for the server (e.g. http://localhost:8999)",
)
@click.option(
    "--file",
    type=str,
    default=None,
    help="Path to JSON or PDF file to upload (supports ~ expansion)",
)
def main(url, file):
    """Send a test request to the OntoCast API server.

    This function sends a POST request to the /process endpoint with either:
    - A file upload (JSON or PDF) as multipart/form-data
    - A JSON text payload as application/json
    - A default test payload if no file or json-text is provided

    Args:
        url: The base URL of the API server (e.g. http://localhost:8999).
        file: Optional path to a JSON or PDF file to upload.

    Example:
        >>> main("http://localhost:8999", None, None)
        # Sends default Apple 10-Q payload

        >>> main("http://localhost:8999", pathlib.Path("document.pdf"), None)
        # Sends PDF file as multipart/form-data

        >>> main("http://localhost:8999", None, '{"text": "Hello"}')
        # Sends JSON text payload
    """
    if not url.endswith("/process"):
        url = f"{url.rstrip('/')}/process"

    if file:
        # Expand ~ and convert to Path
        file_path = pathlib.Path(file).expanduser()
        if not file_path.exists():
            raise click.BadParameter(
                f"File does not exist: {file_path}",
                param_hint="--file",
            )

        file_ext = file_path.suffix.lower()
        if file_ext not in [".json", ".pdf"]:
            raise click.BadParameter(
                f"File must be .json or .pdf, got {file_ext}",
                param_hint="--file",
            )

        print(f"POSTing file '{file_path.name}' to: {url}")
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Determine MIME type
        mime_type = "application/pdf" if file_ext == ".pdf" else "application/json"
        # Use BytesIO to create a file-like object for requests
        file_obj = io.BytesIO(file_content)
        files = {"file": (file_path.name, file_obj, mime_type)}
        r = requests.post(url, files=files)
    else:
        # Default test payload
        payload = {
            "text": (
                "## UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
                "Washington, D.C. 20549 ## FORM 10-Q\n\n"
                "<!-- image -->\n\n"
                "(Mark One)\n\n"
                "☒ QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\n\n"
                "For the quarterly period ended April 1, 2023\n\n"
                "or\n\n"
                "☐ TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\n\n"
                "For the transition period from              to             .\n\n"
                "Commission File Number: 001-36743 ## Apple Inc.\n\n"
                "(Exact name of Registrant as specified in its charter)\n\n"
                "California\n\n"
                "94-2404110\n\n"
                "(State or other jurisdiction of incorporation or organization)\n\n"
                "(I.R.S. Employer Identification No.)\n\n"
                "One Apple Park Way Cupertino, California\n\n"
                "95014\n\n"
                "(Address of principal executive offices)\n\n"
                "(Zip Code) ## (408) 996-1010\n\n"
                "(Registrant's telephone number, including area code)\n\n"
                "Securities registered pursuant to Section 12(b) of the Act:\n\n"
                "Title of each class\n\n"
                "Trading symbol(s)\n\n"
                "Name of each exchange on which registered\n\n"
                "Common Stock, $0.00001 par value per share\n\n"
                "AAPL\n\n"
                "The Nasdaq Stock Market LLC\n\n"
                "15,728,702,000 shares of common stock were issued and outstanding as of April 21, 2023. "
                "## Apple Inc. ## Form 10-Q ## For the Fiscal Quarter Ended April 1, 2023"
            ),
        }
        print(f"POSTing default test payload to: {url}")
        r = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )

    print(f"Status: {r.status_code}")
    print("Response:")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


if __name__ == "__main__":
    main()
