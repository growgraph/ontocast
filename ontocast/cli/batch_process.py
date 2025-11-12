"""Batch processing client for OntoCast.

This module provides a command-line client for batch processing multiple files
through the OntoCast API server. It supports async processing with configurable
concurrency limits.

The client supports:
- Recursive directory scanning
- File pattern matching (e.g., by extension)
- Async processing with concurrency control
- Progress tracking and error reporting
- JSON and PDF file types

Example:
    # Process all JSON files in a directory (max 3 concurrent)
    python batch_process.py --url http://localhost:8999 --path ./data --pattern "*.json" --max-concurrent 3

    # Process all PDF files recursively
    python batch_process.py --url http://localhost:8999 --path ./documents --pattern "*.pdf" --recursive
"""

import asyncio
import json
import pathlib
from typing import Optional

import click
import httpx


async def process_file(
    client: httpx.AsyncClient,
    url: str,
    file_path: pathlib.Path,
    semaphore: asyncio.Semaphore,
    results: dict,
    dataset: Optional[str] = None,
) -> None:
    """Process a single file by sending it to the OntoCast API.

    Args:
        client: httpx async client
        url: API endpoint URL
        file_path: Path to the file to process
        semaphore: Semaphore to limit concurrent requests
        results: Dictionary to store results (success/error counts)
        dataset: Optional dataset name for triple store storage
    """
    async with semaphore:
        try:
            file_ext = file_path.suffix.lower()
            mime_type = "application/pdf" if file_ext == ".pdf" else "application/json"

            with open(file_path, "rb") as f:
                file_content = f.read()

            files = {"file": (file_path.name, file_content, mime_type)}

            # Add dataset as query parameter if provided
            params = {}
            if dataset:
                params["dataset"] = dataset

            response = await client.post(url, files=files, params=params)
            status = response.status_code

            if status == 200:
                results["success"] += 1
                click.echo(f"✓ {file_path.name} - Success")
            else:
                error_text = (
                    response.text[:200] if response.text else "No error message"
                )
                results["errors"] += 1
                results["error_details"][file_path.name] = {
                    "status": status,
                    "error": error_text,
                }
                click.echo(f"✗ {file_path.name} - Error {status}")

        except Exception as e:
            results["errors"] += 1
            results["error_details"][file_path.name] = {
                "status": None,
                "error": str(e)[:200],
            }
            click.echo(f"✗ {file_path.name} - Exception: {str(e)[:100]}")


async def process_files_async(
    url: str,
    file_paths: list[pathlib.Path],
    max_concurrent: int,
    dataset: Optional[str] = None,
) -> dict:
    """Process multiple files asynchronously with concurrency control.

    Args:
        url: API endpoint URL
        file_paths: List of file paths to process
        max_concurrent: Maximum number of concurrent requests
        dataset: Optional dataset name for triple store storage

    Returns:
        Dictionary with processing results (success count, error count, details)
    """
    results = {
        "success": 0,
        "errors": 0,
        "error_details": {},
        "total": len(file_paths),
    }

    if not file_paths:
        click.echo("No files found to process.")
        return results

    semaphore = asyncio.Semaphore(max_concurrent)
    click.echo(
        f"Processing {len(file_paths)} file(s) with max {max_concurrent} concurrent requests..."
    )
    if dataset:
        click.echo(f"Using dataset: {dataset}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            process_file(client, url, file_path, semaphore, results, dataset)
            for file_path in file_paths
        ]
        await asyncio.gather(*tasks)

    return results


def find_files(
    path: pathlib.Path, pattern: Optional[str], recursive: bool
) -> list[pathlib.Path]:
    """Find files matching the given pattern.

    Args:
        path: Base path to search
        pattern: Glob pattern (e.g., "*.json", "*.pdf") or None for all files
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    if not path.exists():
        raise click.BadParameter(f"Path does not exist: {path}", param_hint="--path")

    if path.is_file():
        return [path]

    if pattern:
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
    else:
        if recursive:
            files = [f for f in path.rglob("*") if f.is_file()]
        else:
            files = [f for f in path.glob("*") if f.is_file()]

    # Filter to only JSON and PDF files
    supported_extensions = {".json", ".pdf"}
    files = [f for f in files if f.suffix.lower() in supported_extensions]

    return sorted(files)


@click.command()
@click.option(
    "--url",
    required=True,
    help="Base URL for the server (e.g. http://localhost:8999)",
)
@click.option(
    "--path",
    type=click.Path(path_type=pathlib.Path, exists=True),
    required=True,
    help="Path to file or directory to process",
)
@click.option(
    "--pattern",
    type=str,
    default=None,
    help="Glob pattern to match files (e.g., '*.json', '*.pdf'). If not provided, processes all supported files.",
)
@click.option(
    "--recursive",
    is_flag=True,
    default=True,
    help="Search for files recursively in subdirectories",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=3,
    help="Maximum number of concurrent requests (default: 3)",
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to save results summary as JSON",
)
@click.option(
    "--dataset",
    type=str,
    default=None,
    help="Dataset name for triple store storage (Fuseki only). If provided, all files will be processed into this dataset.",
)
def main(
    url: str,
    path: pathlib.Path,
    pattern: Optional[str],
    recursive: bool,
    max_concurrent: int,
    output: Optional[pathlib.Path],
    dataset: Optional[str],
):
    """Batch process files through the OntoCast API server.

    This command finds files matching the given pattern (or all supported files)
    and sends them to the OntoCast API server for processing. Files are processed
    asynchronously with a configurable concurrency limit.

    Supported file types: .json, .pdf

    Examples:
        # Process all JSON files in a directory
        batch_process.py --url http://localhost:8999 --path ./data --pattern "*.json"

        # Process all PDFs recursively with 5 concurrent requests
        batch_process.py --url http://localhost:8999 --path ./documents --pattern "*.pdf" --recursive --max-concurrent 5

        # Process files into a specific dataset
        batch_process.py --url http://localhost:8999 --path ./data --pattern "*.json" --dataset my_dataset

        # Process a single file
        batch_process.py --url http://localhost:8999 --path ./document.pdf
    """
    if not url.endswith("/process"):
        url = f"{url.rstrip('/')}/process"

    if max_concurrent < 1:
        raise click.BadParameter(
            "max-concurrent must be at least 1", param_hint="--max-concurrent"
        )

    # Expand user path
    path = path.expanduser()

    # Find files
    try:
        file_paths = find_files(path, pattern, recursive)
    except Exception as e:
        raise click.ClickException(f"Error finding files: {e}")

    if not file_paths:
        click.echo(f"No files found matching pattern '{pattern or '*.*'}' in {path}")
        return

    click.echo(f"Found {len(file_paths)} file(s) to process")
    if pattern:
        click.echo(f"Pattern: {pattern}")
    if recursive:
        click.echo("Recursive search: enabled")
    if dataset:
        click.echo(f"Dataset: {dataset}")
    click.echo(f"Max concurrent requests: {max_concurrent}")
    click.echo("")

    # Process files
    results = asyncio.run(process_files_async(url, file_paths, max_concurrent, dataset))

    # Print summary
    click.echo("")
    click.echo("=" * 60)
    click.echo("Processing Summary")
    click.echo("=" * 60)
    click.echo(f"Total files: {results['total']}")
    click.echo(f"Successful: {results['success']}")
    click.echo(f"Errors: {results['errors']}")

    if results["error_details"]:
        click.echo("")
        click.echo("Error Details:")
        for filename, details in results["error_details"].items():
            click.echo(f"  {filename}: {details['error']}")

    # Save results if output path provided
    if output:
        output = output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo("")
        click.echo(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
