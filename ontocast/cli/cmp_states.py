import pathlib

import click
from rich.console import Console
from rich.table import Table

from ontocast.onto.state import AgentState

console = Console()


def get_state_files(
    directory: pathlib.Path, pattern: str = "agent_state.onto.update*.json"
) -> list[pathlib.Path]:
    """Get all AgentState files matching the pattern in the directory."""
    return sorted(directory.glob(pattern))


def compare_states(states: list[tuple[pathlib.Path, AgentState]]) -> None:
    """Compare states and print a table with graph lengths."""
    table = Table(title="AgentState Comparison")
    table.add_column("File", style="orange1")
    table.add_column("Graph Facts", justify="right")
    table.add_column("Current Ontology", justify="right")
    table.add_column("Ontology Addendum", justify="right")
    table.add_column("Success Score", justify="right")

    # Sort rows by the last number in the filename
    sorted_rows = sorted(
        [(fp, state) for fp, state in states],
        key=lambda x: int(x[0].stem.split(".")[-1])
        if x[0].stem.split(".")[-1].isdigit()
        else 0,
    )

    for fp, state in sorted_rows:
        table.add_row(
            str(fp.stem),
            str(len(state.current_chunk.graph)),
            str(len(state.current_ontology.graph))
            if state.current_ontology is not None
            else "",
            str(len(state.ontology_addendum.graph)),
            str(state.success_score),
        )
    console.print(table)


@click.command()
@click.argument(
    "directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--pattern",
    default="agent_state.onto.update*.json",
    help="Pattern to match state files",
)
def main(directory: pathlib.Path, pattern: str):
    """Compare AgentState files in a directory."""
    state_files = get_state_files(directory, pattern)
    if not state_files:
        console.print(
            f"[red]No state files found matching "
            f"pattern '{pattern}' in {directory}[/red]"
        )
        return

    states = []
    for file_path in sorted(state_files):
        try:
            state = AgentState.load(file_path)
            states.append((file_path, state))
        except Exception as e:
            console.print(f"[red]Error loading {file_path}: {str(e)}[/red]")

    if states:
        compare_states(states)
    else:
        console.print("[red]No valid state files found[/red]")


if __name__ == "__main__":
    main()
