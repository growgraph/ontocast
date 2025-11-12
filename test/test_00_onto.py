import asyncio

from ontocast.onto.ontology import OntologyProperties
from ontocast.toolbox import render_ontology_summary


def test_extract_metadata(test_ontology, llm_tool):
    # Clear the fields that should be extracted by LLM to test the extraction
    test_ontology.title = None
    test_ontology.description = None
    test_ontology.ontology_id = None

    summary = asyncio.run(render_ontology_summary(test_ontology, llm_tool))

    # Validate output
    assert isinstance(summary, OntologyProperties)
    assert summary.title is not None, "title should not be None"
    assert summary.description is not None, "description should not be None"
    # Title should contain "test" (may be "test_onto", "test ontology", etc.)
    assert "test" in summary.title.lower(), (
        f"Title '{summary.title}' should contain 'test'"
    )
    # Description should contain "test"
    assert "test" in summary.description.lower(), "Description should contain 'test'"
