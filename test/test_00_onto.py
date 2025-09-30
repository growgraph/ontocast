from ontocast.onto.ontology import OntologyProperties
from ontocast.toolbox import render_ontology_summary


def test_extract_metadata(test_ontology, llm_tool):
    summary = render_ontology_summary(test_ontology, llm_tool)

    # Validate output
    assert isinstance(summary, OntologyProperties)
    assert "test" in summary.title.lower() and "ontology" in summary.title.lower()
    assert "test" in summary.description.lower()
