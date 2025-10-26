from ontocast.onto.enum import WorkflowNode
from ontocast.onto.model import Suggestions
from ontocast.prompt.common import (
    suggestion_concrete_template,
    suggestion_general_template,
)
from ontocast.prompt.render_facts import (
    improvement_instruction_template as facts_template,
)
from ontocast.prompt.render_ontology import (
    improvement_instruction_template as ontology_template,
)


def render_suggestions_prompt(suggestions: Suggestions, stage: WorkflowNode) -> str:
    """Generate prompt templates from the suggestions.

    Returns:
        Combined string with general and concrete templates.
        Returns empty string if both fields are empty.
    """

    # Generate general template if systemic_critique_summary is not empty
    general_template = ""
    if suggestions.systemic_critique_summary.strip():
        general_template = suggestion_general_template.format(
            general_suggestion=suggestions.systemic_critique_summary
        )

    concrete_template = ""
    if suggestions.actionable_fixes:
        # Generate concrete template if actionable_fixes is not empty
        concrete_template = suggestion_concrete_template.format(
            suggestion_str=suggestions.to_markdown()
        )

    if stage == WorkflowNode.TEXT_TO_FACTS:
        template = facts_template
    elif stage == WorkflowNode.TEXT_TO_ONTOLOGY:
        template = ontology_template
    else:
        raise ValueError(f"Stage {stage} not supported")
    if general_template or concrete_template:
        final_prompt = template.format(
            suggestions_instruction=f"\n\n{general_template}\n\n{concrete_template}"
        )
    else:
        final_prompt = ""
    return final_prompt
