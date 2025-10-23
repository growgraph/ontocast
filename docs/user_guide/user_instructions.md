# User Instructions

User instructions allow you to provide specific guidance to OntoCast about what to focus on during ontology and facts extraction. This feature is particularly useful when you want to direct the AI's attention to specific types of entities, relationships, or concepts in your documents.

---

## Overview

User instructions work by injecting custom instructions into the AI prompts used during:

- **Ontology Extraction**: When the system extracts domain concepts and relationships
- **Facts Extraction**: When the system extracts specific facts from your documents

This allows you to customize the extraction process based on your specific needs and domain requirements.

---

## How User Instructions Work

### 1. Ontology User Instructions

Ontology user instructions guide the AI when extracting domain concepts and relationships from your documents. These instructions help focus on specific types of entities or relationships.

**Example:**
```
Focus on extracting geographical locations, organizations, and their relationships. Pay special attention to company mergers, acquisitions, and partnerships.
```

### 2. Facts User Instructions

Facts user instructions guide the AI when extracting specific facts and instances from your documents. These instructions help focus on particular types of facts or data points.

**Example:**
```
Extract financial data, dates, and numerical values. Focus on revenue, profit, and growth metrics. Include all monetary amounts with proper currency information.
```

---

## Usage Methods

### 1. JSON API Requests

When sending JSON requests to the API, include user instructions in your payload:

```json
{
  "text": "Your document text here...",
  "ontology_user_instruction": "Focus on extracting geographical locations and organizations",
  "facts_user_instruction": "Extract financial data and numerical values with proper currency information"
}
```

### 2. Form Data (Multipart)

When using multipart form data, include user instructions as form fields:

```bash
curl -X POST http://localhost:8999/process \
  -F "file=@document.pdf" \
  -F "ontology_user_instruction=Focus on extracting geographical locations and organizations" \
  -F "facts_user_instruction=Extract financial data and numerical values"
```

### 3. Programmatic Usage

When using OntoCast programmatically, set user instructions in the AgentState:

```python
from ontocast.onto.state import AgentState

# Create state with user instructions
state = AgentState(
    input_text="Your document text...",
    ontology_user_instruction="Focus on extracting geographical locations and organizations",
    facts_user_instruction="Extract financial data and numerical values"
)
```

---

## Best Practices

### 1. Be Specific and Clear

**Good:**
```
Focus on extracting company names, financial metrics, and business relationships. Pay special attention to revenue, profit, and growth data.
```

**Avoid:**
```
Extract everything important.
```

### 2. Use Domain-Specific Language

**Good:**
```
Extract medical diagnoses, symptoms, treatments, and patient information. Focus on ICD-10 codes and medical terminology.
```

**Avoid:**
```
Extract medical stuff.
```

### 3. Provide Context

**Good:**
```
Extract legal entities, court cases, and legal relationships. Focus on case numbers, dates, and legal precedents mentioned in the document.
```

**Avoid:**
```
Extract legal information.
```

### 4. Specify Data Types

**Good:**
```
Extract numerical data with proper units (currency, percentages, measurements). Include dates in ISO format and geographical coordinates.
```

**Avoid:**
```
Extract numbers and dates.
```

---

## Common Use Cases

### 1. Financial Documents

**Ontology Instruction:**
```
Focus on extracting financial concepts, business entities, and economic relationships. Pay attention to revenue streams, cost structures, and financial metrics.
```

**Facts Instruction:**
```
Extract all monetary amounts with currency codes, percentages, and financial ratios. Include dates for financial periods and growth rates.
```

### 2. Medical Documents

**Ontology Instruction:**
```
Focus on extracting medical conditions, treatments, symptoms, and healthcare relationships. Pay attention to medical terminology and clinical concepts.
```

**Facts Instruction:**
```
Extract patient information, medical codes (ICD-10, CPT), dosages, and treatment timelines. Include all medical measurements and lab values.
```

### 3. Legal Documents

**Ontology Instruction:**
```
Focus on extracting legal entities, court cases, legal relationships, and regulatory frameworks. Pay attention to legal terminology and precedents.
```

**Facts Instruction:**
```
Extract case numbers, court dates, legal citations, and regulatory compliance information. Include all legal references and precedents.
```

### 4. Scientific Papers

**Ontology Instruction:**
```
Focus on extracting scientific concepts, methodologies, and research relationships. Pay attention to scientific terminology and theoretical frameworks.
```

**Facts Instruction:**
```
Extract experimental data, measurements, statistical results, and research findings. Include all numerical data with proper units and significance levels.
```

---

## Advanced Examples

### 1. Multi-Domain Extraction

```json
{
  "text": "Your document text...",
  "ontology_user_instruction": "Extract both business and technical concepts. Focus on companies, products, technologies, and their relationships.",
  "facts_user_instruction": "Extract business metrics, technical specifications, and performance data. Include all numerical values with proper context."
}
```

### 2. Temporal Focus

```json
{
  "text": "Your document text...",
  "ontology_user_instruction": "Focus on extracting entities and relationships that are time-sensitive or have temporal aspects.",
  "facts_user_instruction": "Extract all dates, time periods, and temporal relationships. Pay special attention to historical events and chronological data."
}
```

### 3. Geographic Focus

```json
{
  "text": "Your document text...",
  "ontology_user_instruction": "Focus on extracting geographical entities, locations, and spatial relationships.",
  "facts_user_instruction": "Extract all geographical coordinates, addresses, and location-specific data. Include all spatial and geographical information."
}
```

---

## Integration with Workflow

User instructions are integrated into the OntoCast workflow at specific points:

1. **Document Processing**: Instructions are extracted from JSON input during document conversion
2. **Ontology Extraction**: Instructions guide the AI when extracting domain concepts
3. **Facts Extraction**: Instructions guide the AI when extracting specific facts
4. **Critique Phase**: Instructions are used during the critique and improvement phases

---

## Troubleshooting

### Common Issues

1. **Instructions Not Applied**: Ensure instructions are properly formatted in your JSON payload
2. **Vague Results**: Make instructions more specific and detailed
3. **Missing Data**: Check if instructions are too restrictive or unclear

### Debug Tips

1. **Check Logs**: Look for debug messages about user instructions in the server logs
2. **Test with Simple Instructions**: Start with basic instructions and refine
3. **Validate JSON**: Ensure your JSON payload is properly formatted

### Example Debug Output

```
DEBUG - Set ontology user instruction: Focus on extracting geographical locations and organizations
DEBUG - Set facts user instruction: Extract financial data and numerical values
```

---

## API Reference

### Request Format

```json
{
  "text": "string",
  "ontology_user_instruction": "string (optional)",
  "facts_user_instruction": "string (optional)"
}
```

### Response Format

The response includes the extracted ontology and facts, with user instructions influencing the extraction process:

```json
{
  "status": "success",
  "ontology": "...",
  "facts": "...",
  "metadata": {
    "ontology_user_instruction": "Focus on extracting geographical locations and organizations",
    "facts_user_instruction": "Extract financial data and numerical values"
  }
}
```

---

## Best Practices Summary

1. **Be Specific**: Provide clear, detailed instructions
2. **Use Domain Language**: Include relevant terminology
3. **Provide Context**: Explain what you're looking for
4. **Test and Refine**: Start simple and improve based on results
5. **Document Your Instructions**: Keep track of what works best for your use case

---

## Examples by Domain

### Healthcare
- **Ontology**: "Focus on medical conditions, treatments, and healthcare relationships"
- **Facts**: "Extract patient data, medical codes, and clinical measurements"

### Finance
- **Ontology**: "Focus on financial entities, business relationships, and economic concepts"
- **Facts**: "Extract monetary amounts, financial ratios, and economic indicators"

### Legal
- **Ontology**: "Focus on legal entities, court cases, and regulatory frameworks"
- **Facts**: "Extract case numbers, legal citations, and compliance information"

### Scientific
- **Ontology**: "Focus on scientific concepts, methodologies, and research relationships"
- **Facts**: "Extract experimental data, measurements, and research findings"

### Technical
- **Ontology**: "Focus on technical concepts, systems, and technological relationships"
- **Facts**: "Extract technical specifications, performance metrics, and system data"
