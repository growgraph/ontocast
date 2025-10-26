# OntoCast Workflow

This document describes the workflow of OntoCast's document processing pipeline.

## Overview

The OntoCast workflow consists of several stages that transform input documents into structured knowledge:

1. **Document Conversion**
   - Input documents are converted to markdown format
   - Supports various input formats (PDF, DOCX, TXT, MD)

2. **Text Chunking**
   - Documents are split into manageable chunks
   - Chunks are processed sequentially
   - Head chunks are processed first to establish context

3. **Ontology Processing**
   - **Selection**: Choose appropriate ontology for content
   - **Extraction**: Extract ontological concepts from text
   - **Sublimation**: Refine and enhance the ontology
   - **Criticism**: Validate ontology structure and relationships
   - **Versioning**: Automatic semantic version increment based on changes (MAJOR/MINOR/PATCH)
   - **Timestamp**: Tracks last update time with `updated_at` field

4. **Fact Processing**
   - **Extraction**: Extract factual information from text
   - **Criticism**: Validate extracted facts
   - **Aggregation**: Combine facts from all chunks

## Detailed Flow

### 1. Document Input
- Accepts text or file input
- Converts to markdown format
- Preserves document structure

### 2. Text Processing
- Splits text into chunks
- Processes head chunks first
- Maintains context between chunks

### 3. Ontology Management
- Selects relevant ontology
- Extracts new concepts
- Validates relationships
- Refines structure
- Automatically increments version based on change analysis
- Updates timestamp when ontology is modified

### 4. Fact Extraction
- Identifies entities
- Extracts relationships
- Validates facts
- Combines information

### 5. Output Generation
- Produces RDF graph
- Generates ontology with version and timestamp
- Provides extracted facts
- Reports budget usage (LLM calls, tokens, triples generated)

## Configuration Options

The workflow can be configured through command-line parameters:

- `--head-chunks`: Number of chunks to process first
- `--max-visits`: Maximum visits per node

## Best Practices

1. **Chunk Size**
   - Keep chunks manageable
   - Consider context preservation
   - Balance between detail and processing time

2. **Ontology Selection**
   - Choose appropriate ontology
   - Consider domain specificity
   - Allow for ontology evolution
   - Monitor version increments to track evolution

3. **Fact Validation**
   - Validate extracted facts
   - Check for consistency
   - Handle contradictions

4. **Resource Management**
   - Monitor memory usage
   - Control processing time
   - Handle large documents
   - Review budget summaries to track LLM usage and costs
   - Use budget metrics to estimate processing costs for large documents

## Next Steps

- Check [API Reference](../reference/onto.md) 