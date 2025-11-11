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
   - **Extraction**: Extract ontological concepts from text using GraphUpdate operations
   - **GraphUpdate**: LLM outputs structured SPARQL operations (insert/delete) instead of full TTL
   - **Update Application**: GraphUpdate operations are applied incrementally to the ontology graph
   - **Sublimation**: Refine and enhance the ontology
   - **Criticism**: Validate ontology structure and relationships
   - **Versioning**: Automatic semantic version increment based on changes (MAJOR/MINOR/PATCH)
   - **Timestamp**: Tracks last update time with `updated_at` field

4. **Fact Processing**
   - **Extraction**: Extract factual information from text using GraphUpdate operations
   - **GraphUpdate**: LLM outputs structured SPARQL operations for facts updates
   - **Update Application**: GraphUpdate operations are applied incrementally to the facts graph
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
- Extracts new concepts using GraphUpdate operations (token-efficient)
- Applies incremental updates to ontology graph
- Validates relationships
- Refines structure
- Automatically increments version based on change analysis (MAJOR/MINOR/PATCH)
- Updates timestamp when ontology is modified
- Tracks version lineage with hash-based identifiers

### 4. Fact Extraction
- Identifies entities
- Extracts relationships using GraphUpdate operations (token-efficient)
- Applies incremental updates to facts graph
- Validates facts
- Combines information from all chunks

### 5. Output Generation
- Produces RDF graph
- Generates ontology with version and timestamp
- Provides extracted facts
- Reports budget usage (LLM calls, characters sent/received, triples generated)
- Logs budget summary at end of processing

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
   - GraphUpdate operations significantly reduce token usage compared to full graph generation
   - Monitor triple generation metrics to understand graph growth

## Next Steps

- Check [API Reference](../reference/onto.md) 