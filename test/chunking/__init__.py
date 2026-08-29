"""Conversion to content units: segmentation, section labels, schema choice.

``prepare``/``sizing``/``semantic_chunker`` build the units;
``section_*`` and ``schema_detect`` decide what each one is called and which
schema it is scored against; ``lexical_trigger`` and
``bibliography_routing`` are the two lanes that key off those labels.
"""
