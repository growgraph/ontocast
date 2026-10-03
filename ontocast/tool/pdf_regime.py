"""Whether a PDF carries a text layer, to choose a conversion profile per document."""

from __future__ import annotations

import logging

from ontocast.util.optional import require

logger = logging.getLogger(__name__)

#: Pages sampled, spread evenly through the document.
SAMPLE_PAGES = 8
#: Non-whitespace characters a page needs to count as carrying text; a scan
#: with a stamp or a page number alone stays below it.
MIN_PAGE_CHARS = 40


def pdf_has_text_layer(data: bytes) -> bool:
    """True when at least half of the sampled pages carry extractable text.

    Invisible text counts: a scan with an OCR text layer is read from that layer
    rather than re-recognised. A PDF that cannot be opened counts as having no
    text layer, so it goes to the OCR profile.
    """
    pdfium = require("pypdfium2", feature="PDF text-layer detection")
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as e:  # pdfium raises its own error types per failure
        logger.debug("Could not open PDF for text-layer detection: %s", e)
        return False
    try:
        n = len(pdf)
        if n == 0:
            return False
        step = max(1, n // SAMPLE_PAGES)
        sampled = list(range(0, n, step))[:SAMPLE_PAGES]
        with_text = 0
        for i in sampled:
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            textpage.close()
            page.close()
            if sum(not c.isspace() for c in text) >= MIN_PAGE_CHARS:
                with_text += 1
        return with_text * 2 >= len(sampled)
    finally:
        pdf.close()
