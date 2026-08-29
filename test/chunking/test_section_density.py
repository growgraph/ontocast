"""Content-density section classification.

The contract that matters here is the *refusal*: an ambiguous passage must
return ``None``. Section labels drive the extraction filters, so a wrong label
silently drops a chunk or extracts the wrong one, whereas an unlabeled chunk is
merely unselectable.
"""

from ontocast.config.section_labels import load_section_label_schema
from ontocast.tool.chunk.density import classify_by_density, score_section_labels

SCHEMA = load_section_label_schema("academic")

_REFERENCE_LIST = """
[1] Smith, J.; Doe, A. Growth of nanocrystal assemblies. J. Chem. Phys. 2019,
150, 114701. doi:10.1063/1.5079321
[2] Brown, K. et al. Optical properties of superlattices. Nature 2020, 580,
221-226. doi:10.1038/s41586-020-2140-0
[3] Lee, M.; Park, S. Assembly kinetics revisited. ACS Nano 2021, 15, 1123.
doi:10.1021/acsnano.0c07731
[4] Wang, T. et al. Cooperative emission in solids. Phys. Rev. Lett. 2018, 121,
123601. doi:10.1103/PhysRevLett.121.123601
[5] Garcia, R. Superradiance in ensembles. Science 2022, 375, 1099. vol. 375,
pp. 1099-1104, doi:10.1126/science.abm1234
"""

_ACKNOWLEDGEMENTS = """
The authors gratefully acknowledge financial support from the National Science
Foundation under grant no. DMR-1729841. We thank the staff of the shared
characterization facility for assistance with measurements, and we thank
colleagues for helpful discussions on the interpretation of the data. This work
was supported in part by the Department of Energy.
"""

_NEUTRAL_PROSE = """
The assemblies were placed on a temperature-controlled stage and allowed to
equilibrate. Over the course of the study the behaviour of the system remained
consistent with earlier expectations, and no unusual features were observed in
any of the runs that were carried out during this period of the investigation.
"""


class TestConservativeTier:
    def test_reference_list_is_recognised(self):
        result = classify_by_density(_REFERENCE_LIST, SCHEMA)

        assert result is not None
        assert result[0] == "references"

    def test_acknowledgements_are_recognised(self):
        result = classify_by_density(_ACKNOWLEDGEMENTS, SCHEMA)

        assert result is not None
        assert result[0] == "acknowledgements"

    def test_neutral_prose_is_refused(self):
        assert classify_by_density(_NEUTRAL_PROSE, SCHEMA) is None

    def test_short_text_is_refused(self):
        assert classify_by_density("Too short to judge.", SCHEMA) is None

    def test_results_and_methods_are_not_guessed(self):
        """The conservative tier must not attempt the low-precision labels."""
        scores = score_section_labels(_NEUTRAL_PROSE, SCHEMA)

        assert "results" not in scores
        assert "methods" not in scores


class TestAggressiveTier:
    def test_aggressive_scores_additional_labels(self):
        results_prose = (
            "Figure 3a shows the emission spectra. As summarised in Table 2, the "
            "measured efficiency reached 24.1% with a spread of ±0.3%. Figure 4b "
            "compares the decay traces, and Figure 5 shows the trend at 1.85 eV "
            "across the series of 12.4 nm particles."
        )
        scores = score_section_labels(results_prose, SCHEMA, aggressive=True)

        assert scores.get("results", 0.0) > 0.0

    def test_aggressive_is_not_used_by_default(self):
        results_prose = (
            "Figure 3a shows the emission spectra. As summarised in Table 2, the "
            "measured efficiency reached 24.1% with a spread of ±0.3%. Figure 4b "
            "compares the decay traces across the full series of measurements."
        )
        assert "results" not in score_section_labels(results_prose, SCHEMA)


class TestSchemaScoping:
    def test_labels_absent_from_the_schema_are_never_scored(self):
        fiction = load_section_label_schema("fiction")
        scores = score_section_labels(_REFERENCE_LIST, fiction)

        assert "references" not in scores
