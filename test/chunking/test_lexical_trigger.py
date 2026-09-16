"""Tests for the lexical-trigger retrieval lane."""

from __future__ import annotations

import pytest
from rdflib import OWL, RDFS, SKOS, Namespace

from ontocast.config import LexicalTriggerFusion
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.vector_store.atomizer import GraphAtomizer
from ontocast.tool.vector_store.core import GraphAtom
from ontocast.tool.vector_store.lexical_trigger import (
    LexicalTriggerIndex,
    looks_like_lexical_code,
    tokenize_for_lexical_match,
)
from ontocast.tool.vector_store.patch_retriever import _merge_lexical_trigger_atoms

pytestmark = pytest.mark.unit

QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
UNITS = Namespace("https://growgraph.dev/ontologies/units#")


def _ontology_from_ttl(ttl: str, *, iri: str = "https://example.org/o") -> Ontology:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return Ontology(iri=iri, graph=graph, ontology_id="o")


def test_tokenize_preserves_case_for_unit_symbols() -> None:
    tokens = tokenize_for_lexical_match("red shift of ~96 meV and ~10 MeV")
    assert "meV" in tokens
    assert "MeV" in tokens


def test_looks_like_lexical_code_accepts_formulae() -> None:
    assert looks_like_lexical_code("CsPbBr3", min_len=2, max_len=24)
    assert not looks_like_lexical_code("photoluminescence", min_len=2, max_len=24)


def test_atomizer_collects_qudt_symbol_as_lexical_trigger() -> None:
    ttl = f"""
    @prefix qudt: <{QUDT}> .
    @prefix unit: <{UNIT}> .
    @prefix rdfs: <{RDFS}> .

    unit:MilliEV a qudt:Unit ;
        rdfs:label "millielectronvolt"@en ;
        qudt:symbol "meV" ;
        qudt:ucumCode "meV" .
    """
    ontology = _ontology_from_ttl(ttl)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    mev_atoms = [a for a in atoms if "meV" in a.lexical_triggers]
    assert mev_atoms
    assert "meV" in mev_atoms[0].lexical_triggers


def test_atomizer_heuristic_promotes_formula_label() -> None:
    ttl = f"""
    @prefix owl: <{OWL}> .
    @prefix rdfs: <{RDFS}> .

    <https://example.org/perovmat#CsPbBr3> a owl:NamedIndividual ;
        rdfs:label "CsPbBr3"@en .
    """
    ontology = _ontology_from_ttl(ttl)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    formula_atoms = [a for a in atoms if "CsPbBr3" in a.lexical_triggers]
    assert formula_atoms


def test_lexical_trigger_index_case_sensitive_match() -> None:
    atom_mev = GraphAtom(
        atom_id="a1",
        ontology_iri="https://example.org/units",
        iri=f"{UNITS}millielectronvolt",
        core_representation="millielectronvolt",
        neighborhood_representation="",
        lexical_triggers=["meV"],
    )
    atom_mev_upper = GraphAtom(
        atom_id="a2",
        ontology_iri="https://example.org/units",
        iri=f"{UNIT}MegaEV",
        core_representation="Mega Electron Volt",
        neighborhood_representation="",
        lexical_triggers=["MeV"],
    )
    index = LexicalTriggerIndex(max_match_atoms=8)
    index.register_atoms([atom_mev, atom_mev_upper])
    mev_hits = index.match("shift of 15 meV")
    mega_hits = index.match("energy of 1 MeV")
    assert mev_hits == ["a1"]
    assert mega_hits == ["a2"]


def test_merge_lexical_trigger_atoms_is_additive_and_dedupes_by_iri() -> None:
    semantic = GraphAtom(
        atom_id="s1",
        ontology_iri="https://example.org/matsci",
        iri="https://example.org/matsci#RedShift",
        core_representation="red shift",
        neighborhood_representation="",
    )
    trigger = GraphAtom(
        atom_id="t1",
        ontology_iri="https://example.org/units",
        iri=f"{UNITS}millielectronvolt",
        core_representation="millielectronvolt meV",
        neighborhood_representation="",
        score=1.0,
    )
    duplicate = trigger.model_copy(update={"atom_id": "t2"})
    merged, promoted, appended = _merge_lexical_trigger_atoms(
        [semantic], [trigger, duplicate]
    )
    assert len(merged) == 2
    assert merged[1].iri == f"{UNITS}millielectronvolt"
    assert promoted == 0
    assert appended == 1


def test_merge_lexical_trigger_max_merge_promotes_weak_semantic_hit() -> None:
    """A case-sensitive trigger match must lift an atom retrieval already found.

    The legacy ``append`` mode silently discarded trigger evidence for IRIs
    already among the semantic hits — the one signal that distinguishes
    ``meV`` from ``MeV`` never landed.
    """
    weak_semantic = GraphAtom(
        atom_id="s1",
        ontology_iri="https://example.org/units",
        iri=f"{UNITS}millielectronvolt",
        core_representation="millielectronvolt meV",
        neighborhood_representation="",
        score=0.1667,
    )
    trigger = weak_semantic.model_copy(update={"atom_id": "t1", "score": 0.35})

    merged, promoted, appended = _merge_lexical_trigger_atoms(
        [weak_semantic], [trigger], fusion=LexicalTriggerFusion.MAX_MERGE
    )
    assert len(merged) == 1
    assert merged[0].score == 0.35
    assert promoted == 1
    assert appended == 0

    # Trigger evidence never lowers an already-strong semantic score.
    strong_semantic = weak_semantic.model_copy(update={"score": 0.583})
    merged, promoted, _ = _merge_lexical_trigger_atoms(
        [strong_semantic], [trigger], fusion=LexicalTriggerFusion.MAX_MERGE
    )
    assert merged[0].score == 0.583
    assert promoted == 0


def test_merge_lexical_trigger_append_mode_keeps_legacy_behavior() -> None:
    weak_semantic = GraphAtom(
        atom_id="s1",
        ontology_iri="https://example.org/units",
        iri=f"{UNITS}millielectronvolt",
        core_representation="millielectronvolt meV",
        neighborhood_representation="",
        score=0.1667,
    )
    trigger = weak_semantic.model_copy(update={"atom_id": "t1", "score": 0.35})
    merged, promoted, appended = _merge_lexical_trigger_atoms(
        [weak_semantic], [trigger], fusion=LexicalTriggerFusion.APPEND
    )
    assert len(merged) == 1
    assert merged[0].score == 0.1667
    assert promoted == 0
    assert appended == 0


def test_skos_notation_becomes_lexical_trigger() -> None:
    ttl = f"""
    @prefix skos: <{SKOS}> .
    @prefix rdfs: <{RDFS}> .

    <https://example.org/chem#CsPbBr3> a skos:Concept ;
        rdfs:label "lead halide perovskite"@en ;
        skos:notation "CsPbBr3" .
    """
    ontology = _ontology_from_ttl(ttl)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    assert any("CsPbBr3" in a.lexical_triggers for a in atoms)


def test_tokenize_admits_non_ascii_symbol_starts() -> None:
    """µm, °C, %, Å must tokenize whole — SI-reporting prose is full of them."""
    tokens = tokenize_for_lexical_match(
        "a 70 µm film at 25 °C with 0.5 % yield and 5 Å spacing"
    )
    assert "µm" in tokens
    assert "°C" in tokens
    assert "%" in tokens
    assert "Å" in tokens
    # Greek mu variant, too.
    assert "μm" in tokenize_for_lexical_match("about 3 μm")


def test_short_non_ascii_trigger_matches() -> None:
    atom = GraphAtom(
        atom_id="a1",
        ontology_iri="https://example.org/units",
        iri=f"{UNITS}micrometre",
        core_representation="micrometre",
        neighborhood_representation="",
        lexical_triggers=["µm"],
    )
    index = LexicalTriggerIndex(max_match_atoms=8)
    index.register_atoms([atom])
    assert index.match("a grain size of 70 µm") == ["a1"]


def test_substring_scan_respects_token_boundaries() -> None:
    """`mA/cm²` in text must not fire the `A/cm²` trigger (measured false hit)."""
    ampere = GraphAtom(
        atom_id="ampere",
        ontology_iri="https://example.org/units",
        iri=f"{UNIT}A-PER-CentiM2",
        core_representation="ampere per square centimetre",
        neighborhood_representation="",
        lexical_triggers=["A/cm²"],
    )
    milliampere = GraphAtom(
        atom_id="milliampere",
        ontology_iri="https://example.org/units",
        iri=f"{UNIT}MilliA-PER-CentiM2",
        core_representation="milliampere per square centimetre",
        neighborhood_representation="",
        lexical_triggers=["mA/cm²"],
    )
    per_centim = GraphAtom(
        atom_id="per_centim",
        ontology_iri="https://example.org/units",
        iri=f"{UNIT}PER-CentiM",
        core_representation="reciprocal centimetre",
        neighborhood_representation="",
        lexical_triggers=["/cm"],
    )
    index = LexicalTriggerIndex(max_match_atoms=8)
    index.register_atoms([ampere, milliampere, per_centim])

    hits = index.match("a current density of 20 mA/cm² was applied")
    assert "milliampere" in hits
    assert "ampere" not in hits
    assert "per_centim" not in hits

    # The exact trigger still fires at a real boundary.
    assert "ampere" in index.match("a current density of 20 A/cm² was applied")


def test_atomizer_collects_dcterms_alternative_as_label() -> None:
    """dcterms:alternative synonyms must reach the surface forms (sf6)."""
    ttl = f"""
    @prefix owl: <{OWL}> .
    @prefix rdfs: <{RDFS}> .
    @prefix dcterms: <http://purl.org/dc/terms/> .

    <https://example.org/matsci#PhotoluminescenceSpectroscopy> a owl:Class ;
        rdfs:label "photoluminescence spectroscopy"@en ;
        dcterms:alternative "PL spectroscopy"@en .
    """
    ontology = _ontology_from_ttl(ttl)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    target = [a for a in atoms if a.iri.endswith("PhotoluminescenceSpectroscopy")]
    assert target
    assert any("pl spectroscopy" in a.minimal_representation for a in target)


def test_atomizer_collects_case_preserved_symbol_surfaces() -> None:
    ttl = f"""
    @prefix qudt: <{QUDT}> .
    @prefix unit: <{UNIT}> .
    @prefix rdfs: <{RDFS}> .

    unit:MegaEV a qudt:Unit ;
        rdfs:label "megaelectronvolt"@en ;
        qudt:symbol "MeV" .
    """
    ontology = _ontology_from_ttl(ttl)
    atoms = GraphAtomizer().atomize(source=ontology, depth=1)
    target = [a for a in atoms if a.iri.endswith("MegaEV")]
    assert target
    # Case preserved — the minimal representation is folded, the surface is not.
    assert "MeV" in target[0].symbol_surfaces
    assert "mev" in target[0].minimal_representation


def test_symbol_case_mismatch_demotes_counterfeit_match() -> None:
    """Regression: prose `meV` must not keep unit:MegaEV ranked
    above the correct millielectronvolt atom."""
    from ontocast.config import SymbolCaseMismatchPolicy
    from ontocast.tool.vector_store.patch_retriever import (
        _demote_case_mismatched_symbol_atoms,
    )

    mega = GraphAtom(
        atom_id="mega",
        ontology_iri="https://example.org/units",
        iri=f"{UNIT}MegaEV",
        core_representation="megaelectronvolt mev",
        neighborhood_representation="",
        symbol_surfaces=["MeV"],
        score=0.6,
    )
    milli = GraphAtom(
        atom_id="milli",
        ontology_iri="https://example.org/units",
        iri=f"{UNITS}millielectronvolt",
        core_representation="millielectronvolt mev",
        neighborhood_representation="",
        symbol_surfaces=["meV"],
        score=0.5,
    )
    query = "an energy shift of 96 meV was observed"

    demoted, penalized = _demote_case_mismatched_symbol_atoms(
        [mega, milli],
        query,
        policy=SymbolCaseMismatchPolicy.DEMOTE,
        demote_factor=0.5,
    )
    assert penalized == 1
    by_id = {a.atom_id: a for a in demoted}
    # Exact-case match untouched; counterfeit match halved and now below it.
    assert by_id["milli"].score == 0.5
    assert by_id["mega"].score == 0.3
    assert float(by_id["milli"].score or 0.0) > float(by_id["mega"].score or 0.0)

    dropped, penalized = _demote_case_mismatched_symbol_atoms(
        [mega, milli],
        query,
        policy=SymbolCaseMismatchPolicy.DROP,
        demote_factor=0.5,
    )
    assert penalized == 1
    assert [a.atom_id for a in dropped] == ["milli"]

    kept, penalized = _demote_case_mismatched_symbol_atoms(
        [mega, milli],
        query,
        policy=SymbolCaseMismatchPolicy.OFF,
        demote_factor=0.5,
    )
    assert penalized == 0
    assert [a.score for a in kept] == [0.6, 0.5]


def test_symbol_case_mismatch_ignores_label_only_atoms() -> None:
    from ontocast.config import SymbolCaseMismatchPolicy
    from ontocast.tool.vector_store.patch_retriever import (
        _demote_case_mismatched_symbol_atoms,
    )

    # No symbol surfaces at all: never penalized, whatever the query casing.
    label_atom = GraphAtom(
        atom_id="label",
        ontology_iri="https://example.org/matsci",
        iri="https://example.org/matsci#RedShift",
        core_representation="red shift",
        neighborhood_representation="",
        score=0.4,
    )
    # Symbol present but absent from the query: untouched.
    unrelated = GraphAtom(
        atom_id="unrelated",
        ontology_iri="https://example.org/units",
        iri=f"{UNIT}KiloGM",
        core_representation="kilogram",
        neighborhood_representation="",
        symbol_surfaces=["kg"],
        score=0.4,
    )
    out, penalized = _demote_case_mismatched_symbol_atoms(
        [label_atom, unrelated],
        "a Red Shift of 96 meV",
        policy=SymbolCaseMismatchPolicy.DEMOTE,
        demote_factor=0.5,
    )
    assert penalized == 0
    assert [a.score for a in out] == [0.4, 0.4]
