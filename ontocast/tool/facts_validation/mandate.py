"""What one gap pass is asked for, and the only thing it may add.

The gap pass (an insert-only model call after the critic loop) is called for
four kinds of gap: statements missing a role their class requires, results
held by a node that is not a statement, instances linked to nothing, and
measurements the text states that the graph lacks (listed only when
measurement recovery is on). :class:`GapMandate` holds the
gaps of one call. The prompt's chapters are rendered from it, and
:meth:`GapMandate.admits` refuses any proposed fix that adds something the
request did not list, so the request and the filter cannot disagree.

Admitted: a role link on a listed statement, a link with a listed detached
instance at one end, a value whose every number the measurement list or a
misplaced result holds (with its components and the link to it), a statement's
result link to a misplaced named value, and the type and label of a node an
admitted triple introduces, with the role links of a statement it introduces.
A misplaced result is restated, never moved: the pass inserts only, and the
exit removes the misplaced link.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal as TypingLiteral

from pydantic import BaseModel, ConfigDict
from rdflib import RDF, BNode, Graph, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.annotation import is_annotation
from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind, TripleFix
from ontocast.tool.agg.signatures import canonical_literal
from ontocast.tool.facts_validation.critic_patch import _parse_fragment
from ontocast.tool.facts_validation.grounding import node_key, node_of
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    value_components,
)
from ontocast.util.numeric_inventory import NumericInventory, canonical_number

RefusalReason = TypingLiteral["value_not_listed", "outside_mandate"]
"""Why a fix is refused: it adds a number the measurement list does not hold,
or any other statement the request did not ask for."""

#: Detached instances that call the gap pass on their own. A lone unlinked
#: instance is often an aside the text relates to nothing, so it does not pay
#: for a call; a pass called for another gap still lists it.
DETACHED_TRIGGER_MIN = 2

Triple = tuple[Node, Node, Node]


class GapMandate(BaseModel):
    """The gaps one gap-pass call is asked to close.

    Build with :meth:`of`; the unit graph is read again by :meth:`admits`.

    Attributes:
        role_gaps: The unit's ``missing_role`` findings.
        misplaced: The unit's ``role_outside_statement`` findings of the
            result role: values to restate as statements.
        detached: Instances linked to nothing that no role gap names.
        inventory: The missing measurements; empty when recovery is off.
        contract: The catalog's statement contract.
        fact_namespaces: Namespaces holding minted instances.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role_gaps: tuple[FactsUnitFinding, ...] = ()
    misplaced: tuple[FactsUnitFinding, ...] = ()
    detached: tuple[URIRef, ...] = ()
    inventory: NumericInventory = NumericInventory()
    contract: StatementContract = StatementContract()
    fact_namespaces: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        findings: Sequence[FactsUnitFinding],
        isolated: Sequence[URIRef],
        inventory: NumericInventory,
        contract: StatementContract,
        fact_namespaces: Sequence[str],
    ) -> GapMandate:
        """The mandate for the unit's current findings.

        Args:
            findings: The unit's findings; ``missing_role`` and the result
                role's ``role_outside_statement`` are read.
            isolated: The unit's instances linked to nothing; those a role gap
                names are left out.
            inventory: The missing measurements, empty when recovery is off.
            contract: The catalog's statement contract.
            fact_namespaces: Namespaces holding minted instances.
        """
        gaps = tuple(
            finding
            for finding in findings
            if finding.kind == FactsUnitFindingKind.MISSING_ROLE
        )
        named = {finding.subject for finding in gaps}
        return cls(
            role_gaps=gaps,
            misplaced=tuple(
                finding
                for finding in findings
                if finding.kind == FactsUnitFindingKind.ROLE_OUTSIDE_STATEMENT
                and finding.facet == "result"
            ),
            detached=tuple(
                sorted((node for node in isolated if str(node) not in named), key=str)
            ),
            inventory=inventory,
            contract=contract,
            fact_namespaces=tuple(ns for ns in fact_namespaces if ns),
        )

    @property
    def calls(self) -> bool:
        """Whether the gaps pay for a call.

        A measurement, a role gap or a misplaced result does; detached
        instances do from :data:`DETACHED_TRIGGER_MIN` on.
        """
        return (
            bool(self.inventory.measurements)
            or bool(self.role_gaps)
            or bool(self.misplaced)
            or len(self.detached) >= DETACHED_TRIGGER_MIN
        )

    def _numbers(self) -> frozenset[str]:
        found = {canonical_number(m.value) for m in self.inventory.measurements}
        return frozenset(value.lstrip("-") for value in found if value is not None)

    def misplaced_values(self, graph: Graph) -> list[tuple[Node, URIRef, Node]]:
        """(holder, link, value) of each misplaced result still in ``graph``."""
        inverse = self.contract.links("result").inverse
        found: list[tuple[Node, URIRef, Node]] = []
        for finding in self.misplaced:
            subject = node_of(finding.subject)
            for obj in graph.objects(subject, URIRef(finding.predicate)):
                if node_key(obj) != finding.value:
                    continue
                link = URIRef(finding.predicate)
                found.append(
                    (obj, link, subject)
                    if finding.predicate in inverse
                    else (subject, link, obj)
                )
        return found

    def _misplaced_numbers(self, graph: Graph) -> frozenset[str]:
        found: set[str] = set()
        for _, _, value in self.misplaced_values(graph):
            if isinstance(value, Literal):
                if _is_number(value):
                    found.add(_magnitude(value))
                continue
            found |= {
                component.number.lstrip("-")
                for component in value_components(graph, value, self.contract).numbers
                if component.number is not None
            }
        return frozenset(found)

    def _instance(self, node: Node) -> bool:
        return isinstance(node, URIRef) and str(node).startswith(self.fact_namespaces)

    def admits(
        self,
        fix: TripleFix,
        graph: Graph,
        introduced: Collection[Node] = frozenset(),
    ) -> RefusalReason | None:
        """Why ``fix`` is refused, or ``None`` when every triple it adds is asked for.

        A payload that does not parse is not judged here:
        :func:`~ontocast.tool.facts_validation.critic_patch.compile_critic_fixes`
        refuses it.

        Args:
            fix: A proposed insert.
            graph: The unit graph before the pass.
            introduced: Nodes that fixes already admitted in this pass
                introduce; their type and label are admitted here too.
        """
        parsed = _parse_fragment(fix.correct_value, graph)
        if parsed is None:
            return None
        inserts = {triple for triple in parsed if triple not in graph}
        if not inserts:
            return None
        earlier = set(introduced)
        admitted: set[Triple] = set()
        numbers = self._numbers() | self._misplaced_numbers(graph)
        holders = {s for s, p, o in inserts if _is_number(o) and not is_annotation(p)}
        for holder in holders:
            stated = {
                _magnitude(o)
                for s, p, o in inserts
                if s == holder and _is_number(o) and not is_annotation(p)
            }
            if not stated <= numbers:
                return "value_not_listed"
            admitted |= {
                triple
                for triple in inserts
                if triple[0] == holder
                and (
                    triple[1] == RDF.type
                    or isinstance(triple[2], Literal)
                    or (isinstance(triple[2], URIRef) and not self._instance(triple[2]))
                )
            }
            if not _known(graph, holder):
                admitted |= {triple for triple in inserts if triple[2] == holder}
        admitted |= {
            triple
            for triple in inserts
            if self._role_link(triple, graph, parsed)
            or self._restates(triple, graph, parsed)
        }
        detached = set(self.detached)
        admitted |= {
            (s, p, o)
            for s, p, o in inserts
            if p != RDF.type
            and isinstance(o, URIRef | BNode)
            and (s in detached or o in detached)
        }
        # Nodes the admitted triples introduce take their type and label; a
        # statement among them, its role links, until nothing changes.
        while True:
            new_nodes = earlier | {
                node
                for triple in admitted
                for node in (triple[0], triple[2])
                if isinstance(node, URIRef | BNode) and not _known(graph, node)
            }
            more = {
                (s, p, o)
                for s, p, o in inserts
                if s in new_nodes
                and (
                    p == RDF.type
                    or (is_annotation(p) and isinstance(o, Literal))
                    or self._statement_link((s, p, o), parsed, graph)
                )
            } | {
                (s, p, o)
                for s, p, o in inserts
                if o in new_nodes and self._statement_link((s, p, o), parsed, graph)
            }
            if more <= admitted:
                break
            admitted |= more
        if inserts <= admitted:
            return None
        return "outside_mandate"

    def admit(
        self, fixes: Sequence[TripleFix], graph: Graph
    ) -> tuple[list[TripleFix], dict[str, int]]:
        """The fixes :meth:`admits`, and the refused ones counted by reason.

        Judged together, in order: a node an admitted fix introduces counts as
        introduced for the fixes after it, so a link and the type and label
        of its new end may come as two fixes.
        """
        kept: list[TripleFix] = []
        refused: dict[str, int] = {}
        introduced: set[Node] = set()
        for fix in fixes:
            reason = self.admits(fix, graph, introduced)
            if reason is None:
                kept.append(fix)
                introduced |= _introduced(fix, graph)
            else:
                refused[reason] = refused.get(reason, 0) + 1
        return kept, refused

    def _role_link(self, triple: Triple, graph: Graph, parsed: Graph) -> bool:
        """Whether ``triple`` gives a listed statement a role it lacks."""
        subject, predicate, obj = triple
        for gap in self.role_gaps:
            node = URIRef(gap.subject)
            types = set(graph.objects(node, RDF.type)) | set(
                parsed.objects(node, RDF.type)
            )
            if gap.facet == "link":
                if subject == node and gap.predicate in self.contract.properties_of(
                    str(predicate)
                ):
                    return True
                continue
            if not self.contract.is_statement(types):
                # A value linked to nothing: any link to or from it.
                if (subject == node or obj == node) and isinstance(obj, URIRef | BNode):
                    return predicate != RDF.type
                continue
            if gap.facet not in self.contract.roles:
                continue
            links = self.contract.links(gap.facet)
            if (
                subject == node
                and str(predicate) in links.forward
                and self.contract.serves(str(predicate), types)
            ):
                return True
            if (
                obj == node
                and str(predicate) in links.inverse
                and self.contract.serves(str(predicate), types)
            ):
                return True
        return False

    def _restates(self, triple: Triple, graph: Graph, parsed: Graph) -> bool:
        """Whether ``triple`` gives a statement a misplaced named value as its result."""
        values = {
            value
            for _, _, value in self.misplaced_values(graph)
            if isinstance(value, URIRef)
        }
        if not values:
            return False
        links = self.contract.links("result")
        subject, predicate, obj = triple
        for node, value, pool in (
            (subject, obj, links.forward),
            (obj, subject, links.inverse),
        ):
            types = set(parsed.objects(node, RDF.type)) | set(
                graph.objects(node, RDF.type)
            )
            if (
                value in values
                and str(predicate) in pool
                and self.contract.is_statement(types)
                and self.contract.serves(str(predicate), types)
            ):
                return True
        return False

    def _statement_link(self, triple: Triple, parsed: Graph, graph: Graph) -> bool:
        """Whether ``triple`` is a role link of the statement at its end."""
        subject, predicate, obj = triple
        for node, direction in ((subject, "forward"), (obj, "inverse")):
            types = set(parsed.objects(node, RDF.type)) | set(
                graph.objects(node, RDF.type)
            )
            if not self.contract.is_statement(types):
                continue
            for role in self.contract.roles:
                links = self.contract.links(role)
                pool = links.forward if direction == "forward" else links.inverse
                if str(predicate) in pool and self.contract.serves(
                    str(predicate), types
                ):
                    return True
        return False


def _magnitude(obj: Node) -> str:
    """The canonical unsigned value of a numeric literal; empty for anything else."""
    canonical = canonical_literal(obj) if isinstance(obj, Literal) else None
    return canonical[0].lstrip("-") if canonical is not None else ""


def _is_number(obj: Node) -> bool:
    if not isinstance(obj, Literal):
        return False
    canonical = canonical_literal(obj)
    return canonical is not None and canonical[1] == "numeric"


def _introduced(fix: TripleFix, graph: Graph) -> set[Node]:
    """The nodes ``fix`` adds that ``graph`` does not name."""
    parsed = _parse_fragment(fix.correct_value, graph)
    if parsed is None:
        return set()
    return {
        node
        for triple in parsed
        if triple not in graph
        for node in (triple[0], triple[2])
        if isinstance(node, URIRef | BNode) and not _known(graph, node)
    }


def _known(graph: Graph, node: Node) -> bool:
    """Whether ``graph`` already names ``node``, in either position."""
    return (node, None, None) in graph or (None, None, node) in graph


__all__ = ["DETACHED_TRIGGER_MIN", "GapMandate", "RefusalReason"]
