"""
Legal-specific evaluation metrics.
"""

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from legalkit.preprocess.citations import SUPPORTED_JURISDICTIONS, CitationParser

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens, ignoring punctuation."""
    return _WORD.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """
    SQuAD-style token F1 between a prediction and a reference answer.

    Returns:
        F1 over word tokens (with repeats), from 0.0 to 1.0.
    """
    predicted, expected = tokenize(prediction), tokenize(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """
    ROUGE-L: precision, recall and F1 of the longest common subsequence of words.

    Rewards summaries that keep the reference's content in the same order.
    """
    predicted, expected = tokenize(prediction), tokenize(reference)
    if not predicted or not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(predicted, expected)
    if lcs == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


class LegalMetrics:
    """
    Metrics for evaluating legal LLM outputs.

    Includes:
    - Citation accuracy against expected authorities
    - Citation grounding: flags cited authorities that do not appear in the
      source material, a sign of fabricated ("hallucinated") citations
    - Legal terminology usage
    - Response quality and similarity to a reference (word F1, ROUGE-L)

    Citations are compared in normalised form, so "[1990] 2 A.C. 605" and
    "[1990]  2 AC 605" count as the same authority.

    Example:
        >>> metrics = LegalMetrics()
        >>> result = metrics.evaluate_response(
        ...     response="As held in Smith v Jones [2024] UKSC 15...",
        ...     reference="The case of Smith v Jones [2024] UKSC 15...",
        ...     expected_citations=["[2024] UKSC 15"],
        ... )
        >>> result["citation_metrics"]["recall"]
        1.0
    """

    # Legal terminology for domain-specific evaluation
    LEGAL_TERMS = {
        "uk": {
            "claimant",
            "defendant",
            "appellant",
            "respondent",
            "tort",
            "breach",
            "duty of care",
            "negligence",
            "statute",
            "regulation",
            "precedent",
            "ratio decidendi",
            "obiter dicta",
            "judicial review",
            "injunction",
            "damages",
            "liability",
            "indemnity",
            "warranty",
            "covenant",
            "consideration",
            "estoppel",
            "ultra vires",
        },
        "us": {
            "plaintiff",
            "defendant",
            "appellant",
            "appellee",
            "tort",
            "breach",
            "due process",
            "negligence",
            "statute",
            "regulation",
            "precedent",
            "stare decisis",
            "dicta",
            "certiorari",
            "injunction",
            "damages",
            "liability",
            "indemnification",
            "warranty",
            "covenant",
            "consideration",
            "estoppel",
            "preemption",
            "standing",
        },
        "eu": {
            "applicant",
            "defendant",
            "appellant",
            "member state",
            "directive",
            "regulation",
            "decision",
            "preliminary ruling",
            "proportionality",
            "subsidiarity",
            "direct effect",
            "supremacy",
            "infringement",
            "annulment",
            "preliminary reference",
            "advocate general",
        },
    }

    HEDGING_PHRASES = (
        "may",
        "might",
        "could",
        "possibly",
        "potentially",
        "it appears",
        "it seems",
        "arguably",
        "generally",
        "in some cases",
        "typically",
        "usually",
        "often",
        "subject to",
        "depending on",
        "it is possible",
    )

    def __init__(self, jurisdiction: str = "uk"):
        """
        Initialize metrics calculator.

        Args:
            jurisdiction: Target jurisdiction for terminology
        """
        self.jurisdiction = jurisdiction.lower()
        self.legal_terms = self.LEGAL_TERMS.get(self.jurisdiction, self.LEGAL_TERMS["uk"])
        parser_jurisdiction = (
            self.jurisdiction if self.jurisdiction in SUPPORTED_JURISDICTIONS else "uk"
        )
        self.parser = CitationParser(jurisdiction=parser_jurisdiction)
        self._hedging = re.compile(
            r"\b(?:" + "|".join(re.escape(p) for p in self.HEDGING_PHRASES) + r")\b"
        )

    def evaluate_response(
        self,
        response: str,
        reference: str | None = None,
        expected_citations: list[str] | None = None,
        sources: str | Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive evaluation of a legal response.

        Args:
            response: Model-generated response
            reference: Optional reference/ground truth response
            expected_citations: Authorities the response should cite
            sources: Source material (documents or citation strings) the
                response may rely on; enables the grounding check

        Returns:
            Dictionary of metrics, including an ``aggregate_score`` from 0 to 1
        """
        results: dict[str, Any] = {
            "citation_metrics": self.evaluate_citations(response, expected_citations),
            "terminology_metrics": self.evaluate_terminology(response),
            "quality_metrics": self.evaluate_quality(response),
        }
        if reference:
            results["similarity_metrics"] = self.evaluate_similarity(response, reference)
        if sources is not None:
            results["grounding_metrics"] = self.evaluate_grounding(response, sources)
        results["aggregate_score"] = self._calculate_aggregate(results)
        return results

    def extract_citations(self, text: str, include_legislation: bool = False) -> list[str]:
        """
        Extract the distinct citations in text, in normalised form.

        Args:
            text: Text to search
            include_legislation: Also return legislation references

        Returns:
            Normalised citations in order of first appearance
        """
        return [
            c.normalised or c.raw
            for c in self.parser.parse(text)
            if include_legislation or c.citation_type == "case"
        ]

    def evaluate_citations(
        self,
        text: str,
        expected: list[str] | None = None,
        include_legislation: bool | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate citation accuracy against the expected authorities.

        Args:
            text: Text containing citations
            expected: Citations the text should contain. Without it only
                the citations found are reported.
            include_legislation: Count legislation as well as cases. By
                default, legislation counts when an expected citation is
                legislation.

        Returns:
            citation_count and citations_found, plus precision, recall, f1,
            missing_citations and extra_citations when ``expected`` is given
        """
        expected_citations = (
            [self._canonical(c) for c in expected] if expected is not None else None
        )
        if include_legislation is None:
            include_legislation = bool(expected) and any(
                c.citation_type == "legislation"
                for e in expected or []
                for c in self.parser.parse(e)
            )

        found = self.extract_citations(text, include_legislation)
        metrics: dict[str, Any] = {"citation_count": len(found), "citations_found": found}
        if expected_citations is None:
            return metrics

        expected_set = set(expected_citations)
        found_set = set(found)
        matched = found_set & expected_set
        if not found_set:
            precision = 1.0 if not expected_set else 0.0
        else:
            precision = len(matched) / len(found_set)
        recall = len(matched) / len(expected_set) if expected_set else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "missing_citations": [
                    c for c in dict.fromkeys(expected_citations) if c not in found_set
                ],
                "extra_citations": [c for c in found if c not in expected_set],
            }
        )
        return metrics

    def evaluate_grounding(
        self,
        response: str,
        sources: str | Iterable[str],
        include_legislation: bool = True,
    ) -> dict[str, Any]:
        """
        Check that every authority cited in a response appears in its sources.

        An authority that appears nowhere in the source material may have
        been fabricated. This does not prove a grounded citation is correct
        or supports the proposition it is cited for: verify authorities
        before relying on them.

        Args:
            response: Model-generated text
            sources: Source documents, or a list of known-good citations
            include_legislation: Check legislation references as well as cases

        Returns:
            cited, grounded and ungrounded citation lists and a
            grounding_rate (1.0 when nothing is cited)
        """
        if isinstance(sources, str):
            sources = [sources]
        known: set[str] = set()
        for source in sources:
            for citation in self.parser.parse(source):
                known.add(citation.normalised or citation.raw)
                # A source citing "section 994 of the Companies Act 2006" also
                # supports a response citing the Act itself.
                if citation.instrument:
                    known.add(citation.instrument)
            if len(source) <= 300:  # a bare citation the parser may not recognise
                known.add(re.sub(r"\s+", " ", source).strip())

        cited = self.extract_citations(response, include_legislation)
        grounded = [c for c in cited if c in known]
        ungrounded = [c for c in cited if c not in known]
        return {
            "cited": cited,
            "grounded": grounded,
            "ungrounded": ungrounded,
            "grounding_rate": len(grounded) / len(cited) if cited else 1.0,
        }

    def evaluate_terminology(self, text: str) -> dict[str, Any]:
        """
        Evaluate use of legal terminology.

        Args:
            text: Text to evaluate

        Returns:
            Terminology metrics
        """
        text_lower = text.lower()
        words = set(tokenize(text))
        terms_used = sorted(
            term
            for term in self.legal_terms
            if (
                term in words
                if " " not in term
                else re.search(rf"\b{re.escape(term)}\b", text_lower)
            )
        )
        return {
            "legal_term_count": len(terms_used),
            "terms_used": terms_used,
            "terminology_density": len(terms_used) / max(len(words), 1),
        }

    def evaluate_quality(self, text: str) -> dict[str, Any]:
        """
        Evaluate general response quality.

        Args:
            text: Response text

        Returns:
            Quality metrics
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words = text.split()
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "has_structure": self._has_structure(text),
            "hedging_score": self._calculate_hedging(text),
        }

    def evaluate_similarity(self, response: str, reference: str) -> dict[str, Any]:
        """
        Compare response to reference.

        Args:
            response: Generated response
            reference: Reference response

        Returns:
            Word-set precision/recall/F1, token F1, ROUGE-L F1, bigram
            overlap and length ratio
        """
        response_words = set(tokenize(response))
        reference_words = set(tokenize(reference))
        overlap = response_words & reference_words

        precision = len(overlap) / max(len(response_words), 1)
        recall = len(overlap) / max(len(reference_words), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {
            "word_precision": precision,
            "word_recall": recall,
            "word_f1": f1,
            "token_f1": token_f1(response, reference),
            "rouge_l_f1": rouge_l(response, reference)["f1"],
            "bigram_overlap": len(self._get_ngrams(response, 2) & self._get_ngrams(reference, 2)),
            "length_ratio": len(response) / max(len(reference), 1),
        }

    def _canonical(self, citation: str) -> str:
        """Normalised form of a citation string, for comparison."""
        parsed = self.parser.parse(citation)
        if len(parsed) == 1:
            return parsed[0].normalised or parsed[0].raw
        return re.sub(r"\s+", " ", citation).strip()

    def _has_structure(self, text: str) -> bool:
        """Check if response has structural elements."""
        structure_indicators = [
            r"\n\d+\.",  # Numbered lists
            r"\n[•\-\*]",  # Bullet points
            r"\n[A-Z][^.]*:",  # Headers
            r"\b(?:First|Second|Third)(?:ly)?,",  # Sequence words
            r"In conclusion|To summarize|To summarise",  # Conclusions
        ]
        return any(re.search(pattern, text) for pattern in structure_indicators)

    def _calculate_hedging(self, text: str) -> float:
        """Hedging phrases per 100 words (higher = more hedging)."""
        hedges = len(self._hedging.findall(text.lower()))
        words = len(text.split())
        return hedges / max(words / 100, 1)

    def _get_ngrams(self, text: str, n: int) -> set:
        """Get set of n-grams from text."""
        words = tokenize(text)
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    def _calculate_aggregate(self, results: dict[str, Any]) -> float:
        """Average of the available 0-1 scores."""
        scores = []
        if "f1" in results.get("citation_metrics", {}):
            scores.append(results["citation_metrics"]["f1"])
        if "grounding_metrics" in results:
            scores.append(results["grounding_metrics"]["grounding_rate"])
        density = results.get("terminology_metrics", {}).get("terminology_density", 0)
        scores.append(min(density * 10, 1.0))
        if "word_f1" in results.get("similarity_metrics", {}):
            scores.append(results["similarity_metrics"]["word_f1"])
        return sum(scores) / len(scores) if scores else 0.0
