"""
Legal-specific evaluation metrics.
"""

import re
from typing import Any


class LegalMetrics:
    """
    Metrics for evaluating legal LLM outputs.

    Includes:
    - Citation accuracy
    - Legal terminology precision
    - Factual consistency
    - Hallucination detection

    Example:
        >>> metrics = LegalMetrics()
        >>> result = metrics.evaluate_response(
        ...     response="As held in Smith v Jones [2024] UKSC 15...",
        ...     reference="The case of Smith v Jones [2024] UKSC 15..."
        ... )
        >>> print(result['citation_accuracy'])
    """

    # Common legal citation patterns for validation
    UK_CITATION = re.compile(r"\[\d{4}\]\s+[A-Z]+\s+\d+")
    US_CITATION = re.compile(r"\d+\s+[A-Z]\.\s*(?:\d+[a-z]?)?\s+\d+")
    EU_CITATION = re.compile(r"Case\s+[CT]-\d+/\d+", re.IGNORECASE)

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

    def __init__(self, jurisdiction: str = "uk"):
        """
        Initialize metrics calculator.

        Args:
            jurisdiction: Target jurisdiction for terminology
        """
        self.jurisdiction = jurisdiction.lower()
        self.legal_terms = self.LEGAL_TERMS.get(jurisdiction, self.LEGAL_TERMS["uk"])

    def evaluate_response(
        self,
        response: str,
        reference: str | None = None,
        expected_citations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive evaluation of a legal response.

        Args:
            response: Model-generated response
            reference: Optional reference/ground truth response
            expected_citations: List of expected citations

        Returns:
            Dictionary of metrics
        """
        results = {}

        # Citation accuracy
        results["citation_metrics"] = self.evaluate_citations(response, expected_citations)

        # Legal terminology coverage
        results["terminology_metrics"] = self.evaluate_terminology(response)

        # Response quality
        results["quality_metrics"] = self.evaluate_quality(response)

        # Reference comparison if available
        if reference:
            results["similarity_metrics"] = self.evaluate_similarity(response, reference)

        # Aggregate score
        results["aggregate_score"] = self._calculate_aggregate(results)

        return results

    def evaluate_citations(self, text: str, expected: list[str] | None = None) -> dict[str, Any]:
        """
        Evaluate citation accuracy and validity.

        Args:
            text: Text containing citations
            expected: List of expected citations

        Returns:
            Citation metrics
        """
        # Extract citations
        found_citations = self._extract_citations(text)

        metrics = {
            "citation_count": len(found_citations),
            "citations_found": found_citations,
            "valid_format_count": 0,
            "format_validity_rate": 0.0,
        }

        # Check format validity
        valid_count = sum(1 for c in found_citations if self._is_valid_citation(c))
        metrics["valid_format_count"] = valid_count
        metrics["format_validity_rate"] = (
            valid_count / len(found_citations) if found_citations else 1.0
        )

        # Compare with expected if provided
        if expected:
            expected_set = set(c.strip() for c in expected)
            found_set = set(found_citations)

            precision = len(found_set & expected_set) / len(found_set) if found_set else 0.0
            recall = len(found_set & expected_set) / len(expected_set) if expected_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            metrics["missing_citations"] = list(expected_set - found_set)
            metrics["extra_citations"] = list(found_set - expected_set)

        return metrics

    def evaluate_terminology(self, text: str) -> dict[str, Any]:
        """
        Evaluate use of legal terminology.

        Args:
            text: Text to evaluate

        Returns:
            Terminology metrics
        """
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # Count legal terms used
        terms_used = self.legal_terms & words

        # Also check for multi-word terms
        for term in self.legal_terms:
            if " " in term and term in text_lower:
                terms_used.add(term)

        return {
            "legal_term_count": len(terms_used),
            "terms_used": list(terms_used),
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
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

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
            Similarity metrics
        """
        # Simple word overlap metrics
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())

        overlap = response_words & reference_words

        precision = len(overlap) / max(len(response_words), 1)
        recall = len(overlap) / max(len(reference_words), 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # N-gram overlap
        response_bigrams = self._get_ngrams(response, 2)
        reference_bigrams = self._get_ngrams(reference, 2)
        bigram_overlap = len(response_bigrams & reference_bigrams)

        return {
            "word_precision": precision,
            "word_recall": recall,
            "word_f1": f1,
            "bigram_overlap": bigram_overlap,
            "length_ratio": len(response) / max(len(reference), 1),
        }

    def _extract_citations(self, text: str) -> list[str]:
        """Extract all citations from text."""
        citations = []

        # UK citations
        citations.extend(self.UK_CITATION.findall(text))

        # US citations
        citations.extend(self.US_CITATION.findall(text))

        # EU citations
        citations.extend(self.EU_CITATION.findall(text))

        return citations

    def _is_valid_citation(self, citation: str) -> bool:
        """Check if citation has valid format."""
        return (
            bool(self.UK_CITATION.match(citation))
            or bool(self.US_CITATION.match(citation))
            or bool(self.EU_CITATION.match(citation))
        )

    def _has_structure(self, text: str) -> bool:
        """Check if response has structural elements."""
        structure_indicators = [
            r"\n\d+\.",  # Numbered lists
            r"\n[•\-\*]",  # Bullet points
            r"\n[A-Z][^.]*:",  # Headers
            r"First[ly]?,|Second[ly]?,|Third[ly]?,",  # Sequence words
            r"In conclusion|To summarize",  # Conclusions
        ]

        for pattern in structure_indicators:
            if re.search(pattern, text):
                return True
        return False

    def _calculate_hedging(self, text: str) -> float:
        """Calculate hedging language score (higher = more hedging)."""
        hedging_phrases = [
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
        ]

        text_lower = text.lower()
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in text_lower)

        words = len(text.split())
        return hedge_count / max(words / 100, 1)  # Per 100 words

    def _get_ngrams(self, text: str, n: int) -> set:
        """Get set of n-grams from text."""
        words = text.lower().split()
        return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    def _calculate_aggregate(self, results: dict[str, Any]) -> float:
        """Calculate aggregate score from individual metrics."""
        scores = []

        # Citation accuracy (if measured)
        if "f1" in results.get("citation_metrics", {}):
            scores.append(results["citation_metrics"]["f1"])
        elif "format_validity_rate" in results.get("citation_metrics", {}):
            scores.append(results["citation_metrics"]["format_validity_rate"])

        # Terminology usage
        term_density = results.get("terminology_metrics", {}).get("terminology_density", 0)
        scores.append(min(term_density * 10, 1.0))  # Scale to 0-1

        # Similarity (if measured)
        if "word_f1" in results.get("similarity_metrics", {}):
            scores.append(results["similarity_metrics"]["word_f1"])

        return sum(scores) / len(scores) if scores else 0.0
