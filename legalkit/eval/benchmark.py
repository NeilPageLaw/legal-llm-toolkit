"""
Legal LLM benchmarking framework.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from legalkit.eval.metrics import LegalMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    task: str
    metrics: dict[str, float]
    samples_evaluated: int
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark: {self.task}",
            f"Samples evaluated: {self.samples_evaluated}",
            "Metrics:",
        ]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary of all benchmarks."""
        lines = ["=" * 50, "LEGAL LLM BENCHMARK SUITE RESULTS", "=" * 50, ""]

        for result in self.results:
            lines.append(result.summary())
            lines.append("-" * 30)

        # Aggregate metrics
        all_metrics = {}
        for result in self.results:
            for name, value in result.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        lines.append("\nAggregate Metrics:")
        for name, values in all_metrics.items():
            avg = sum(values) / len(values)
            lines.append(f"  avg_{name}: {avg:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [
                {
                    "task": r.task,
                    "metrics": r.metrics,
                    "samples_evaluated": r.samples_evaluated,
                }
                for r in self.results
            ]
        }

    def save(self, path: str):
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class LegalBenchmark:
    """
    Benchmark suite for evaluating legal LLMs.

    Supports multiple evaluation tasks:
    - citation_accuracy: Correct use of legal citations
    - legal_reasoning: Quality of legal analysis
    - contract_qa: Contract-specific question answering
    - case_summarization: Case law summarization
    - legal_ner: Named entity recognition for legal text

    Example:
        >>> benchmark = LegalBenchmark(tasks=["citation_accuracy", "legal_reasoning"])
        >>> results = benchmark.evaluate(model_path="./my-legal-model")
        >>> print(results.summary())
    """

    AVAILABLE_TASKS = [
        "citation_accuracy",
        "legal_reasoning",
        "contract_qa",
        "case_summarization",
        "legal_ner",
        "clause_classification",
    ]

    def __init__(
        self,
        tasks: list[str] | None = None,
        jurisdiction: str = "uk",
        max_samples: int | None = None,
    ):
        """
        Initialize benchmark.

        Args:
            tasks: List of tasks to run (default: all)
            jurisdiction: Target jurisdiction
            max_samples: Maximum samples per task (for quick evaluation)
        """
        self.tasks = tasks or self.AVAILABLE_TASKS
        self.jurisdiction = jurisdiction
        self.max_samples = max_samples
        self.metrics = LegalMetrics(jurisdiction=jurisdiction)

        # Validate tasks
        for task in self.tasks:
            if task not in self.AVAILABLE_TASKS:
                raise ValueError(f"Unknown task: {task}. Available: {self.AVAILABLE_TASKS}")

    def evaluate(
        self,
        model_path: str | None = None,
        model=None,
        tokenizer=None,
        generate_fn: Callable | None = None,
        test_data: dict[str, list[dict]] | None = None,
    ) -> BenchmarkSuite:
        """
        Run benchmark evaluation.

        Args:
            model_path: Path to model (will load if model not provided)
            model: Pre-loaded model
            tokenizer: Pre-loaded tokenizer
            generate_fn: Custom generation function(prompt) -> response
            test_data: Custom test data per task

        Returns:
            BenchmarkSuite with results
        """
        # Load model if needed
        if generate_fn is None:
            if model is None and model_path:
                model, tokenizer = self._load_model(model_path)

            if model is not None:
                generate_fn = lambda prompt: self._generate(model, tokenizer, prompt)
            else:
                raise ValueError("Provide model_path, model+tokenizer, or generate_fn")

        results = BenchmarkSuite()

        for task in self.tasks:
            logger.info(f"Running benchmark: {task}")

            # Get test data for task
            task_data = test_data.get(task) if test_data else self._load_test_data(task)

            if self.max_samples:
                task_data = task_data[: self.max_samples]

            # Run evaluation
            result = self._evaluate_task(task, task_data, generate_fn)
            results.results.append(result)

        return results

    def _evaluate_task(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate a single task."""
        evaluators = {
            "citation_accuracy": self._eval_citation_accuracy,
            "legal_reasoning": self._eval_legal_reasoning,
            "contract_qa": self._eval_contract_qa,
            "case_summarization": self._eval_summarization,
            "legal_ner": self._eval_ner,
            "clause_classification": self._eval_classification,
        }

        evaluator = evaluators.get(task, self._eval_generic)
        return evaluator(task, test_data, generate_fn)

    def _eval_citation_accuracy(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate citation accuracy task."""
        per_sample = []
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        for sample in tqdm(test_data, desc="Citation accuracy"):
            prompt = sample.get("prompt", sample.get("text", ""))
            expected_citations = sample.get("citations", [])

            response = generate_fn(prompt)

            metrics = self.metrics.evaluate_citations(response, expected_citations)

            per_sample.append(
                {
                    "prompt": prompt[:100],
                    "response": response[:200],
                    "metrics": metrics,
                }
            )

            total_precision += metrics.get("precision", 0)
            total_recall += metrics.get("recall", 0)
            total_f1 += metrics.get("f1", 0)

        n = len(test_data)

        return BenchmarkResult(
            task=task,
            metrics={
                "precision": total_precision / n if n else 0,
                "recall": total_recall / n if n else 0,
                "f1": total_f1 / n if n else 0,
            },
            samples_evaluated=n,
            per_sample_results=per_sample,
        )

    def _eval_legal_reasoning(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate legal reasoning quality."""
        per_sample = []
        scores = []

        for sample in tqdm(test_data, desc="Legal reasoning"):
            prompt = sample.get("prompt", sample.get("question", ""))
            reference = sample.get("reference", sample.get("answer", ""))

            response = generate_fn(prompt)

            metrics = self.metrics.evaluate_response(response, reference)

            per_sample.append(
                {
                    "prompt": prompt[:100],
                    "response": response[:200],
                    "aggregate_score": metrics["aggregate_score"],
                }
            )

            scores.append(metrics["aggregate_score"])

        return BenchmarkResult(
            task=task,
            metrics={
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
            },
            samples_evaluated=len(test_data),
            per_sample_results=per_sample,
        )

    def _eval_contract_qa(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate contract QA task."""
        per_sample = []
        correct = 0
        partial = 0

        for sample in tqdm(test_data, desc="Contract QA"):
            context = sample.get("context", "")
            question = sample.get("question", "")
            answers = sample.get("answers", [])

            prompt = f"Contract:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            response = generate_fn(prompt)

            # Simple answer matching
            response_lower = response.lower()
            is_correct = any(ans.lower() in response_lower for ans in answers)
            is_partial = any(
                word in response_lower
                for ans in answers
                for word in ans.lower().split()
                if len(word) > 3
            )

            if is_correct:
                correct += 1
            elif is_partial:
                partial += 1

            per_sample.append(
                {
                    "question": question[:100],
                    "response": response[:200],
                    "correct": is_correct,
                    "partial": is_partial,
                }
            )

        n = len(test_data)

        return BenchmarkResult(
            task=task,
            metrics={
                "exact_accuracy": correct / n if n else 0,
                "partial_accuracy": (correct + partial) / n if n else 0,
            },
            samples_evaluated=n,
            per_sample_results=per_sample,
        )

    def _eval_summarization(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate summarization task."""
        per_sample = []
        scores = []

        for sample in tqdm(test_data, desc="Summarization"):
            document = sample.get("document", sample.get("text", ""))
            reference_summary = sample.get("summary", "")

            prompt = f"Summarize the following legal document:\n\n{document}\n\nSummary:"
            response = generate_fn(prompt)

            similarity = self.metrics.evaluate_similarity(response, reference_summary)

            per_sample.append(
                {
                    "document": document[:100],
                    "response": response[:200],
                    "word_f1": similarity["word_f1"],
                }
            )

            scores.append(similarity["word_f1"])

        return BenchmarkResult(
            task=task,
            metrics={
                "avg_word_f1": sum(scores) / len(scores) if scores else 0,
            },
            samples_evaluated=len(test_data),
            per_sample_results=per_sample,
        )

    def _eval_ner(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate NER task (placeholder)."""
        # NER evaluation would require structured output parsing
        return BenchmarkResult(
            task=task,
            metrics={"note": "NER evaluation requires structured output"},
            samples_evaluated=0,
        )

    def _eval_classification(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Evaluate classification task."""
        per_sample = []
        correct = 0

        for sample in tqdm(test_data, desc="Classification"):
            text = sample.get("text", "")
            label = sample.get("label", "")
            options = sample.get("options", [])

            options_str = ", ".join(options) if options else "the appropriate category"
            prompt = f"Classify the following clause as {options_str}:\n\n{text}\n\nClassification:"

            response = generate_fn(prompt).strip().lower()

            is_correct = label.lower() in response
            if is_correct:
                correct += 1

            per_sample.append(
                {
                    "text": text[:100],
                    "response": response[:50],
                    "correct": is_correct,
                }
            )

        n = len(test_data)

        return BenchmarkResult(
            task=task,
            metrics={
                "accuracy": correct / n if n else 0,
            },
            samples_evaluated=n,
            per_sample_results=per_sample,
        )

    def _eval_generic(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: Callable,
    ) -> BenchmarkResult:
        """Generic evaluation for unknown tasks."""
        per_sample = []

        for sample in test_data:
            prompt = sample.get("prompt", sample.get("text", ""))
            response = generate_fn(prompt)

            metrics = self.metrics.evaluate_response(response)
            per_sample.append(
                {
                    "prompt": prompt[:100],
                    "response": response[:200],
                    "metrics": metrics,
                }
            )

        return BenchmarkResult(
            task=task,
            metrics={"samples_processed": len(test_data)},
            samples_evaluated=len(test_data),
            per_sample_results=per_sample,
        )

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        )

        return model, tokenizer

    def _generate(self, model, tokenizer, prompt: str, max_length: int = 512) -> str:
        """Generate response from model."""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt) :]

        return response.strip()

    def _load_test_data(self, task: str) -> list[dict]:
        """Load test data for a task."""
        # This would load from bundled test sets or download from HF
        # For now, return sample data structure
        logger.warning(
            f"No bundled test data for {task}. Provide custom test_data or use HF datasets."
        )

        # Return minimal sample data for demonstration
        sample_data = {
            "citation_accuracy": [
                {
                    "prompt": "What are the key cases on duty of care in negligence?",
                    "citations": ["[1932] AC 562", "[1990] 2 AC 605"],
                }
            ],
            "legal_reasoning": [
                {
                    "question": "What is the test for negligence?",
                    "answer": "Duty of care, breach, causation, and damage.",
                }
            ],
            "contract_qa": [
                {
                    "context": "The agreement shall be for a period of 12 months.",
                    "question": "What is the duration of the agreement?",
                    "answers": ["12 months", "one year"],
                }
            ],
        }

        return sample_data.get(task, [])
