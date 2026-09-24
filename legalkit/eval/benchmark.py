"""
Legal LLM benchmarking framework.
"""

import importlib.util
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from legalkit.data.formatting import to_instruction_format
from legalkit.eval.metrics import LegalMetrics, token_f1, tokenize
from legalkit.eval.samples import get_sample_data

logger = logging.getLogger(__name__)

GenerateFn = Callable[[str], str]

# Fields each task reads from a test record. Alternatives are separated by "|".
TASK_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "citation_accuracy": {
        "required": ("prompt|text|question", "citations"),
        "optional": ("context",),
    },
    "legal_reasoning": {
        "required": ("prompt|question",),
        "optional": ("reference|answer", "citations", "context"),
    },
    "contract_qa": {"required": ("question", "answers"), "optional": ("context",)},
    "case_summarization": {"required": ("document|text", "summary"), "optional": ()},
    "legal_ner": {"required": ("text", "entities"), "optional": ("labels",)},
    "clause_classification": {"required": ("text", "label"), "optional": ("options",)},
}

_ENTITY_LINE = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?(?P<label>[A-Za-z][A-Za-z _]*?)\s*:\s*(?P<text>.+?)\s*$",
    re.MULTILINE,
)


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

    def to_dict(self, include_samples: bool = True) -> dict[str, Any]:
        """Convert to dictionary."""
        record: dict[str, Any] = {
            "task": self.task,
            "metrics": self.metrics,
            "samples_evaluated": self.samples_evaluated,
        }
        if include_samples:
            record["per_sample_results"] = self.per_sample_results
        return record


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary of all benchmarks."""
        lines = ["=" * 50, "LEGAL LLM BENCHMARK SUITE RESULTS", "=" * 50, ""]

        for result in self.results:
            lines.append(result.summary())
            lines.append("-" * 30)

        if self.skipped:
            lines.append("Skipped:")
            for task, reason in self.skipped.items():
                lines.append(f"  {task}: {reason}")

        all_metrics: dict[str, list[float]] = {}
        for result in self.results:
            for name, value in result.metrics.items():
                all_metrics.setdefault(name, []).append(value)
        if all_metrics:
            lines.append("\nAggregate Metrics:")
            for name, values in all_metrics.items():
                lines.append(f"  avg_{name}: {sum(values) / len(values):.4f}")

        return "\n".join(lines)

    def to_dict(self, include_samples: bool = True) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict(include_samples) for r in self.results],
            "skipped": self.skipped,
        }

    def save(self, path: str | Path, include_samples: bool = True) -> Path:
        """
        Save results to JSON, creating parent folders.

        Per-sample results include model outputs; if the test data is
        confidential, so is this file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_samples), f, indent=2, ensure_ascii=False)
        return path


class LegalBenchmark:
    """
    Benchmark suite for evaluating legal LLMs.

    Supports multiple evaluation tasks:
    - citation_accuracy: Does the model cite the right authorities?
      Reports precision/recall/F1, and grounding when a context is given.
    - legal_reasoning: Quality of legal analysis against a reference answer
    - contract_qa: Contract question answering (exact match, token F1)
    - case_summarization: Case law summarisation (word F1, ROUGE-L)
    - legal_ner: Named entity recognition for legal text
    - clause_classification: Contract clause classification

    Test data is a dict of task name to records, or a directory holding one
    ``<task>.jsonl`` file per task (see TASK_FIELDS for the record fields).
    Without test data, a handful of built-in samples is used: enough to
    smoke-test a pipeline, not to measure a model.

    Example:
        >>> benchmark = LegalBenchmark(tasks=["citation_accuracy", "legal_reasoning"])
        >>> results = benchmark.evaluate(model_path="./my-legal-model", test_data="./eval_data")
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
        max_new_tokens: int = 512,
        prompt_template: str | None = None,
        trust_remote_code: bool = False,
        show_progress: bool = True,
    ):
        """
        Initialize benchmark.

        Args:
            tasks: List of tasks to run (default: all)
            jurisdiction: Target jurisdiction
            max_samples: Maximum samples per task (for quick evaluation)
            max_new_tokens: Generation length limit when loading a model
            prompt_template: Wrap prompts in the template used for
                fine-tuning ("alpaca", "chatml" or a format string; see
                to_instruction_format). Default: plain prompts.
            trust_remote_code: Allow model repositories to run their own
                code when loading. Only enable for models you trust.
            show_progress: Show progress bars
        """
        self.tasks = list(tasks) if tasks else list(self.AVAILABLE_TASKS)
        self.jurisdiction = jurisdiction
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template
        self.trust_remote_code = trust_remote_code
        self.show_progress = show_progress
        self.metrics = LegalMetrics(jurisdiction=jurisdiction)

        for task in self.tasks:
            if task not in self.AVAILABLE_TASKS:
                raise ValueError(f"Unknown task: {task}. Available: {self.AVAILABLE_TASKS}")

    def evaluate(
        self,
        model_path: str | None = None,
        model=None,
        tokenizer=None,
        generate_fn: GenerateFn | None = None,
        test_data: dict[str, list[dict]] | str | Path | None = None,
    ) -> BenchmarkSuite:
        """
        Run benchmark evaluation.

        Args:
            model_path: Path or Hub id of the model (loaded if model not provided)
            model: Pre-loaded model
            tokenizer: Pre-loaded tokenizer
            generate_fn: Custom generation function(prompt) -> response
            test_data: Test records per task, or a directory of
                ``<task>.jsonl`` files. Tasks without data are skipped.
                Defaults to the built-in samples.

        Returns:
            BenchmarkSuite with results
        """
        if isinstance(test_data, (str, Path)):
            test_data = self.load_test_data(test_data)

        data_by_task: dict[str, list[dict]] = {}
        suite = BenchmarkSuite()
        if test_data is None:
            logger.warning(
                "No test data given: using the built-in samples, a smoke test "
                "rather than a benchmark. Pass test_data for real results."
            )
        for task in self.tasks:
            task_data = get_sample_data(task) if test_data is None else test_data.get(task)
            if task_data is None:
                suite.skipped[task] = "no test data provided"
                logger.warning(f"Skipping {task}: no test data provided")
                continue
            if self.max_samples:
                task_data = task_data[: self.max_samples]
            if not task_data:
                suite.skipped[task] = "no samples"
                logger.warning(f"Skipping {task}: no samples")
                continue
            self._validate(task, task_data)
            data_by_task[task] = task_data

        if not data_by_task:
            return suite
        if generate_fn is None:
            generate_fn = self._model_generate_fn(model_path, model, tokenizer)

        for task, task_data in data_by_task.items():
            logger.info(f"Running benchmark: {task}")
            suite.results.append(self._evaluate_task(task, task_data, generate_fn))

        return suite

    def _model_generate_fn(self, model_path: str | None, model, tokenizer) -> GenerateFn:
        if model is None and model_path:
            model, tokenizer = self._load_model(model_path)
        if model is None or tokenizer is None:
            raise ValueError("Provide model_path, model and tokenizer, or generate_fn")

        def generate(prompt: str) -> str:
            return self._generate(model, tokenizer, prompt)

        return generate

    @classmethod
    def load_test_data(cls, path: str | Path) -> dict[str, list[dict]]:
        """
        Load test data from a directory of ``<task>.jsonl`` (or ``.json``)
        files, or from a JSON file mapping task names to record lists.
        """
        path = Path(path)
        if path.is_dir():
            data = {}
            for task in cls.AVAILABLE_TASKS:
                for suffix in (".jsonl", ".json"):
                    file = path / f"{task}{suffix}"
                    if file.exists():
                        data[task] = _read_records(file)
                        break
            ignored = sorted(
                f.name
                for f in path.iterdir()
                if f.suffix in (".jsonl", ".json") and f.stem not in cls.AVAILABLE_TASKS
            )
            if ignored:
                logger.warning(f"Ignoring files that do not match a task name: {ignored}")
            if not data:
                raise ValueError(
                    f"No test data found in {path}. Expected files named "
                    f"<task>.jsonl for tasks: {', '.join(cls.AVAILABLE_TASKS)}"
                )
            return data

        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected a JSON object mapping task names to records")
            unknown = set(data) - set(cls.AVAILABLE_TASKS)
            if unknown:
                raise ValueError(f"{path}: unknown tasks {sorted(unknown)}")
            return data

        raise ValueError(f"Expected a directory or a .json file, got {path}")

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def _evaluate_task(
        self,
        task: str,
        test_data: list[dict],
        generate_fn: GenerateFn,
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
        return evaluators[task](task, test_data, generate_fn)

    def _eval_citation_accuracy(
        self, task: str, test_data: list[dict], generate_fn: GenerateFn
    ) -> BenchmarkResult:
        """Evaluate whether responses cite the expected authorities."""
        per_sample = []
        scores: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
        grounding = []

        for sample in self._progress(test_data, "Citation accuracy"):
            question = _first(sample, "prompt", "text", "question")
            context = sample.get("context")
            response = generate_fn(self._prompt(question, context))

            metrics = self.metrics.evaluate_citations(response, sample["citations"])
            for name in scores:
                scores[name].append(metrics[name])
            record = {
                "prompt": question[:200],
                "response": response,
                "citations_found": metrics["citations_found"],
                "missing_citations": metrics["missing_citations"],
                "extra_citations": metrics["extra_citations"],
            }
            if context:
                check = self.metrics.evaluate_grounding(response, context)
                grounding.append(check["grounding_rate"])
                record["ungrounded_citations"] = check["ungrounded"]
            per_sample.append(record)

        summary = {name: _mean(values) for name, values in scores.items()}
        if grounding:
            summary["grounding_rate"] = _mean(grounding)
        return BenchmarkResult(task, summary, len(test_data), per_sample)

    def _eval_legal_reasoning(
        self, task: str, test_data: list[dict], generate_fn: GenerateFn
    ) -> BenchmarkResult:
        """Evaluate legal reasoning quality."""
        per_sample = []
        scores = []

        for sample in self._progress(test_data, "Legal reasoning"):
            question = _first(sample, "prompt", "question")
            context = sample.get("context")
            reference = sample.get("reference", sample.get("answer"))
            response = generate_fn(self._prompt(question, context))

            metrics = self.metrics.evaluate_response(
                response,
                reference=reference,
                expected_citations=sample.get("citations"),
                sources=context,
            )
            scores.append(metrics["aggregate_score"])
            per_sample.append(
                {
                    "prompt": question[:200],
                    "response": response,
                    "aggregate_score": metrics["aggregate_score"],
                }
            )

        return BenchmarkResult(
            task,
            {
                "avg_score": _mean(scores),
                "min_score": min(scores, default=0.0),
                "max_score": max(scores, default=0.0),
            },
            len(test_data),
            per_sample,
        )

    def _eval_contract_qa(
        self, task: str, test_data: list[dict], generate_fn: GenerateFn
    ) -> BenchmarkResult:
        """Evaluate contract QA: exact answer match, partial match and token F1."""
        per_sample = []
        correct = partial = 0
        f1_scores = []

        for sample in self._progress(test_data, "Contract QA"):
            context = sample.get("context", "")
            question = sample["question"]
            answers = sample["answers"]
            plain = f"Contract:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            response = generate_fn(self._prompt(question, context, plain=plain))

            normalised = " ".join(tokenize(response))
            is_correct = any(
                re.search(rf"\b{re.escape(' '.join(tokenize(a)))}\b", normalised) for a in answers
            )
            is_partial = is_correct or any(
                word in normalised.split() for a in answers for word in tokenize(a) if len(word) > 3
            )
            best_f1 = max((token_f1(response, a) for a in answers), default=0.0)
            correct += is_correct
            partial += is_partial
            f1_scores.append(best_f1)
            per_sample.append(
                {
                    "question": question[:200],
                    "response": response,
                    "correct": is_correct,
                    "partial": is_partial,
                    "token_f1": best_f1,
                }
            )

        n = len(test_data)
        return BenchmarkResult(
            task,
            {
                "exact_accuracy": correct / n,
                "partial_accuracy": partial / n,
                "token_f1": _mean(f1_scores),
            },
            n,
            per_sample,
        )

    def _eval_summarization(
        self, task: str, test_data: list[dict], generate_fn: GenerateFn
    ) -> BenchmarkResult:
        """Evaluate summarisation against a reference summary."""
        per_sample = []
        word_f1, rouge = [], []

        for sample in self._progress(test_data, "Summarization"):
            document = _first(sample, "document", "text")
            instruction = "Summarise the following legal document."
            plain = f"Summarise the following legal document:\n\n{document}\n\nSummary:"
            response = generate_fn(self._prompt(instruction, document, plain=plain))

            similarity = self.metrics.evaluate_similarity(response, sample["summary"])
            word_f1.append(similarity["word_f1"])
            rouge.append(similarity["rouge_l_f1"])
            per_sample.append(
                {
                    "document": document[:200],
                    "response": response,
                    "word_f1": similarity["word_f1"],
                    "rouge_l_f1": similarity["rouge_l_f1"],
                }
            )

        return BenchmarkResult(
            task,
            {"avg_word_f1": _mean(word_f1), "avg_rouge_l_f1": _mean(rouge)},
            len(test_data),
            per_sample,
        )

    def _eval_ner(
        self, task: str, test_data: list[dict], generate_fn: GenerateFn
    ) -> BenchmarkResult:
        """
        Evaluate legal named entity recognition.

        The model lists entities one per line as "LABEL: text"; predictions
        are scored against the gold (label, text) pairs with micro-averaged
        precision, recall and F1.
        """
        labels = sorted(
            {_entity(e)[0] for sample in test_data for e in sample["entities"]}
            | {label.upper() for sample in test_data for label in sample.get("labels", [])}
        )
        per_sample = []
        tp = fp = fn = 0

        for sample in self._progress(test_data, "Legal NER"):
            text = sample["text"]
            instruction = (
                "Extract the legal entities from the text. List each entity on its own "
                f"line as LABEL: entity text, using only these labels: {', '.join(labels)}."
            )
            plain = f"{instruction}\n\nText: {text}\n\nEntities:"
            response = generate_fn(self._prompt(instruction, text, plain=plain))

            gold = {_entity(e) for e in sample["entities"]}
            predicted = _parse_entities(response, labels)
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            per_sample.append(
                {
                    "text": text[:200],
                    "response": response,
                    "missed": sorted(gold - predicted),
                    "spurious": sorted(predicted - gold),
                }
            )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return BenchmarkResult(
            task,
            {"precision": precision, "recall": recall, "f1": f1},
            len(test_data),
            per_sample,
        )

    def _eval_classification(
        self, task: str, test_data: list[dict], generate_fn: GenerateFn
    ) -> BenchmarkResult:
        """Evaluate clause classification."""
        per_sample = []
        correct = 0

        for sample in self._progress(test_data, "Classification"):
            text = sample["text"]
            label = sample["label"]
            options = sample.get("options", [])

            choices = ", ".join(options) if options else "the appropriate category"
            instruction = (
                f"Classify the clause as one of: {choices}. Answer with the category only."
            )
            plain = f"Classify the following clause as one of: {choices}.\n\nClause: {text}\n\nClassification:"
            response = generate_fn(self._prompt(instruction, text, plain=plain))

            if options:
                predicted = _predict_option(response, options)
                is_correct = predicted is not None and _phrase(predicted) == _phrase(label)
            else:
                predicted = response.strip()
                is_correct = _contains_phrase(response, label)
            correct += is_correct
            per_sample.append(
                {
                    "text": text[:200],
                    "response": response,
                    "predicted": predicted,
                    "label": label,
                    "correct": is_correct,
                }
            )

        return BenchmarkResult(
            task, {"accuracy": correct / len(test_data)}, len(test_data), per_sample
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prompt(
        self, instruction: str, context: str | None = None, plain: str | None = None
    ) -> str:
        """Build the prompt, in the fine-tuning template when one is set."""
        if self.prompt_template:
            return to_instruction_format(
                instruction, context=context or None, template=self.prompt_template
            )
        if plain is not None:
            return plain
        return f"{context}\n\n{instruction}" if context else instruction

    def _progress(self, items: list[dict], description: str):
        return tqdm(items, desc=description, disable=not self.show_progress)

    def _validate(self, task: str, test_data: list[dict]) -> None:
        """Check every record has the fields the task needs, before any generation."""
        for index, record in enumerate(test_data):
            if not isinstance(record, dict):
                raise ValueError(
                    f"{task} record {index}: expected an object, got {type(record).__name__}"
                )
            for requirement in TASK_FIELDS[task]["required"]:
                if not any(record.get(name) is not None for name in requirement.split("|")):
                    raise ValueError(
                        f"{task} record {index} is missing {requirement.replace('|', ' or ')!r}"
                    )

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Loading a model needs the evaluation extras: pip install 'legal-llm-toolkit[eval]'"
            ) from e

        from legalkit._compat import dtype_kwargs

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            **dtype_kwargs("auto"),
        }
        if importlib.util.find_spec("accelerate") is not None:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.eval()
        return model, tokenizer

    def _generate(self, model, tokenizer, prompt: str) -> str:
        """Generate a response greedily, so results are reproducible."""
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _read_records(path: Path) -> list[dict]:
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a list of records")
        return records
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_number}: invalid JSON ({e.msg})") from e
    return records


def _first(record: dict, *names: str) -> str:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return ""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _phrase(text: str) -> str:
    return " ".join(tokenize(text))


def _contains_phrase(text: str, phrase: str) -> bool:
    target = _phrase(phrase)
    return bool(target) and re.search(rf"\b{re.escape(target)}\b", _phrase(text)) is not None


def _predict_option(response: str, options: list[str]) -> str | None:
    """The option mentioned first in the response (the longest, if several start together)."""
    text = _phrase(response)
    best: tuple[int, int, str] | None = None
    for option in options:
        target = _phrase(option)
        match = re.search(rf"\b{re.escape(target)}\b", text) if target else None
        if match and (best is None or (match.start(), -len(target)) < (best[0], best[1])):
            best = (match.start(), -len(target), option)
    return best[2] if best else None


def _normalise_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("\"'“”‘’.,;").lower()


def _entity(entity: dict | list | tuple) -> tuple[str, str]:
    """A gold entity as (LABEL, normalised text); accepts {"text", "label"} or (label, text)."""
    if isinstance(entity, dict):
        label, text = entity["label"], entity["text"]
    else:
        label, text = entity
    return label.upper().replace(" ", "_"), _normalise_entity(text)


def _parse_entities(response: str, labels: list[str]) -> set[tuple[str, str]]:
    allowed = set(labels)
    entities = set()
    for match in _ENTITY_LINE.finditer(response):
        label = match["label"].strip().upper().replace(" ", "_")
        if label in allowed:
            entities.add((label, _normalise_entity(match["text"])))
    return entities
