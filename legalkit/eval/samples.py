"""
Built-in sample data for each benchmark task.

A handful of examples per task, for smoke-testing a model or pipeline and
showing the expected record format. They are far too few to measure
performance: evaluate on your own held-out data (see LegalBenchmark).
"""

from copy import deepcopy
from typing import Any

CLAUSE_TYPES = [
    "termination",
    "confidentiality",
    "limitation of liability",
    "governing law",
    "force majeure",
    "payment",
]

SAMPLE_DATA: dict[str, list[dict[str, Any]]] = {
    "citation_accuracy": [
        {
            "prompt": "Which House of Lords decision established the 'neighbour principle' "
            "in the tort of negligence? Give the citation.",
            "citations": ["[1932] AC 562"],
        },
        {
            "prompt": "Which House of Lords decision set out the three-stage test for "
            "a duty of care in negligence? Give the citation.",
            "citations": ["[1990] 2 AC 605"],
        },
        {
            "prompt": "Which Supreme Court decision restated the approach to establishing "
            "a duty of care, in a claim against a police force? Give the neutral citation.",
            "citations": ["[2018] UKSC 4"],
        },
        {
            "prompt": "Which case established the rule on remoteness of damage for breach "
            "of contract? Give the citation.",
            "citations": ["(1854) 9 Exch 341"],
        },
        {
            "prompt": "What is the neutral citation of the Supreme Court decision on "
            "penalty clauses in Cavendish Square Holding BV v Makdessi?",
            "citations": ["[2015] UKSC 67"],
        },
        {
            "prompt": "What is the neutral citation of Montgomery v Lanarkshire Health "
            "Board, on informed consent to medical treatment?",
            "citations": ["[2015] UKSC 11"],
        },
    ],
    "legal_reasoning": [
        {
            "question": "What must a claimant prove to succeed in a negligence claim in "
            "England and Wales?",
            "answer": "The claimant must prove that the defendant owed them a duty of care, "
            "that the defendant breached that duty, and that the breach caused damage "
            "which is not too remote.",
        },
        {
            "question": "What are the essential elements of a binding contract under English law?",
            "answer": "Offer, acceptance, consideration, an intention to create legal "
            "relations and sufficient certainty of terms.",
        },
        {
            "question": "When is a clause requiring payment on breach of contract an "
            "unenforceable penalty?",
            "answer": "Following Cavendish Square Holding BV v Makdessi [2015] UKSC 67, a "
            "clause is a penalty if it is a secondary obligation which imposes a detriment "
            "on the contract-breaker out of all proportion to any legitimate interest of "
            "the innocent party in the enforcement of the primary obligation.",
            "citations": ["[2015] UKSC 67"],
        },
    ],
    "contract_qa": [
        {
            "context": "This Agreement shall commence on the Effective Date and continue "
            "for a period of 12 months, unless terminated earlier in accordance with "
            "clause 14.",
            "question": "What is the duration of the agreement?",
            "answers": ["12 months", "twelve months", "one year"],
        },
        {
            "context": "Either party may terminate this Agreement by giving not less than "
            "30 days' written notice to the other party.",
            "question": "How much notice is required to terminate the agreement?",
            "answers": ["30 days", "thirty days"],
        },
        {
            "context": "This Agreement and any dispute or claim arising out of or in "
            "connection with it shall be governed by and construed in accordance with "
            "the law of England and Wales.",
            "question": "Which law governs the agreement?",
            "answers": ["the law of England and Wales", "English law"],
        },
        {
            "context": "The Supplier's total liability to the Customer in respect of all "
            "losses arising under or in connection with this Agreement shall in no "
            "circumstances exceed 150% of the Charges paid in the 12 months preceding "
            "the claim.",
            "question": "What is the cap on the Supplier's liability?",
            "answers": ["150% of the Charges", "150%"],
        },
    ],
    "case_summarization": [
        {
            "document": "The pursuer drank ginger beer bought for her by a friend in a "
            "café. She alleged that the opaque bottle contained the decomposed remains "
            "of a snail and that she suffered shock and gastroenteritis. She sued the "
            "manufacturer, with whom she had no contract. The House of Lords held by a "
            "majority that a manufacturer owes a duty of care to the ultimate consumer of "
            "its products where there is no reasonable possibility of intermediate "
            "examination. Lord Atkin stated that you must take reasonable care to avoid "
            "acts or omissions which you can reasonably foresee would be likely to injure "
            "your neighbour.",
            "summary": "A manufacturer owes a duty of care to the ultimate consumer of its "
            "products. Lord Atkin's neighbour principle requires reasonable care to avoid "
            "acts or omissions likely to injure those one can reasonably foresee being "
            "affected.",
        },
        {
            "document": "The claimants' mill stopped when its crankshaft broke. They "
            "engaged the defendants, carriers, to take the broken shaft to engineers as "
            "a pattern for a new one. Delivery was delayed, the mill stayed idle for "
            "longer, and the claimants claimed their lost profits. The court held that "
            "damages for breach of contract are recoverable only for losses arising "
            "naturally from the breach, or such as may reasonably be supposed to have "
            "been in the contemplation of both parties when they made the contract. The "
            "lost profits were not recoverable because the carriers were not told that "
            "the mill would be idle until the shaft was returned.",
            "summary": "Damages for breach of contract cover losses arising naturally from "
            "the breach or within both parties' reasonable contemplation when contracting. "
            "The carriers were not liable for lost profits they could not have contemplated.",
        },
    ],
    "legal_ner": [
        {
            "text": "In Donoghue v Stevenson [1932] AC 562 the House of Lords considered "
            "the liability of a manufacturer.",
            "entities": [
                {"text": "Donoghue v Stevenson", "label": "CASE"},
                {"text": "[1932] AC 562", "label": "CITATION"},
                {"text": "House of Lords", "label": "COURT"},
            ],
        },
        {
            "text": "Section 2(1) of the Unfair Contract Terms Act 1977 restricts the "
            "exclusion of liability for death or personal injury resulting from negligence.",
            "entities": [{"text": "Unfair Contract Terms Act 1977", "label": "STATUTE"}],
        },
        {
            "text": "Lord Reed gave the leading judgment in Robinson v Chief Constable of "
            "West Yorkshire Police [2018] UKSC 4.",
            "entities": [
                {"text": "Lord Reed", "label": "JUDGE"},
                {"text": "Robinson v Chief Constable of West Yorkshire Police", "label": "CASE"},
                {"text": "[2018] UKSC 4", "label": "CITATION"},
            ],
        },
    ],
    "clause_classification": [
        {
            "text": "Either party may terminate this Agreement with immediate effect by "
            "written notice if the other party commits a material breach which is not "
            "remedied within 30 days.",
            "label": "termination",
            "options": CLAUSE_TYPES,
        },
        {
            "text": "Each party shall keep confidential all information disclosed to it by "
            "the other party and shall not disclose it to any third party without prior "
            "written consent.",
            "label": "confidentiality",
            "options": CLAUSE_TYPES,
        },
        {
            "text": "Neither party shall be liable for any indirect or consequential loss, "
            "and each party's total liability shall not exceed the Charges paid in the "
            "preceding 12 months.",
            "label": "limitation of liability",
            "options": CLAUSE_TYPES,
        },
        {
            "text": "This Agreement shall be governed by and construed in accordance with "
            "the law of England and Wales.",
            "label": "governing law",
            "options": CLAUSE_TYPES,
        },
        {
            "text": "Neither party shall be in breach of this Agreement if it is prevented "
            "from performing its obligations by events beyond its reasonable control.",
            "label": "force majeure",
            "options": CLAUSE_TYPES,
        },
    ],
}


def get_sample_data(task: str) -> list[dict[str, Any]]:
    """Return a copy of the built-in samples for a task (empty if there are none)."""
    return deepcopy(SAMPLE_DATA.get(task, []))
