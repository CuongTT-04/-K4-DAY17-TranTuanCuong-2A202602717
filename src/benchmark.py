from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode."""
    if not answer or not answer.strip():
        return 0.0
    points = recall_points(answer, expected)
    length = len(answer.strip())
    structure_score = 1.0 if 20 <= length <= 600 else 0.5
    return round(0.7 * points + 0.3 * structure_score, 2)


def run_agent_benchmark(
    agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: Any
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """
    unique_users = set(conv["user_id"] for conv in conversations)
    # Ensure fresh state for this benchmark suite run
    if hasattr(agent, "profile_store"):
        for uid in unique_users:
            p = agent.profile_store.path_for(uid)
            if p.exists():
                p.unlink()

    all_threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv.get("turns", [])
        recall_questions = conv.get("recall_questions", [])

        # 1. Feed conversation turns in the conversation's thread
        all_threads.append(conv_id)
        for turn in turns:
            agent.reply(user_id=user_id, thread_id=conv_id, message=turn)

        # 2. Ask recall questions in fresh threads (cross-session evaluation)
        for idx, rq in enumerate(recall_questions):
            recall_thread_id = f"{conv_id}-recall-{idx}"
            all_threads.append(recall_thread_id)
            question = rq["question"]
            expected = rq["expected_contains"]

            reply_dict = agent.reply(
                user_id=user_id, thread_id=recall_thread_id, message=question
            )
            ans = reply_dict.get("response", "")

            r_pt = recall_points(ans, expected)
            q_pt = heuristic_quality(ans, expected)
            recall_scores.append(r_pt)
            quality_scores.append(q_pt)

    # Aggregate token statistics
    agent_tokens = sum(agent.token_usage(t) for t in all_threads)
    prompt_tokens = sum(agent.prompt_token_usage(t) for t in all_threads)
    avg_recall = (
        round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0
    )
    avg_quality = (
        round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0
    )

    # Memory growth in bytes
    if hasattr(agent, "memory_file_size"):
        memory_bytes = sum(agent.memory_file_size(uid) for uid in unique_users)
    else:
        memory_bytes = 0

    # Total compactions
    compactions = sum(agent.compaction_count(t) for t in all_threads)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table with exactly the 6 required columns."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites and print comparison tables."""
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)

    # 1. Standard Benchmark
    print("\n" + "=" * 70)
    print("### Standard Benchmark (data/conversations.json)")
    print("=" * 70)
    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)
    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, standard_convs, config),
        run_agent_benchmark("Advanced", advanced_std, standard_convs, config),
    ]
    print(format_rows(std_rows))

    # 2. Long-Context Stress Benchmark
    print("\n" + "=" * 70)
    print("### Long-Context Stress Benchmark (data/advanced_long_context.json)")
    print("=" * 70)
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)
    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))
    print()


if __name__ == "__main__":
    main()
