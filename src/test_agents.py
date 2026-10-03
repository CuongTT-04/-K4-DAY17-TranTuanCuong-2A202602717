from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,  # small threshold so compaction triggers early
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    profile_store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # 1. Non-existent initially returns empty string and 0 size
    assert profile_store.read_text(user_id) == ""
    assert profile_store.file_size(user_id) == 0

    # 2. Write profile content
    initial_content = (
        "# User Profile\n- **location**: Đà Nẵng\n- **profession**: backend engineer"
    )
    written_path = profile_store.write_text(user_id, initial_content)
    assert written_path.exists()
    assert profile_store.read_text(user_id) == initial_content
    assert profile_store.file_size(user_id) > 0

    # 3. Edit text (correction scenario)
    edited = profile_store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert edited is True
    updated_content = profile_store.read_text(user_id)
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    cfg = make_config(tmp_path)
    compact_mgr = CompactMemoryManager(
        threshold_tokens=cfg.compact_threshold_tokens,
        keep_messages=cfg.compact_keep_messages,
    )
    thread_id = "thread_compact_test"

    assert compact_mgr.compaction_count(thread_id) == 0

    # Push messages that exceed 80 tokens
    for i in range(8):
        compact_mgr.append(
            thread_id,
            "user",
            f"Tin tức chi tiết số {i} với độ dài văn bản tương đối lớn để nhanh chóng vượt qua ngưỡng 80 tokens của bài kiểm tra.",
        )

    assert compact_mgr.compaction_count(thread_id) > 0
    ctx = compact_mgr.context(thread_id)
    assert len(ctx["messages"]) <= cfg.compact_keep_messages + 1
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)

    # Initialize both agents with isolated config
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "dungct_test"

    # Session 1: Provide facts
    intro_msg = "Chào bạn, mình tên là DũngCT và mình ở Đà Nẵng."
    baseline.reply(user_id=user_id, thread_id="thread_session_1", message=intro_msg)
    advanced.reply(user_id=user_id, thread_id="thread_session_1", message=intro_msg)

    # Session 2: Recall in a fresh thread
    recall_question = "Bạn có thể nhắc lại tên mình không?"
    baseline_reply = baseline.reply(
        user_id=user_id, thread_id="thread_session_2", message=recall_question
    )
    advanced_reply = advanced.reply(
        user_id=user_id, thread_id="thread_session_2", message=recall_question
    )

    # Advanced MUST recall the name
    assert "DũngCT" in advanced_reply["response"], (
        f"Advanced agent should remember user name across sessions. Got: {advanced_reply['response']}"
    )

    # Baseline MUST NOT recall the name
    assert "DũngCT" not in baseline_reply["response"], (
        f"Baseline agent must NOT remember user name across sessions. Got: {baseline_reply['response']}"
    )


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "stress_test_user"
    thread_id = "long_thread_test"

    long_messages = [
        f"Lượt thảo luận số {i}: Chúng ta cùng phân tích chi tiết về kiến trúc memory systems cho AI Agent, "
        f"cách thức hoạt động của short term memory, persistent User.md và compact memory khi hội thoại kéo dài."
        for i in range(12)
    ]

    for msg in long_messages:
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced must have triggered compaction
    assert advanced.compaction_count(thread_id) > 0, "Compaction must have triggered for Advanced"
    # Baseline compaction count must remain 0
    assert baseline.compaction_count(thread_id) == 0

    # On a long thread, Advanced's prompt tokens processed must be strictly lower than Baseline
    assert advanced_prompt_tokens < baseline_prompt_tokens, (
        f"Advanced prompt tokens ({advanced_prompt_tokens}) should be strictly less than "
        f"Baseline ({baseline_prompt_tokens}) on a long thread."
    )
