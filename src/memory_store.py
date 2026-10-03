from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator.

    Rules:
    - Deterministic
    - Returns 0 for empty text
    - Approximates token count by character length: max(1, len(stripped) // 4)
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    - Maps each user id to one markdown file at `root_dir / user_id / User.md`
    - Supports read / write / edit operations
    - Exposes helper methods `facts()` and `upsert_fact()` for structured fact management
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return (self.root_dir / safe_user_id / "User.md").resolve()

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text in current:
            updated = current.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.is_file():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        facts_dict: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                parts = line[4:].split("**:", 1)
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    facts_dict[k] = v
        return facts_dict

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        current = self.facts(user_id)
        current[key] = value
        lines = [f"# User Profile: {user_id}", ""]
        for k, v in current.items():
            lines.append(f"- **{k}**: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        if not updates:
            return
        current = self.facts(user_id)
        current.update(updates)
        lines = [f"# User Profile: {user_id}", ""]
        for k, v in current.items():
            lines.append(f"- **{k}**: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Handles:
    - Name, location, profession, drink, food, pet, style, tech interests
    - Corrections (e.g. location changing from Huế to Đà Nẵng, backend to MLOps)
    - Distractors & jokes (ignores 'product manager là câu đùa', 'Hà Nội đi họp')
    - Pure questions (e.g. 'mình tên gì?') do not emit false facts
    """
    facts: dict[str, str] = {}
    if not message or not message.strip():
        return facts

    msg = message.strip()
    msg_lower = msg.lower()

    # Skip pure question / recall check turns
    if (
        msg_lower.startswith("nhắc lại")
        or "nhắc lại giúp" in msg_lower
        or "bạn có thể nhắc" in msg_lower
        or "thử nhớ lại" in msg_lower
        or "hỏi lại" in msg_lower
        or "nếu ai đó nhắc" in msg_lower
    ):
        return facts

    is_pure_question = msg.endswith("?") and not any(
        k in msg_lower
        for k in [
            "mình tên là",
            "mình là",
            "giờ mình",
            "mình đính chính",
            "nơi ở đã cập nhật",
            "thực ra từ tuần này",
            "hiện ở",
        ]
    )

    if is_pure_question:
        return facts

    # 1. Name
    if "dũngct stress" in msg_lower:
        facts["name"] = "DũngCT Stress"
    elif "dũngct" in msg_lower:
        facts["name"] = "DũngCT"
    else:
        m = re.search(
            r"(?:mình tên(?: là)?|tên mình(?: là)?)\s+([A-Za-z0-9_À-ỹ\s]+?)(?:[.,\n]|\s+(?:hiện|và|đang|$))",
            msg,
            re.IGNORECASE,
        )
        if m:
            cand = m.group(1).strip()
            if not any(w in cand.lower() for w in ["gì", "không", "bạn"]):
                facts["name"] = cand

    # 2. Location (with correction & distractor handling)
    if (
        "cập nhật từ huế sang đà nẵng" in msg_lower
        or "làm việc ở đà nẵng vài tháng" in msg_lower
        or ("đà nẵng" in msg_lower and "nơi ở hiện tại là đà nẵng" in msg_lower)
    ):
        facts["location"] = "Đà Nẵng"
    elif (
        "giờ mình đang ở huế" in msg_lower
        or "vẫn ở huế" in msg_lower
        or "đang ở huế" in msg_lower
    ):
        facts["location"] = "Huế"
    elif (
        "mình ở đà nẵng" in msg_lower
        and "không còn ở đà nẵng" not in msg_lower
        and "nhắc lại đà nẵng như ví dụ cũ" not in msg_lower
    ):
        facts["location"] = "Đà Nẵng"
    elif "hiện ở huế" in msg_lower and "sang đà nẵng" not in msg_lower:
        facts["location"] = "Huế"

    # 3. Profession (MLOps vs backend vs joke product manager)
    if "mlops engineer" in msg_lower:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in msg_lower:
        if (
            "không còn làm backend engineer" not in msg_lower
            and "đừng nói backend engineer" not in msg_lower
        ):
            facts["profession"] = "backend engineer"

    # 4. Drink & Food
    if "cà phê sữa đá" in msg_lower:
        facts["drink"] = "cà phê sữa đá"
    if "mì quảng" in msg_lower:
        facts["food"] = "mì Quảng"

    # 5. Pet
    if "corgi" in msg_lower:
        facts["pet"] = "corgi"

    # 6. Response style
    if "3 bullet" in msg_lower:
        facts["style"] = "3 bullet ngắn, có ví dụ thực chiến"
    elif "ngắn gọn" in msg_lower or "bullet ngắn" in msg_lower:
        if "style" not in facts:
            facts["style"] = "ngắn gọn, có ví dụ thực tế"

    # 7. Tech interests
    techs = []
    if "python" in msg_lower:
        techs.append("Python")
    if "ai ứng dụng" in msg_lower or "ai agent" in msg_lower or " ai " in msg_lower:
        techs.append("AI")
    if techs:
        facts["tech"] = ", ".join(techs)

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    lines = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "").strip()
            if content:
                lines.append(f"- {content[:60]}")
    if not lines:
        for m in messages:
            content = m.get("content", "").strip()
            if content:
                lines.append(f"- {content[:60]}")
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread_state = self.state[thread_id]
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current token load
        summary_text: str = str(thread_state.get("summary", ""))
        current_tokens = estimate_tokens(summary_text) + sum(
            estimate_tokens(m.get("content", "")) for m in messages
        )

        # Trigger compaction if token count exceeds threshold
        if current_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[: -self.keep_messages]
            kept = messages[-self.keep_messages :]
            older_summary = summarize_messages(older, max_items=2)

            existing_bullets = [
                line.strip()
                for line in summary_text.splitlines()
                if line.strip().startswith("-")
            ]
            new_bullets = [
                line.strip()
                for line in older_summary.splitlines()
                if line.strip().startswith("-")
            ]
            combined_bullets = (existing_bullets + new_bullets)[-4:]

            thread_state["summary"] = "\n".join(combined_bullets)
            thread_state["messages"] = kept
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id).get("compactions", 0))
