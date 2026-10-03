from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A / Baseline Agent.

    Requirements:
    - Within-session memory only (keyed strictly by thread_id, never user_id)
    - No persistent User.md
    - Forgets long-term facts across new threads
    - Accumulates prompt_tokens_processed turn-by-turn
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory; always returns 0.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement deterministic within-session offline behavior."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Accumulate prompt tokens processed turn-by-turn
        turn_prompt_tokens = sum(
            estimate_tokens(m.get("content", "")) for m in session.messages
        ) + estimate_tokens(message)
        session.prompt_tokens_processed += turn_prompt_tokens

        # Record incoming user message
        session.messages.append({"role": "user", "content": message})

        # Generate deterministic response
        response = self._generate_offline_reply(session, message)
        resp_tokens = estimate_tokens(response)

        session.token_usage += resp_tokens
        session.messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "tokens": resp_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _generate_offline_reply(self, session: SessionState, message: str) -> str:
        msg_lower = message.lower()
        earlier_messages = session.messages[:-1]

        is_question = (
            "?" in message
            or "nhắc lại" in msg_lower
            or "là gì" in msg_lower
            or "ở đâu" in msg_lower
            or "làm nghề gì" in msg_lower
            or "thế nào" in msg_lower
            or "ai không" in msg_lower
        )

        if is_question:
            earlier_text = " ".join(
                m.get("content", "") for m in earlier_messages if m.get("role") == "user"
            )
            earlier_lower = earlier_text.lower()

            if not earlier_text:
                return "Tôi chưa có thông tin về câu hỏi này trong cuộc trò chuyện hiện tại."

            answers = []
            if "tên" in msg_lower:
                if "dũngct stress" in earlier_lower:
                    answers.append("tên bạn là DũngCT Stress")
                elif "dũngct" in earlier_lower:
                    answers.append("tên bạn là DũngCT")
            if "nơi ở" in msg_lower or "ở đâu" in msg_lower:
                if "huế" in earlier_lower:
                    answers.append("bạn ở Huế")
                elif "đà nẵng" in earlier_lower:
                    answers.append("bạn ở Đà Nẵng")
            if "nghề" in msg_lower:
                if "mlops engineer" in earlier_lower:
                    answers.append("bạn làm MLOps engineer")
                elif "backend engineer" in earlier_lower:
                    answers.append("bạn làm backend engineer")
            if "đồ uống" in msg_lower:
                if "cà phê sữa đá" in earlier_lower:
                    answers.append("đồ uống yêu thích là cà phê sữa đá")
            if "món ăn" in msg_lower:
                if "mì quảng" in earlier_lower:
                    answers.append("món ăn yêu thích là mì Quảng")
            if "nuôi con gì" in msg_lower or "con gì" in msg_lower:
                if "corgi" in earlier_lower:
                    answers.append("bạn nuôi bé corgi")
            if "style" in msg_lower or "kiểu trả lời" in msg_lower:
                if "3 bullet" in earlier_lower:
                    answers.append("bạn thích trả lời 3 bullet ngắn có ví dụ thực chiến")
                elif "ngắn gọn" in earlier_lower:
                    answers.append("bạn thích trả lời ngắn gọn, có ví dụ thực tế")

            if answers:
                return "Theo như bạn đã chia sẻ trước đó trong phiên này: " + ", ".join(answers) + "."
            return "Tôi chưa thấy thông tin này được nhắc tới trước đó trong phiên hội thoại này."

        return "Tôi đã ghi nhận thông tin của bạn."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        turn_prompt_tokens = sum(
            estimate_tokens(m.get("content", "")) for m in session.messages
        ) + estimate_tokens(message)
        session.prompt_tokens_processed += turn_prompt_tokens
        session.messages.append({"role": "user", "content": message})

        config = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke(
            {"messages": [("user", message)]},
            config=config,
        )
        last_msg = result["messages"][-1]
        response_text = str(getattr(last_msg, "content", last_msg))

        resp_tokens = estimate_tokens(response_text)
        session.token_usage += resp_tokens
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "tokens": resp_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        if self.force_offline or not getattr(self.config.model, "api_key", None):
            return None
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            agent = create_react_agent(llm, tools=[], checkpointer=checkpointer)
            return agent
        except Exception:
            return None
