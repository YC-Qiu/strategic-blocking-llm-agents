"""CPU-only prompt and parser port from the captured installed RAGEN source.

The five private methods below retain the source behavior, including first-match
parsing and truncation to one action. See evaluation/colab-ragen-contract-v2.json.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "924f16d1ff1764a2f4ac77e008d08ff2e9f458d13cdaff1de4edcb7dee2e346f"


def read_protocol():
    raw = (ROOT / "evaluation/colab-protocol-v2.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen v2 protocol changed; create a new version.")
    protocol = json.loads(raw)
    contract = (ROOT / "evaluation/colab-ragen-contract-v2.json").read_bytes()
    if hashlib.sha256(contract).hexdigest() != protocol["contract_sha256"]:
        raise ValueError("Captured RAGEN contract changed.")
    return protocol


class RagenTextProtocol:
    def __init__(self, protocol):
        self.protocol = protocol
        self.config = SimpleNamespace(agent_proxy=SimpleNamespace(
            enable_think=False, max_actions_per_turn=1))
        self.action_sep = protocol["action_sep"]
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>",
                                   "<|im_start|>", "<|im_end|>"]
        self.prefix_lookup = {0: protocol["instruction"]}
        self.env_config_lookup = {0: {"max_tokens": protocol["prompt_response_words"]}}

    def messages(self, cache):
        window = self.protocol["history_turns_including_current"]
        offset = max(0, len(cache) - window)
        return self._build_infer_multi_turn_messages({"env_id": 0}, cache[offset:], offset)

    def prompt(self, tokenizer, messages):
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        ) + self.protocol["assistant_prefix"]

    def parse_completion(self, completion):
        response = self.protocol["assistant_prefix"] + completion
        normalized, actions = self._parse_response(response)
        return response, normalized, actions

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)


            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions

    def _build_format_prompt(self, env_id: int) -> Tuple[str, str]:
        """Build FORMAT_PROMPT and LENGTH_PROMPT for an environment."""
        FORMAT_PROMPT = (
            "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
            if self.config.agent_proxy.enable_think
            else "<answer> [your answer] </answer>"
        )
        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_id]['max_tokens']} words (tokens)."
        return FORMAT_PROMPT, LENGTH_PROMPT

    def _build_system_content(self, env_id: int) -> str:
        env_instruction = self.prefix_lookup.get(env_id, "")
        if env_instruction:
            return "You're a helpful assistant. " + env_instruction
        return "You're a helpful assistant. "

    def _build_turn_state_content(
        self,
        turn: Dict,
        turn_number: int,
        env_id: int,
        include_warning: bool = False,
    ) -> str:
        """Build state content for a single turn."""
        FORMAT_PROMPT, LENGTH_PROMPT = self._build_format_prompt(env_id)
        warning = ""
        if include_warning and turn.get('manager_invalid_action'):
            warning = "No valid action provided previously. Environment state remains the same. Please try again.\n"

        content = f"\nTurn {turn_number}:\n"
        content += (
            f"State:\n{turn['state']}\n{warning}"
            f"You have {turn['actions_left']} actions left. Always output: {FORMAT_PROMPT} "
            f"with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
        )
        return content

    def _build_infer_multi_turn_messages(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_offset: int,
    ) -> List[Dict]:
        """Build messages for multi-turn inference (full or limited_multi_turn)."""
        messages = [
            {"role": "system", "content": self._build_system_content(env_output["env_id"])},
            {"role": "user", "content": ""}
        ]

        for idx, content in enumerate(history):
            actual_turn = idx + 1 + turn_offset
            if "state" in content:
                messages[-1]["content"] += self._build_turn_state_content(
                    content, actual_turn, env_output["env_id"], include_warning=True
                )
            if "llm_response" in content:
                messages.append({"role": "assistant", "content": content["llm_response"]})
            if "reward" in content:
                messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

        return messages
