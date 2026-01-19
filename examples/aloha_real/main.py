import dataclasses
import logging
import sys
import threading

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import runtime as _runtime
import tyro

from examples.aloha_real import env as _env


class PromptState:
    def __init__(self, prompt: str | None) -> None:
        self._prompt = prompt
        self._version = 0
        self._lock = threading.Lock()

    def get(self) -> str | None:
        with self._lock:
            return self._prompt

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def set(self, prompt: str) -> None:
        with self._lock:
            if self._prompt == prompt:
                return
            self._prompt = prompt
            self._version += 1


class PromptSwitchingAgent(_agent.Agent):
    """Injects a prompt into observations before policy inference."""

    def __init__(self, policy, prompt_state: PromptState) -> None:
        self._policy = policy
        self._prompt_state = prompt_state
        self._last_prompt_version = prompt_state.get_version()

    def get_action(self, observation: dict) -> dict:
        prompt_version = self._prompt_state.get_version()
        if prompt_version != self._last_prompt_version:
            # Drop any cached action chunk so the next inference uses the new prompt.
            self._policy.reset()
            self._last_prompt_version = prompt_version

        prompt = self._prompt_state.get()
        if prompt:
            observation = dict(observation)
            observation["prompt"] = prompt
        return self._policy.infer(observation)

    def reset(self) -> None:
        self._policy.reset()


def _start_prompt_reader(prompt_state: PromptState) -> None:
    if not sys.stdin.isatty():
        logging.info("stdin is not a TTY; prompt switching disabled.")
        return

    def _reader() -> None:
        logging.info("Type a new prompt and press Enter to switch tasks.")
        while True:
            try:
                line = input()
            except EOFError:
                return
            prompt = line.strip()
            if not prompt:
                continue
            prompt_state.set(prompt)
            logging.info("Updated prompt: %s", prompt)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000
    prompt: str | None = None


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    prompt_state = PromptState(args.prompt)
    _start_prompt_reader(prompt_state)

    metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(     
        environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        agent=PromptSwitchingAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            ),
            prompt_state=prompt_state,
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
