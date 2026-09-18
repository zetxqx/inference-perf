"""Worker permits must bound in-flight requests, not queued session events.

Session replay enqueues every event of a session at dispatch, pinned to one
worker, and events run in order behind their predecessors. If a parked event
held a worker_max_concurrency permit, a couple of long sessions would pin all
permits and the worker could not even read the first turn of the next session
(effective concurrency ~= permits / events_per_session, see upstream #648).
"""

import asyncio
import os
import time
from typing import Generator, List, Optional, Tuple

from inference_perf.apis import ChatCompletionAPIData, ChatMessage, InferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.datagen import MockDataGenerator
from test_stage_teardown import _fork_start_method, _Harness, _TestClientBase, ChainedChatData  # noqa: F401

PREDECESSOR_SECONDS = 3.0


class TwoSessionDataGenerator(MockDataGenerator):
    """Session A: a1 (slow) then a2 (parked on a1). Session B: b1 (plain).

    With worker_max_concurrency=2 and this queue order, a1 and a2 consume both
    permits if parked events hold them, and b1 is not read until a1 finishes.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, a1_done_flag: str) -> None:
        super().__init__(api_config, config, None)
        self.a1_done_flag = a1_done_flag

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        yield ChainedChatData(messages=[ChatMessage(role="user", content="a1")])
        yield ChainedChatData(messages=[ChatMessage(role="user", content="a2")], wait_flag_path=self.a1_done_flag)
        yield ChainedChatData(messages=[ChatMessage(role="user", content="b1")])
        while True:
            yield ChainedChatData(messages=[ChatMessage(role="user", content="filler")])


class TimestampingClient(_TestClientBase):
    """Records dispatch time per request; a1 is slow and raises a2's flag."""

    def __init__(self, dispatch_log: str, a1_done_flag: str) -> None:
        super().__init__()
        self.dispatch_log = dispatch_log
        self.a1_done_flag = a1_done_flag

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        assert isinstance(data, ChatCompletionAPIData)
        name = data.messages[0].content
        with open(self.dispatch_log, "a") as f:
            f.write(f"{name} {time.perf_counter():.3f}\n")
        if name == "a1":
            await asyncio.sleep(PREDECESSOR_SECONDS)
            with open(self.a1_done_flag, "w") as f:
                f.write("done\n")
        else:
            await asyncio.sleep(0.1)


def _read_dispatches(path: str) -> List[Tuple[str, float]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(line.split()[0], float(line.split()[1])) for line in f if line.strip()]


async def test_parked_events_do_not_hold_worker_permits(tmp_path: object) -> None:
    dispatch_log = os.path.join(str(tmp_path), "dispatch.log")
    a1_done_flag = os.path.join(str(tmp_path), "a1.done")
    api_config = APIConfig(type=APIType.Chat)
    datagen = TwoSessionDataGenerator(api_config, DataConfig(type=DataGenType.Mock), a1_done_flag)
    harness = _Harness(
        TimestampingClient(dispatch_log, a1_done_flag),
        teardown_grace_seconds=15.0,
        datagen=datagen,
        worker_max_concurrency=2,
    )
    try:
        # rate=3, duration=1: a1, a2, b1 are enqueued within the first second.
        await harness.run_stage(0, timeout=20.0, rate=3)

        dispatched = dict(_read_dispatches(dispatch_log))
        assert set(dispatched) == {"a1", "a2", "b1"}, dispatched
        assert harness.finished_counter.value == 3

        # b1 belongs to another session and must go out while a2 is still
        # parked on a1, i.e. well before a1 completes.
        b1_delay = dispatched["b1"] - dispatched["a1"]
        assert b1_delay < PREDECESSOR_SECONDS - 1.0, f"b1 waited {b1_delay:.2f}s behind a parked event"
        assert dispatched["a2"] - dispatched["a1"] >= PREDECESSOR_SECONDS - 0.5
    finally:
        harness.shutdown()
