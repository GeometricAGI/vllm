# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay GLM-5.3's representative production workload against one engine.

The shape is production's Sat 2026-09-19 02:00-10:00 UTC window, one
customer keeping the 64-request gateway cap full with long-context requests:

- prompt lengths drawn from the observed bins (median ~70k tokens),
- output lengths drawn from the observed bins (median ~200, mean ~1k),
- multi-turn sessions, so most of each prompt is a previous turn's context
  (production computed ~29% of submitted prompt tokens),
- a closed loop of --concurrency requests in flight.

Prompts are token IDs sent to /v1/completions, so lengths are exact, and a
fixed seed makes every run send the same request sequence, which is what an
A/B between two engine configurations needs. Run it next to the engine:

    OPENAI_API_KEY=... python glm_workload_replay.py \
        --base-url http://127.0.0.1:30000 --num-requests 600 --out run.json
"""

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time

import aiohttp

# (low, high, share) from vllm:request_prompt_tokens / generation_tokens.
PROMPT_BINS = [
    (10_000, 20_000, 4.4),
    (20_000, 50_000, 17.6),
    (50_000, 100_000, 49.6),
    (100_000, 200_000, 28.3),
]
OUTPUT_BINS = [
    (10, 20, 2.2),
    (20, 50, 12.6),
    (50, 100, 16.4),
    (100, 200, 17.1),
    (200, 500, 21.9),
    (500, 1_000, 11.8),
    (1_000, 2_000, 7.1),
    (2_000, 5_000, 5.9),
    (5_000, 10_000, 3.4),
    (10_000, 20_000, 1.2),
    (20_000, 50_000, 0.4),
    (50_000, 100_000, 0.1),
]
# Token IDs well inside GLM-5.3's vocabulary, clear of special tokens.
TOKEN_LOW, TOKEN_HIGH = 1_000, 150_000


def draw(rng: random.Random, bins, cap: int | None = None) -> int:
    low, high, _ = rng.choices(bins, weights=[b[2] for b in bins])[0]
    n = int(math.exp(rng.uniform(math.log(low), math.log(high))))
    return min(n, cap) if cap else n


def build_requests(args) -> list[dict]:
    """The request sequence: each request continues a session or starts one.

    A continued session's prompt is its previous prompt plus the previous
    turn's output length in new tokens (standing in for the reply) plus new
    user tokens, up to the drawn prompt length, so its prefix is reusable.
    """
    rng = random.Random(args.seed)
    sessions: list[list[int]] = []
    reqs, reused = [], 0
    for i in range(args.num_requests):
        target = draw(rng, PROMPT_BINS)
        out_len = draw(rng, OUTPUT_BINS, args.max_output)
        base = None
        if sessions and rng.random() < args.continue_prob:
            fits = [s for s in sessions if len(s) < target]
            if fits:
                # The longest conversation that fits: turns grow, as in
                # production, rather than restarting from a short one.
                base = max(fits, key=len)
        prefix = list(base) if base is not None else []
        new = [
            rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(target - len(prefix))
        ]
        prompt = prefix + new
        reused += len(prefix)
        if base is not None:
            sessions.remove(base)
        # The next turn of this session sees this turn's output as context.
        sessions.append(
            prompt + [rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(out_len)]
        )
        sessions = sessions[-args.max_sessions :]
        reqs.append({"id": i, "prompt": prompt, "max_tokens": out_len})
    total = sum(len(r["prompt"]) for r in reqs)
    print(
        f"{len(reqs)} requests, {total / len(reqs):.0f} prompt tokens on average, "
        f"{reused / total * 100:.0f}% of prompt tokens reuse a previous turn, "
        f"{statistics.mean(r['max_tokens'] for r in reqs):.0f} output tokens on average"
    )
    return reqs


async def send(session, url, headers, model, req) -> dict:
    body = {
        "model": model,
        "prompt": req["prompt"],
        "max_tokens": req["max_tokens"],
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = {"id": req["id"], "prompt_tokens": len(req["prompt"])}
    start = time.perf_counter()
    rec["start"] = time.time()
    chunk_times = []
    try:
        async with session.post(url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                data = json.loads(line[5:])
                if data.get("choices") and data["choices"][0].get("text"):
                    chunk_times.append(time.perf_counter())
                if data.get("usage"):
                    rec["output_tokens"] = data["usage"]["completion_tokens"]
    except Exception as e:  # recorded, not fatal: a failed request is data
        rec["error"] = repr(e)
    end = time.perf_counter()
    rec["e2e"] = end - start
    if chunk_times:
        rec["ttft"] = chunk_times[0] - start
        n = rec.get("output_tokens", len(chunk_times))
        if n > 1:
            rec["tpot"] = (end - chunk_times[0]) / (n - 1)
        rec["itls"] = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    return rec


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


async def main_async(args) -> None:
    reqs = build_requests(args)
    url = args.base_url.rstrip("/") + "/v1/completions"
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    queue: asyncio.Queue = asyncio.Queue()
    for r in reqs:
        queue.put_nowait(r)
    records = []
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.request_timeout)

    async def worker():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not queue.empty() and time.time() - t0 < args.max_duration:
                req = queue.get_nowait()
                records.append(await send(session, url, headers, args.model, req))
                done = len(records)
                if done % 25 == 0:
                    print(f"{done}/{len(reqs)} done, {time.time() - t0:.0f}s")

    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall = time.time() - t0

    # Steady state: drop requests that started in the warm-up.
    ok = [r for r in records if "error" not in r and r["start"] - t0 >= args.warmup]
    itls = [x for r in ok for x in r.get("itls", [])]
    summary = {
        "requests_sent": len(records),
        "errors": sum("error" in r for r in records),
        "steady_state_requests": len(ok),
        "wall_s": wall,
        "request_throughput_per_min": len(records) / wall * 60,
        "output_tokens_per_s": sum(r.get("output_tokens", 0) for r in records) / wall,
        "prompt_tokens_per_s": sum(r["prompt_tokens"] for r in records) / wall,
    }
    for key in ("ttft", "tpot", "e2e"):
        vals = [r[key] for r in ok if key in r]
        for p in (50, 90, 99):
            summary[f"{key}_p{p}"] = pct(vals, p)
        summary[f"{key}_mean"] = statistics.mean(vals) if vals else float("nan")
    for p in (50, 90, 99):
        summary[f"itl_p{p}"] = pct(itls, p)
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "summary": summary,
                    "records": [
                        {k: v for k, v in r.items() if k != "itls"} for r in records
                    ],
                },
                f,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--num-requests", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--continue-prob",
        type=float,
        default=0.75,
        help="chance a request continues an earlier session",
    )
    parser.add_argument("--max-sessions", type=int, default=256)
    parser.add_argument(
        "--max-output",
        type=int,
        default=20_000,
        help="cap on drawn output lengths, to bound run time",
    )
    parser.add_argument(
        "--warmup", type=float, default=300, help="seconds excluded from latencies"
    )
    parser.add_argument("--max-duration", type=float, default=3 * 3600)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--out")
    parser.add_argument(
        "--dry-run", action="store_true", help="only print the workload's shape"
    )
    args = parser.parse_args()
    if args.dry_run:
        build_requests(args)
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
