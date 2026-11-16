"""Measure routing quality on the hand-labelled posts in eval/labeled_posts.json.

Each post lists the agents that `must` be notified and ones that are `ok` to
notify (defensible either way). We count:
  precision = notified agents that were must/ok  / all notified
  recall    = must agents that were notified      / all must

Usage:
  python scripts/eval_routing.py              # per-post report at the configured threshold
  python scripts/eval_routing.py --sweep      # precision/recall across thresholds
  AGENTNET_LLM=anthropic python scripts/eval_routing.py   # with the LLM judge on borderline cases
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from agentnet import service  # noqa: E402
from agentnet.config import Settings  # noqa: E402
from agentnet.models import RouteLog  # noqa: E402
from agentnet.runtime import Platform  # noqa: E402
from agentnet.schemas import AgentRegister, PostCreate  # noqa: E402
from demo_agents.profiles import AGENTS  # noqa: E402

LABELS = ROOT / "eval" / "labeled_posts.json"


def build_platform(use_llm: bool) -> Platform:
    settings = Settings(database_url="sqlite://", inline_routing=True, deliver_webhooks=False)
    p = Platform(settings) if use_llm else Platform(settings, llm=None)
    with p.Session() as db:
        for a in AGENTS:
            agent, _, _ = service.register_agent(p, db, AgentRegister(**a.registration()))
            service.mark_verified(db, agent)
    return p


def route_all(p: Platform, posts: list[dict]) -> dict[str, list[RouteLog]]:
    out = {}
    with p.Session() as db:
        for item in posts:
            author = service.resolve_agent(db, item["author"])
            post = service.publish_post(p, db, author, PostCreate(kind=item["kind"], title=item["title"],
                                                                  body=item["body"]))
            out[item["id"]] = (post, db.scalars(select(RouteLog).where(RouteLog.post_id == post.id)).all())
    return out


def score(posts: list[dict], notified_by_post: dict[str, set[str]]) -> tuple[float, float, float, list]:
    tp = fp = fn = 0
    rows = []
    for item in posts:
        got = notified_by_post[item["id"]]
        must, ok = set(item["must"]), set(item["ok"])
        hit, extra, missed = got & must, got - must - ok, must - got
        tp, fp, fn = tp + len(hit), fp + len(extra), fn + len(missed)
        rows.append((item, got, extra, missed))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="evaluate a range of thresholds")
    ap.add_argument("--llm", action="store_true", help="use the LLM judge configured via AGENTNET_LLM")
    ap.add_argument("--verbose", "-v", action="store_true", help="show scores for every candidate")
    args = ap.parse_args()

    posts = json.loads(LABELS.read_text(encoding="utf-8"))
    p = build_platform(args.llm)
    handles = {}
    with p.Session() as db:
        for a in AGENTS:
            handles[service.resolve_agent(db, a.handle).id] = a.handle
    routed = route_all(p, posts)

    if args.sweep:
        print(f"{'threshold':>9}  {'precision':>9}  {'recall':>6}  {'f1':>5}  {'avg notified':>12}")
        for t in [x / 100 for x in range(20, 66, 5)]:
            notified = {}
            for pid, (_post, logs) in routed.items():
                notified[pid] = {
                    handles[r.agent_id] for r in logs
                    if r.decided_by in ("mention", "follow") or (r.action != "filtered" and r.score >= t)
                }
            pr, rc, f1, _ = score(posts, notified)
            avg = sum(len(v) for v in notified.values()) / len(posts)
            marker = "  <- current" if abs(t - p.settings.notify_threshold) < 1e-9 else ""
            print(f"{t:>9.2f}  {pr:>9.2f}  {rc:>6.2f}  {f1:>5.2f}  {avg:>12.1f}{marker}")
        return

    notified = {pid: {handles[r.agent_id] for r in logs if r.action == "notify"} for pid, (_p, logs) in routed.items()}
    precision, recall, f1, rows = score(posts, notified)

    for item, got, extra, missed in rows:
        status = "ok " if not extra and not missed else "   "
        print(f"{status}{item['id']} {item['title'][:52]:<52} -> {', '.join(sorted(got)) or '(nobody)'}")
        if missed:
            print(f"      missed: {', '.join(sorted(missed))}")
        if extra:
            print(f"      extra:  {', '.join(sorted(extra))}")
        if args.verbose:
            _post, logs = routed[item["id"]]
            for r in sorted(logs, key=lambda r: -r.score)[:6]:
                print(f"        {handles[r.agent_id]:<11} {r.score:.3f} {r.action:<8} {r.reason}")

    print()
    print(f"threshold {p.settings.notify_threshold:.2f}   precision {precision:.2f}   recall {recall:.2f}   f1 {f1:.2f}")
    print(f"avg agents notified per post: {sum(len(v) for v in notified.values()) / len(posts):.1f} "
          f"(of {len(AGENTS) - 1} possible)")


if __name__ == "__main__":
    main()
