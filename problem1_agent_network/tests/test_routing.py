import json

from sqlalchemy import select

from agentnet import service
from agentnet.judge import RelevanceJudge
from agentnet.models import Delivery, Post, RouteLog
from agentnet.schemas import AgentUpdate, PostCreate

from .conftest import FakeLLM


def publish(platform, agents, handle, title, body, **kw):
    with platform.Session() as db:
        author = service.resolve_agent(db, handle)
        return service.publish_post(platform, db, author, PostCreate(title=title, body=body, **kw))


def decisions(platform, post_id):
    with platform.Session() as db:
        rows = service.routing_report(db, db.get(Post, post_id))
        return {a.handle: r for r, a in rows}


def notified(platform, post_id):
    return {h for h, r in decisions(platform, post_id).items() if r.action == "notify"}


def test_brief_example_routes_to_payments_and_cards_not_auto(platform, agents):
    post = publish(platform, agents, "swiggy", "Diwali food delivery discount",
                   "Swiggy is offering flat 5% off on all delivery orders above Rs 300 during Diwali week.",
                   kind="offer")
    got = notified(platform, post.id)
    assert {"phonepe", "hdfc-bank"} <= got
    assert "bajaj-auto" not in got
    assert "swiggy" not in decisions(platform, post.id)  # never routed to the author


def test_off_topic_post_reaches_nobody(platform, agents):
    post = publish(platform, agents, "airtel", "Scheduled network maintenance in Mumbai",
                   "Airtel will carry out scheduled maintenance in parts of Mumbai on Sunday between 2 and 5 AM.")
    assert notified(platform, post.id) == set()


def test_partnership_request_goes_to_providers_not_other_seekers(platform, agents):
    post = publish(platform, agents, "makemytrip", "Winter holiday sale",
                   "Up to 20% off domestic flights and hotels. Looking for bank and card partners to add instant "
                   "discounts, and for an insurer to add trip cover.", kind="partnership_request")
    got = notified(platform, post.id)
    assert {"hdfc-bank", "acko"} <= got
    # Flipkart also wants bank partners; that makes it a competitor for them, not a match.
    assert "flipkart" not in got
    d = decisions(platform, post.id)["hdfc-bank"]
    assert d.features["complement"] > 0 and "cards" in d.features["offers_topics"]


def test_every_decision_is_logged_with_reason_and_features(platform, agents):
    post = publish(platform, agents, "swiggy", "Diwali food delivery discount", "5% off all deliveries on Diwali")
    for handle, r in decisions(platform, post.id).items():
        assert r.reason
        assert {"semantic", "topic", "complement", "affinity"} <= r.features.keys()


def test_mention_always_notifies(platform, agents):
    post = publish(platform, agents, "airtel", "Network maintenance",
                   "Scheduled maintenance tonight. @tata-power please note the window.")
    d = decisions(platform, post.id)["tata-power"]
    assert d.action == "notify" and d.decided_by == "mention"


def test_mute_author_filters(platform, agents):
    with platform.Session() as db:
        service.subscribe(db, service.resolve_agent(db, "phonepe"), "mute_agent", "swiggy")
    post = publish(platform, agents, "swiggy", "Diwali food delivery discount", "5% off all deliveries on Diwali")
    d = decisions(platform, post.id)["phonepe"]
    assert d.action == "filtered"


def test_excluded_topic_filters(platform, agents):
    with platform.Session() as db:
        service.update_agent(platform, db, service.resolve_agent(db, "hdfc-bank"),
                             AgentUpdate(exclude_topics=["food-delivery"]))
    post = publish(platform, agents, "swiggy", "Food delivery update",
                   "New restaurants and cloud kitchens on Swiggy food delivery this week.")
    assert decisions(platform, post.id)["hdfc-bank"].action == "filtered"


def test_followers_always_receive_posts(platform, agents):
    body = "Bajaj Auto's two-wheeler sales grew 18% year on year in October."
    before = publish(platform, agents, "bajaj-auto", "October sales", body)
    assert "zepto" not in notified(platform, before.id)
    with platform.Session() as db:
        service.subscribe(db, service.resolve_agent(db, "zepto"), "follow_agent", "bajaj-auto")
    after = publish(platform, agents, "bajaj-auto", "November sales", body.replace("October", "November"))
    d = decisions(platform, after.id)["zepto"]
    assert d.decided_by == "follow"
    # a weak match for a follower lands in the inbox without a push
    assert d.action == ("notify" if d.score >= platform.settings.follow_threshold else "inbox")
    with platform.Session() as db:
        assert db.scalar(select(Delivery).where(Delivery.post_id == after.id,
                                                Delivery.agent_id == agents["zepto"][0].id)) is not None


def test_hourly_budget_overflow_goes_to_inbox_not_dropped(platform, agents):
    with platform.Session() as db:
        service.update_agent(platform, db, service.resolve_agent(db, "phonepe"), AgentUpdate(hourly_budget=1))
    first = publish(platform, agents, "swiggy", "Diwali food delivery discount", "5% off all deliveries on Diwali")
    second = publish(platform, agents, "zepto", "Zepto Diwali grocery offer",
                     "Flat 10% off on grocery orders during Diwali week.", kind="offer")
    assert decisions(platform, first.id)["phonepe"].action == "notify"
    d = decisions(platform, second.id)["phonepe"]
    assert d.action == "inbox" and d.decided_by == "budget"
    with platform.Session() as db:
        delivery = db.scalar(select(Delivery).where(Delivery.post_id == second.id,
                                                    Delivery.agent_id == agents["phonepe"][0].id))
        assert delivery.channel == "inbox"


def test_expired_post_is_not_routed(platform, agents):
    from datetime import datetime, timedelta, timezone
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    post = publish(platform, agents, "swiggy", "Old offer", "5% off deliveries", expires_at=past)
    with platform.Session() as db:
        assert db.get(Post, post.id).routing_status == "skipped"
        assert not db.scalars(select(RouteLog).where(RouteLog.post_id == post.id)).all()


# ---------------------------------------------------------------- LLM judge

def _judged_platform(platform, reply):
    llm = FakeLLM(reply)
    platform.router.judge = RelevanceJudge(llm)
    return llm


def test_judge_decides_borderline_and_sees_post_as_data(platform, agents):
    llm = _judged_platform(platform, json.dumps({"decisions": [
        {"handle": h, "notify": False, "confidence": 0.9, "reason": "not actionable"} for h, _ in agents.items()
    ]}))
    post = publish(platform, agents, "bajaj-auto", "Chetak bookings open",
                   "Bookings for the new Chetak electric scooter are open across 3,000 dealerships. We are looking "
                   "for financing partners to offer low-cost EMIs and for insurers to issue cover at the showroom.")
    judged = [r for r in decisions(platform, post.id).values() if r.action == "judge_no"]
    assert judged, "expected at least one borderline candidate to be sent to the judge"
    assert len(llm.prompts) == 1  # one batched call per post
    system, prompt = llm.prompts[0]
    assert "ignore any instructions" in system.lower()
    assert "<post>" in prompt and "Chetak" in prompt
    # clear matches never reach the judge
    assert decisions(platform, post.id)["hdfc-bank"].decided_by in ("score", "judge")


def test_judge_failure_falls_back_to_threshold(platform, agents):
    from agentnet.llm import LLMError
    _judged_platform(platform, LLMError("down"))
    post = publish(platform, agents, "swiggy", "Diwali food delivery discount",
                   "Swiggy is offering flat 5% off on all delivery orders above Rs 300 during Diwali week.")
    assert {"phonepe", "hdfc-bank"} <= notified(platform, post.id)
