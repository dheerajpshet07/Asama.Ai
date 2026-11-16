import json
import time
from datetime import timedelta

import httpx
from sqlalchemy import select

from agentnet import service
from agentnet.client import InvalidSignature, verify_webhook
from agentnet.delivery import CONTENT_NOTICE, sign, verify_signature
from agentnet.models import Agent, Delivery
from agentnet.schemas import PostCreate
from agentnet.util import utcnow

import pytest


def publish_brief_example(platform):
    with platform.Session() as db:
        author = service.resolve_agent(db, "swiggy")
        return service.publish_post(platform, db, author, PostCreate(
            kind="offer", title="Diwali food delivery discount",
            body="Swiggy is offering flat 5% off on all delivery orders above Rs 300 during Diwali week."))


def phonepe_delivery(platform, agents, post):
    with platform.Session() as db:
        return db.scalar(select(Delivery).where(Delivery.post_id == post.id,
                                                Delivery.agent_id == agents["phonepe"][0].id))


def test_signature_roundtrip_and_replay_protection():
    body = b'{"x":1}'
    ts = str(int(time.time()))
    sig = sign("whsec_abc", ts, body)
    assert verify_signature("whsec_abc", ts, body, sig)
    assert not verify_signature("whsec_other", ts, body, sig)
    assert not verify_signature("whsec_abc", ts, body + b" ", sig)
    old = str(int(time.time()) - 3600)
    assert not verify_signature("whsec_abc", old, body, sign("whsec_abc", old, body))


def test_webhook_payload_is_signed_and_marks_content_untrusted(platform, agents, transport):
    publish_brief_example(platform)
    url, body, headers = next(c for c in transport.calls if c[0].endswith("/phonepe"))
    event = verify_webhook(agents["phonepe"][2], headers, body)
    assert event["event"] == "post.matched"
    assert event["post"]["author"]["handle"] == "swiggy"
    assert event["content_notice"] == CONTENT_NOTICE
    assert event["match"]["reason"]
    with pytest.raises(InvalidSignature):
        verify_webhook(agents["hdfc-bank"][2], headers, body)


def test_retry_with_backoff_then_success(platform, agents, transport):
    transport.status = 503
    post = publish_brief_example(platform)
    d = phonepe_delivery(platform, agents, post)
    assert d.status == "retry" and d.attempts == 1 and d.next_attempt_at > utcnow()

    # make it due again; the endpoint has recovered
    with platform.Session() as db:
        db.get(Delivery, d.id).next_attempt_at = utcnow() - timedelta(seconds=1)
        db.commit()
    transport.status = 200
    platform.dispatcher.run_once()
    d = phonepe_delivery(platform, agents, post)
    assert d.status == "delivered" and d.attempts == 2


def test_permanent_client_error_is_not_retried(platform, agents, transport):
    transport.status = 410
    post = publish_brief_example(platform)
    d = phonepe_delivery(platform, agents, post)
    assert d.status == "failed" and d.attempts == 1
    # still available by pulling the inbox
    with platform.Session() as db:
        inbox = service.inbox(db, db.get(Agent, agents["phonepe"][0].id))
        assert [x.id for x in inbox] == [d.id]


def test_network_errors_are_retried(platform, agents, transport):
    transport.status = httpx.ConnectTimeout("timed out")
    post = publish_brief_example(platform)
    d = phonepe_delivery(platform, agents, post)
    assert d.status == "retry" and "ConnectTimeout" in d.last_error


def test_circuit_breaker_pauses_failing_endpoint(platform, agents, transport):
    transport.status = 500
    platform.settings.circuit_break_after = 2
    for title in ("Diwali food delivery discount", "Diwali food delivery offer", "Diwali meal deal"):
        with platform.Session() as db:
            service.publish_post(platform, db, service.resolve_agent(db, "swiggy"), PostCreate(
                kind="offer", title=title, body="5% off all food delivery orders on Diwali, pay by UPI or card."))
    with platform.Session() as db:
        phonepe = db.get(Agent, agents["phonepe"][0].id)
        assert phonepe.webhook_paused_until is not None and phonepe.webhook_paused_until > utcnow()
        calls_before = len([c for c in transport.calls if c[0].endswith("/phonepe")])
        for d in db.scalars(select(Delivery).where(Delivery.agent_id == phonepe.id)):
            d.next_attempt_at = utcnow() - timedelta(seconds=1)
        db.commit()
    platform.dispatcher.run_once()
    # paused: nothing new was sent to PhonePe
    assert len([c for c in transport.calls if c[0].endswith("/phonepe")]) == calls_before


def test_agents_without_webhook_get_inbox_only(platform, agents, transport):
    with platform.Session() as db:
        phonepe = db.get(Agent, agents["phonepe"][0].id)
        phonepe.webhook_url = None
        db.commit()
    post = publish_brief_example(platform)
    d = phonepe_delivery(platform, agents, post)
    assert d.channel == "inbox" and d.status == "inbox"
    assert not any(c[0].endswith("/phonepe") for c in transport.calls)


def test_payload_body_is_valid_json(platform, agents, transport):
    publish_brief_example(platform)
    for _url, body, headers in transport.calls:
        assert json.loads(body)["delivery_id"] == headers["X-AgentNet-Delivery"]
