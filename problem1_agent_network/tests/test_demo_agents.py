import json
import time

from fastapi.testclient import TestClient

from agentnet.delivery import sign
from demo_agents.profiles import BY_HANDLE
from demo_agents.responder import Responder
from demo_agents.server import AgentRuntime, create_agents_app

from .conftest import FakeLLM

SWIGGY_EVENT = {
    "event": "post.matched",
    "delivery_id": "d-1",
    "post": {
        "id": "p-1",
        "kind": "offer",
        "title": "Diwali food delivery discount",
        "body": "Swiggy is offering flat 5% off on all delivery orders above Rs 300 during Diwali week.",
        "topics": ["food-delivery", "offers", "festive"],
        "asks_for": [],
        "author": {"handle": "swiggy", "company_name": "Swiggy", "industry": "food-delivery"},
    },
}


def test_playbook_reproduces_the_brief_example():
    r = Responder(BY_HANDLE["phonepe"]).react(SWIGGY_EVENT)
    assert r.engage
    assert r.comment == "Extra ₹50 cashback when you pay for Swiggy orders with PhonePe UPI on Diwali."


def test_agent_with_nothing_to_offer_declines():
    r = Responder(BY_HANDLE["bajaj-auto"]).react(SWIGGY_EVENT)
    assert not r.engage and r.comment is None


def test_llm_reply_is_used_and_post_is_passed_as_data():
    llm = FakeLLM('{"engage": true, "comment": "Extra Rs 50 cashback on Swiggy via UPI.", "reason": "fit"}')
    r = Responder(BY_HANDLE["phonepe"], llm).react(SWIGGY_EVENT)
    assert r.engage and r.comment == "Extra Rs 50 cashback on Swiggy via UPI."
    system, prompt = llm.prompts[0]
    assert "untrusted" in system and "<post>" in prompt
    assert "UPI payments" in system  # grounded in what the company can actually offer


def test_bad_llm_output_falls_back_to_playbook():
    from agentnet.llm import LLMError
    r = Responder(BY_HANDLE["phonepe"], FakeLLM(LLMError("timeout"))).react(SWIGGY_EVENT)
    assert r.engage and "PhonePe UPI" in r.comment


class StubClient:
    def __init__(self):
        self.comments, self.acks = [], []

    def comment(self, post_id, body):
        self.comments.append((post_id, body))

    def ack(self, delivery_id, outcome):
        self.acks.append((delivery_id, outcome))


def _signed(secret, event):
    body = json.dumps(event).encode()
    ts = str(int(time.time()))
    return body, {"X-AgentNet-Timestamp": ts, "X-AgentNet-Signature": sign(secret, ts, body),
                  "X-AgentNet-Delivery": event["delivery_id"], "X-AgentNet-Event": event["event"],
                  "Content-Type": "application/json"}


def test_webhook_receiver_verifies_and_responds():
    stub = StubClient()
    rt = AgentRuntime(BY_HANDLE["phonepe"], stub, "whsec_test", Responder(BY_HANDLE["phonepe"]))
    app = create_agents_app({"phonepe": rt})
    with TestClient(app) as client:
        body, headers = _signed("whsec_test", SWIGGY_EVENT)
        assert client.post("/hooks/phonepe", content=body, headers=headers).status_code == 200
        assert stub.comments and stub.comments[0][0] == "p-1"

        bad = dict(headers, **{"X-AgentNet-Signature": "v1=forged"})
        assert client.post("/hooks/phonepe", content=body, headers=bad).status_code == 401
        assert client.post("/hooks/unknown", content=body, headers=headers).status_code == 404


def test_receiver_acks_not_relevant_when_it_has_nothing_to_say():
    stub = StubClient()
    rt = AgentRuntime(BY_HANDLE["bajaj-auto"], stub, "whsec_b", Responder(BY_HANDLE["bajaj-auto"]))
    with TestClient(create_agents_app({"bajaj-auto": rt})) as client:
        body, headers = _signed("whsec_b", SWIGGY_EVENT)
        client.post("/hooks/bajaj-auto", content=body, headers=headers)
    assert stub.acks == [("d-1", "not_relevant")]
