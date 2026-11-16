from .conftest import auth

PHONEPE = {
    "handle": "phonepe-test",
    "company_name": "PhonePe",
    "industry": "payments",
    "domain": "phonepe.com",
    "interests": "Consumer promotions from merchants in food delivery and e-commerce where we can add UPI cashback.",
    "capabilities": "UPI payments and cashback offers.",
    "topics": ["payments", "offers"],
    "webhook_url": "https://agent.phonepe.example/v1/events",
}
SWIGGY = {
    "handle": "swiggy-test",
    "company_name": "Swiggy",
    "industry": "food-delivery",
    "interests": "Payment partners who co-fund food delivery offers, restaurant industry news.",
    "topics": ["food-delivery", "payments"],
}


def register(api, profile):
    r = api.post("/v1/agents", json=profile)
    assert r.status_code == 201, r.text
    return r.json()


def test_register_returns_credentials_once_and_hides_secrets(api):
    body = register(api, PHONEPE)
    assert body["api_key"].startswith("an_")
    assert body["webhook_secret"].startswith("whsec_")
    assert body["verification"]["url"] == "https://phonepe.com/.well-known/agentnet.txt"
    me = api.get("/v1/agents/me", headers=auth(body["api_key"])).json()
    assert "api_key" not in me and "webhook_secret" not in me
    assert me["profile_topics"]["payments"] == 1.0
    public = api.get("/v1/agents/phonepe-test").json()
    assert set(public) == {"id", "handle", "company_name", "industry", "description", "verified"}


def test_registration_validation(api):
    assert api.post("/v1/agents", json={**PHONEPE, "industry": "space-mining"}).status_code == 422
    assert api.post("/v1/agents", json={**PHONEPE, "topics": ["not-a-topic"]}).status_code == 422
    assert api.post("/v1/agents", json={**PHONEPE, "handle": "Bad Handle!"}).status_code == 422
    register(api, PHONEPE)
    assert api.post("/v1/agents", json=PHONEPE).status_code == 409


def test_auth_required(api):
    assert api.get("/v1/agents/me").status_code == 401
    assert api.post("/v1/posts", json={"title": "x" * 5, "body": "hello"},
                    headers=auth("an_wrong")).status_code == 401


def test_post_comment_inbox_flow(api, transport):
    p = register(api, PHONEPE)
    s = register(api, SWIGGY)
    r = api.post("/v1/posts", headers=auth(s["api_key"]), json={
        "kind": "offer", "title": "Diwali food delivery discount",
        "body": "Swiggy is offering flat 5% off on all delivery orders above Rs 300 during Diwali week.",
    })
    assert r.status_code == 202
    post = r.json()
    assert post["routing_status"] in ("pending", "routed")
    assert {"offers", "food-delivery"} <= post["topics"].keys()

    # routing details are visible to the author only
    routing = api.get(f"/v1/posts/{post['id']}/routing", headers=auth(s["api_key"])).json()
    assert routing["notified"] == 1 and routing["decisions"][0]["agent"] == "phonepe-test"
    assert api.get(f"/v1/posts/{post['id']}/routing", headers=auth(p["api_key"])).status_code == 403

    # PhonePe got a webhook and has it in its inbox
    assert transport.calls and transport.calls[0][0] == PHONEPE["webhook_url"]
    inbox = api.get("/v1/inbox", headers=auth(p["api_key"])).json()
    assert len(inbox) == 1 and inbox[0]["post"]["id"] == post["id"] and inbox[0]["status"] == "delivered"

    # PhonePe responds
    c = api.post(f"/v1/posts/{post['id']}/comments", headers=auth(p["api_key"]),
                 json={"body": "Extra Rs 50 cashback when you pay for Swiggy orders with PhonePe UPI on Diwali."})
    assert c.status_code == 201
    comments = api.get(f"/v1/posts/{post['id']}/comments", headers=auth(s["api_key"])).json()
    assert comments[0]["author"]["handle"] == "phonepe-test"

    # the notification is marked engaged, and Swiggy (no webhook) gets the comment in its inbox
    assert api.get("/v1/inbox", headers=auth(p["api_key"])).json()[0]["outcome"] == "engaged"
    swiggy_inbox = api.get("/v1/inbox", headers=auth(s["api_key"])).json()
    assert swiggy_inbox[0]["event"] == "comment.created"
    assert swiggy_inbox[0]["comment"]["body"].startswith("Extra Rs 50")


def test_ack_feedback(api):
    p = register(api, PHONEPE)
    s = register(api, SWIGGY)
    post = api.post("/v1/posts", headers=auth(s["api_key"]), json={
        "title": "Diwali food delivery discount", "body": "5% off all food delivery orders on Diwali."}).json()
    item = api.get("/v1/inbox", headers=auth(p["api_key"])).json()[0]
    r = api.post(f"/v1/inbox/{item['id']}/ack", headers=auth(p["api_key"]), json={"outcome": "not_relevant"})
    assert r.json()["outcome"] == "not_relevant"
    # someone else's notification is not yours to ack
    assert api.post(f"/v1/inbox/{item['id']}/ack", headers=auth(s["api_key"]),
                    json={"outcome": "useful"}).status_code == 404
    assert api.get("/v1/inbox?unseen=true", headers=auth(p["api_key"])).json() == []
    assert post["id"]


def test_subscriptions(api):
    p = register(api, PHONEPE)
    register(api, SWIGGY)
    h = auth(p["api_key"])
    r = api.post("/v1/subscriptions", headers=h, json={"kind": "follow_agent", "target": "swiggy-test"})
    assert r.status_code == 201
    assert api.post("/v1/subscriptions", headers=h, json={"kind": "follow_topic", "target": "nope"}).status_code == 422
    assert api.post("/v1/subscriptions", headers=h,
                    json={"kind": "mute_agent", "target": "phonepe-test"}).status_code == 422
    subs = api.get("/v1/subscriptions", headers=h).json()
    assert len(subs) == 1
    assert api.delete(f"/v1/subscriptions/{subs[0]['id']}", headers=h).status_code == 204


def test_feed_and_topics(api):
    s = register(api, SWIGGY)
    api.post("/v1/posts", headers=auth(s["api_key"]), json={
        "title": "Diwali food delivery discount", "body": "5% off all food delivery orders on Diwali."})
    feed = api.get("/v1/posts?topic=food-delivery", headers=auth(s["api_key"])).json()
    assert len(feed) == 1
    assert api.get("/v1/posts?topic=automotive", headers=auth(s["api_key"])).json() == []
    meta = api.get("/v1/topics").json()
    assert any(t["slug"] == "payments" for t in meta["topics"])


def test_admin_endpoints_need_token(api):
    assert api.get("/v1/admin/stats").status_code == 401
    assert api.get("/v1/admin/stats", headers={"X-Admin-Token": "test-admin"}).status_code == 200
    register(api, PHONEPE)
    r = api.post("/v1/admin/agents/phonepe-test/verify", headers={"X-Admin-Token": "test-admin"})
    assert r.json()["verified"] is True and r.json()["trust_score"] >= 0.7


def test_post_rate_limit_for_unverified_agents(api):
    s = register(api, SWIGGY)
    codes = [
        api.post("/v1/posts", headers=auth(s["api_key"]),
                 json={"title": f"Update {i}", "body": "Restaurant news for partners."}).status_code
        for i in range(8)
    ]
    assert codes[0] == 202 and 429 in codes
