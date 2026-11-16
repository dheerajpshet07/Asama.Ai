from datetime import timedelta

from sqlalchemy import select

from agentnet import feedback, service
from agentnet.models import AffinityStat, Agent, Delivery, Post
from agentnet.schemas import CommentCreate, PostCreate
from agentnet.util import utcnow


def publish(platform, handle, title, body):
    with platform.Session() as db:
        return service.publish_post(platform, db, service.resolve_agent(db, handle),
                                    PostCreate(title=title, body=body))


def score_for(platform, post_id, handle):
    with platform.Session() as db:
        rows = service.routing_report(db, db.get(Post, post_id))
        return next(r for r, a in rows if a.handle == handle)


def delivery_for(platform, post_id, handle):
    with platform.Session() as db:
        agent = service.resolve_agent(db, handle)
        return db.scalar(select(Delivery).where(Delivery.post_id == post_id, Delivery.agent_id == agent.id))


SALES = "Bajaj Auto's domestic two-wheeler sales grew {n}% year on year in {m}, led by festive demand."


def test_affinity_math():
    assert feedback.affinity({}, [("topic:x", 1.0)]) == 0.0
    always = {"topic:x": (10.0, 10.0)}
    never = {"topic:x": (10.0, 0.0)}
    assert feedback.affinity(always, [("topic:x", 1.0)]) > 0.6
    assert feedback.affinity(never, [("topic:x", 1.0)]) < -0.6


def test_not_relevant_feedback_stops_similar_notifications(platform, agents):
    months = ["October", "November", "December", "January", "February", "March"]
    first = publish(platform, "bajaj-auto", "Two-wheeler sales update", SALES.format(n=18, m=months[0]))
    start = score_for(platform, first.id, "hdfc-bank")
    assert start.action == "notify"

    last = start
    for i, month in enumerate(months[1:], start=1):
        # HDFC keeps saying these are not relevant to it
        d = delivery_for(platform, last.post_id, "hdfc-bank")
        if d is not None:
            with platform.Session() as db:
                service.acknowledge(platform, db, db.get(Agent, d.agent_id), d.id, "not_relevant")
        post = publish(platform, "bajaj-auto", "Two-wheeler sales update", SALES.format(n=10 + i, m=month))
        last = score_for(platform, post.id, "hdfc-bank")
        if last.action != "notify":
            break
    assert last.action == "drop", "router should learn that HDFC does not want sales updates"
    assert last.score < start.score
    assert last.features["affinity"] < 0


def test_engagement_raises_scores_for_similar_posts(platform, agents):
    first = publish(platform, "bajaj-auto", "Two-wheeler sales update", SALES.format(n=18, m="October"))
    before = score_for(platform, first.id, "acko")
    with platform.Session() as db:
        service.add_comment(platform, db, service.resolve_agent(db, "acko"), db.get(Post, first.id),
                            CommentCreate(body="Acko can insure every new two-wheeler at the dealership."))
    second = publish(platform, "bajaj-auto", "Two-wheeler sales update", SALES.format(n=11, m="November"))
    after = score_for(platform, second.id, "acko")
    assert after.score > before.score
    assert after.features["affinity"] > 0
    assert delivery_for(platform, first.id, "acko").outcome == "engaged"


def test_comment_without_notification_counts_as_missed_match(platform, agents):
    post = publish(platform, "airtel", "Scheduled network maintenance in Mumbai",
                   "Airtel will carry out scheduled maintenance in parts of Mumbai on Sunday night.")
    assert delivery_for(platform, post.id, "phonepe") is None
    with platform.Session() as db:
        phonepe = service.resolve_agent(db, "phonepe")
        service.add_comment(platform, db, phonepe, db.get(Post, post.id),
                            CommentCreate(body="We'll show a banner for UPI users in Mumbai during the window."))
        stats = {s.key: s for s in db.scalars(select(AffinityStat).where(AffinityStat.agent_id == phonepe.id))}
    assert stats["author_industry:telecom"].engaged > 0


def test_sweep_marks_stale_notifications_ignored(platform, agents):
    post = publish(platform, "swiggy", "Diwali food delivery discount", "5% off all food delivery orders on Diwali.")
    with platform.Session() as db:
        for d in db.scalars(select(Delivery).where(Delivery.post_id == post.id)):
            d.created_at = utcnow() - timedelta(days=2)
        db.commit()
    n = service.sweep_ignored(platform)
    assert n >= 1
    assert delivery_for(platform, post.id, "phonepe").outcome == "ignored"
    assert service.sweep_ignored(platform) == 0  # idempotent


def test_profile_update_resets_learned_drift(platform, agents):
    from agentnet.schemas import AgentUpdate
    post = publish(platform, "swiggy", "Diwali food delivery discount", "5% off all food delivery orders on Diwali.")
    d = delivery_for(platform, post.id, "phonepe")
    with platform.Session() as db:
        service.acknowledge(platform, db, db.get(Agent, d.agent_id), d.id, "not_relevant")
        agent = db.get(Agent, d.agent_id)
        assert agent.drift_vec is not None
        service.update_agent(platform, db, agent, AgentUpdate(
            interests="Only interested in NPCI and RBI regulation of UPI and wallets."))
        assert agent.drift_vec is None
        assert "regulation" in agent.profile_topics
