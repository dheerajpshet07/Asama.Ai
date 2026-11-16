"""Controlled vocabulary of topics and industries.

Topics are what make routing explainable ("notified because: payments, offers")
and give the cheap embedder some cross-vocabulary reach: "UPI" and "wallet" both
land on `payments` even though they share no characters.

Each topic has an importance weight. Domain topics (payments, automotive, ...)
are 1.0. Context topics that many unrelated posts share get less, so that two
companies that both "care about Diwali" are not treated as a match on that alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import phrase, tokens

INDUSTRIES: dict[str, str] = {
    "food-delivery": "Food delivery",
    "quick-commerce": "Quick commerce & grocery",
    "e-commerce": "E-commerce",
    "payments": "Payments & fintech",
    "banking": "Banking",
    "lending": "Lending & NBFC",
    "insurance": "Insurance",
    "automotive": "Automotive",
    "travel": "Travel & hospitality",
    "telecom": "Telecom",
    "energy": "Energy & utilities",
    "logistics": "Logistics",
    "media": "Media & entertainment",
    "retail": "Retail",
    "other": "Other",
}

# Posts from a company are, a priori, a little bit about that company's own domain.
INDUSTRY_TOPIC = {
    "food-delivery": "food-delivery",
    "quick-commerce": "quick-commerce",
    "e-commerce": "e-commerce",
    "payments": "payments",
    "banking": "banking",
    "lending": "lending",
    "insurance": "insurance",
    "automotive": "automotive",
    "travel": "travel",
    "telecom": "telecom",
    "energy": "power",
    "logistics": "logistics",
    "media": "entertainment",
}


@dataclass
class Topic:
    label: str
    weight: float
    terms: list[tuple[str, float]]
    patterns: list[tuple[re.Pattern, float]] = field(default_factory=list)


def _t(*items) -> list[tuple[str, float]]:
    """Terms are plain strings (weight 1.0) or (term, weight) tuples."""
    out = []
    for item in items:
        term, w = (item, 1.0) if isinstance(item, str) else item
        out.append((phrase(term), w))
    return out


TOPICS: dict[str, Topic] = {
    "offers": Topic(
        "Offers & discounts",
        0.4,
        _t("offer", "discount", "sale", "cashback", "coupon", "deal", "promo", "promotion", "voucher",
           "free delivery", "price drop", "buy one get one", "lowest price", ("flat", 0.5), ("off", 0.4),
           ("save", 0.3)),
        [(re.compile(r"\d+\s*%\s*(off|discount|cashback)", re.I), 1.0),
         (re.compile(r"(rs\.?|₹|inr)\s*\d+[\d,]*\s*(off|cashback)", re.I), 1.0)],
    ),
    "festive": Topic(
        "Festivals & seasonal events",
        0.25,
        _t("diwali", "deepavali", "dhanteras", "dussehra", "navratri", "holi", "eid", "christmas", "new year",
           "onam", "pongal", "durga puja", "ganesh chaturthi", "raksha bandhan", "festive", "festival",
           "big billion days", "great indian festival", "end of season sale", "holiday season"),
    ),
    "food-delivery": Topic(
        "Food delivery",
        1.0,
        _t("food delivery", "food", "restaurant", "meal", "dining", "takeaway", "cloud kitchen", "biryani",
           "pizza", "delivery order", ("delivery", 0.4), ("kitchen", 0.5), ("order", 0.2)),
    ),
    "quick-commerce": Topic(
        "Grocery & quick commerce",
        1.0,
        _t("grocery", "quick commerce", "10 minute delivery", "10 minute", "dark store", "daily needs",
           "fmcg", ("essentials", 0.5), ("vegetables", 0.6), ("fruits", 0.5), ("milk", 0.4)),
    ),
    "e-commerce": Topic(
        "Online shopping",
        1.0,
        _t("e commerce", "ecommerce", "online shopping", "marketplace", "big billion days",
           ("shopping", 0.6), ("seller", 0.4), ("cart", 0.4)),
    ),
    "electronics": Topic(
        "Consumer electronics",
        0.8,
        _t("smartphone", "laptop", "television", "tv", "electronics", "appliance", "home appliance",
           "headphone", "smartwatch", ("phone", 0.5)),
    ),
    "fashion": Topic(
        "Fashion & lifestyle",
        0.8,
        _t("fashion", "apparel", "clothing", "footwear", "beauty", "cosmetics", ("lifestyle", 0.5)),
    ),
    "payments": Topic(
        "Payments & UPI",
        1.0,
        _t("upi", "payment", "wallet", "payment gateway", "scan and pay", "qr", "bill payment", "autopay",
           "upi lite", "npci", "payment provider", ("pay", 0.6), ("checkout", 0.3), ("transaction", 0.6),
           ("merchant", 0.4), ("recharge", 0.4)),
    ),
    "cards": Topic(
        "Credit & debit cards",
        1.0,
        _t("credit card", "debit card", "card offer", "no cost emi", "co branded card", "instant discount",
           "reward point", ("card", 0.6), ("emi", 0.5), ("visa", 0.4), ("mastercard", 0.6), ("rupay", 0.6)),
    ),
    "banking": Topic(
        "Banking",
        1.0,
        _t("bank", "banking", "retail banking", "savings account", "current account", "fixed deposit", "net banking",
           "account opening", ("deposit", 0.6), ("atm", 0.6)),
    ),
    "lending": Topic(
        "Loans & credit",
        1.0,
        _t("loan", "lending", "financing", "pay later", "bnpl", "credit line", "nbfc", "down payment",
           "two wheeler loan", "auto loan", "car loan", "vehicle loan", "personal loan", "home loan",
           "ev financing", "vehicle financing", "solar financing", "consumer loan", ("emi", 0.5),
           ("credit", 0.4), ("finance", 0.5), ("interest rate", 0.6)),
    ),
    "insurance": Topic(
        "Insurance",
        1.0,
        _t("insurance", "insure", "insurer", "motor insurance", "travel insurance", "trip cover", "health cover",
           "device protection", ("premium", 0.4), ("claim", 0.4), ("cover", 0.3)),
    ),
    "investments": Topic(
        "Investments & wealth",
        1.0,
        _t("mutual fund", "sip", "stock", "equity", "invest", "investment", "portfolio", "ipo", "demat",
           ("gold", 0.5)),
    ),
    "automotive": Topic(
        "Automotive",
        1.0,
        _t("vehicle", "car", "motorcycle", "scooter", "two wheeler", "three wheeler", "automobile",
           "dealership", "showroom", "test ride", "test drive", "ex showroom", "commercial vehicle",
           ("bike", 0.6), ("auto", 0.4), ("dealer", 0.6)),
    ),
    "ev": Topic(
        "Electric mobility",
        1.0,
        _t("ev", "electric vehicle", "electric scooter", "e scooter", "charging", "charging station",
           "fast charging", "battery swap", "fleet electrification", ("electric", 0.6), ("battery", 0.4)),
    ),
    "travel": Topic(
        "Travel & hospitality",
        1.0,
        _t("travel", "flight", "airline", "hotel", "holiday", "trip", "vacation", "airport", "tourism",
           "resort", ("train", 0.5), ("booking", 0.3)),
    ),
    "telecom": Topic(
        "Telecom & connectivity",
        1.0,
        _t("telecom", "data plan", "prepaid", "postpaid", "5g", "4g", "broadband", "mobile network",
           "unlimited data", "roaming", ("recharge", 0.6), ("sim", 0.6), ("fiber", 0.5), ("network", 0.4)),
    ),
    "entertainment": Topic(
        "Media & entertainment",
        0.8,
        _t("ott", "streaming", "movie", "cricket", "ipl", "world cup", "concert", "music", "gaming",
           ("subscription", 0.3)),
    ),
    # Fuel and electricity are separate on purpose: a two-wheeler maker cares
    # about petrol prices, a rooftop-solar post is not for them.
    "fuel": Topic(
        "Fuel prices",
        1.0,
        _t("fuel", "petrol", "diesel", "cng", "fuel price", "fuel cost"),
    ),
    "power": Topic(
        "Electricity & renewables",
        1.0,
        _t("electricity", "solar", "rooftop solar", "renewable", "power tariff", "electricity tariff", "grid",
           "energy", ("power", 0.4)),
    ),
    "logistics": Topic(
        "Logistics & supply chain",
        1.0,
        _t("logistics", "shipping", "shipment", "courier", "warehouse", "last mile", "last mile delivery",
           "supply chain", "fleet",
           "freight", "delivery capacity", ("delivery partner", 0.6), ("rider", 0.5), ("van", 0.4)),
    ),
    "regulation": Topic(
        "Policy & regulation",
        1.0,
        _t("rbi", "sebi", "npci", "irdai", "regulation", "regulatory", "compliance", "guideline", "mandate",
           "government", "ministry", "gst", "subsidy", "kyc", "repo rate", ("policy", 0.4), ("tax", 0.5),
           ("limit", 0.3)),
    ),
    "market-update": Topic(
        "Market & economy updates",
        0.5,
        _t("market share", "sales figure", "quarterly", "earnings", "inflation", "repo rate", "economy",
           "consumer sentiment", "year on year", "yoy", "monthly sales", "sales growth",
           ("demand", 0.5), ("growth", 0.4), ("market", 0.4), ("result", 0.3), ("volume", 0.4)),
    ),
    # Nearly every agent "cares about partnerships", so on its own it says little.
    "partnerships": Topic(
        "Partnerships & integrations",
        0.3,
        _t("partnership", "tie up", "collaboration", "co branded", "integration", "alliance",
           "looking for partner", "partner program", ("partner", 0.6), ("api", 0.4), ("embed", 0.4)),
    ),
    "loyalty": Topic(
        "Loyalty & memberships",
        0.5,
        _t("loyalty", "membership", "reward point", "super coin", ("reward", 0.6), ("member", 0.4), ("pass", 0.3),
           ("subscription", 0.3)),
    ),
    "merchants": Topic(
        "Merchants & small business",
        0.5,
        _t("merchant", "msme", "small business", "kirana", "store owner", "sme", ("seller", 0.5),
           ("retailer", 0.6)),
    ),
    "security": Topic(
        "Fraud & security",
        1.0,
        _t("fraud", "fraudster", "scam", "phishing", "data breach", "cyber", "vulnerability", "account takeover",
           "fake link", ("security", 0.6), ("otp", 0.5), ("pin", 0.3)),
    ),
    "service-status": Topic(
        "Outages & service status",
        1.0,
        _t("outage", "downtime", "service disruption", "scheduled maintenance", "degraded", "incident",
           ("maintenance", 0.6)),
    ),
}


def _term_index() -> dict[str, list[tuple[tuple[str, ...], str, float]]]:
    idx: dict[str, list] = {}
    for slug, topic in TOPICS.items():
        for term, w in topic.terms:
            toks = tuple(term.split())
            if toks:
                idx.setdefault(toks[0], []).append((toks, slug, w))
    return idx


_TERMS = _term_index()


def detect_topics(text: str, explain: bool = False):
    """Return {topic: confidence in (0, 1]}; with explain=True also the matched terms.

    Where phrases overlap, the longer one wins: "two wheeler loan" is lending,
    and does not also count as the automotive term "two wheeler".
    """
    toks = tokens(text, keep_stopwords=True)
    matches = []
    for i, tok in enumerate(toks):
        for term_toks, slug, w in _TERMS.get(tok, ()):
            n = len(term_toks)
            if tuple(toks[i:i + n]) == term_toks:
                matches.append((i, i + n, slug, w, " ".join(term_toks)))

    matches.sort(key=lambda m: m[0] - m[1])  # longest first
    covered = [0] * len(toks)                # length of the longest accepted match per token
    found: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for start, end, slug, w, term in matches:
        length = end - start
        if any(covered[k] > length for k in range(start, end)):
            continue
        for k in range(start, end):
            covered[k] = max(covered[k], length)
        if term in hits.get(slug, []):
            continue  # each distinct term counts once per topic
        found[slug] = found.get(slug, 0.0) + w
        hits.setdefault(slug, []).append(term)

    for slug, topic in TOPICS.items():
        for pattern, w in topic.patterns:
            m = pattern.search(text)
            if m:
                found[slug] = found.get(slug, 0.0) + w
                hits.setdefault(slug, []).append(m.group(0))

    found = {t: round(min(1.0, s), 3) for t, s in found.items()}
    return (found, hits) if explain else found


# "...is looking for bank and card partners" -> the need follows the cue.
_NEED_AFTER = re.compile(
    r"\b(looking for|seeking|in search of|needs?|wants? to add|is choosing|are choosing|choosing|"
    r"calling for|inviting|open to)\b",
    re.I,
)
# "Bank partners and payment providers can apply" -> the need precedes the cue.
_NEED_BEFORE = re.compile(r"\bcan (apply|join|embed|integrate|sign up|partner|plug in)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_needs(text: str) -> tuple[str, str]:
    """Split a post into what it is about and what the author is asking for.

    "Up to 20% off flights. Looking for bank partners to add discounts."
      -> ("Up to 20% off flights.", "bank partners to add discounts.")

    The distinction matters for routing: an agent that *also wants* bank
    partners is a competitor for them, not a match. Only agents that *offer*
    what is asked for should be matched on the need.
    """
    about, need = [], []
    for sentence in _SENTENCE.split(text):
        m = _NEED_AFTER.search(sentence)
        if m:
            about.append(sentence[: m.start()])
            need.append(sentence[m.end():])
            continue
        m = _NEED_BEFORE.search(sentence)
        if m:
            need.append(sentence[: m.start()])
            about.append(sentence[m.end():])
            continue
        about.append(sentence)
    return " ".join(s.strip() for s in about if s.strip()), " ".join(s.strip() for s in need if s.strip())


def topic_mass(topics: dict[str, float]) -> float:
    return sum(conf * TOPICS[t].weight for t, conf in topics.items() if t in TOPICS)


def top_topics(topics: dict[str, float], n: int = 3) -> list[str]:
    ranked = sorted(topics.items(), key=lambda kv: kv[1] * TOPICS[kv[0]].weight, reverse=True)
    return [t for t, _ in ranked[:n]]


def validate_topics(slugs: list[str]) -> list[str]:
    unknown = [s for s in slugs if s not in TOPICS]
    if unknown:
        raise ValueError(f"unknown topics: {', '.join(unknown)}")
    return list(dict.fromkeys(slugs))
