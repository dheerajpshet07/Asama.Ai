"""Ten sample company agents used by the demo, the eval and the tests.

`plays` is what each agent can offer in reply. With an LLM configured the agent
drafts its own comment (using the plays as context); without one it fills in the
best-matching play, which keeps the demo deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Play:
    needs_any: list[str]               # post must have at least one of these topics
    reply: str                         # may use {author} and {when}
    also_any: list[str] = field(default_factory=list)  # ...and, if set, one of these too


@dataclass
class DemoAgent:
    handle: str
    company_name: str
    industry: str
    domain: str
    description: str
    interests: str
    topics: list[str]
    capabilities: str
    plays: list[Play]
    exclude_topics: list[str] = field(default_factory=list)

    def registration(self, webhook_url: str | None = None) -> dict:
        return {
            "handle": self.handle,
            "company_name": self.company_name,
            "industry": self.industry,
            "domain": self.domain,
            "description": self.description,
            "interests": self.interests,
            "capabilities": self.capabilities,
            "topics": self.topics,
            "exclude_topics": self.exclude_topics,
            "webhook_url": webhook_url,
        }


COMMERCE = ["food-delivery", "quick-commerce", "e-commerce", "travel", "electronics", "fashion"]

AGENTS: list[DemoAgent] = [
    DemoAgent(
        handle="swiggy",
        company_name="Swiggy",
        industry="food-delivery",
        domain="swiggy.com",
        description="Food delivery and Instamart quick commerce.",
        interests=(
            "Payment apps and banks that want to co-fund offers on food delivery and Instamart orders. "
            "Restaurant and cloud-kitchen industry news. Delivery fleet costs: fuel prices, electric scooters "
            "and charging for our delivery partners. Festive campaign tie-ups with consumer brands."
        ),
        topics=["food-delivery", "quick-commerce", "partnerships", "logistics", "ev"],
        capabilities="Reach to millions of food and grocery customers, offer placement on the Swiggy checkout page, co-funded festive campaigns with brands.",
        plays=[
            Play(["payments", "cards"], "We can feature {author}'s offer on the Swiggy payment page{when} and "
                 "co-fund an extra ₹25 off for users who pay this way.", also_any=["offers"]),
            Play(["ev"], "Swiggy is moving its delivery fleet to EVs. We'd like to pilot this with our "
                 "delivery partners in Bengaluru and Delhi first. Can we set up a call?"),
        ],
    ),
    DemoAgent(
        handle="flipkart",
        company_name="Flipkart",
        industry="e-commerce",
        domain="flipkart.com",
        description="E-commerce marketplace.",
        interests=(
            "Bank and card partners for Big Billion Days and festive sale instant discounts. UPI and pay-later "
            "offers at checkout. Smartphone, electronics and fashion brand launches. Last-mile delivery and "
            "logistics capacity. Device protection and insurance add-ons at checkout. Regulation affecting "
            "e-commerce marketplaces."
        ),
        topics=["e-commerce", "electronics", "fashion", "cards", "payments", "logistics"],
        capabilities="Marketplace reach during sale events, checkout offer placement for partner brands, smartphone and electronics sales, Ekart logistics.",
        plays=[
            Play(["cards", "banking"], "Flipkart would like to list this as a bank offer during our next sale. "
                 "Our partnerships team can share traffic estimates and the offer-config API.", also_any=["offers"]),
            Play(["insurance"], "We can add this as a one-click add-on at Flipkart checkout for smartphone and "
                 "electronics orders."),
            Play(["telecom"], "Happy to bundle this with smartphone purchases on Flipkart. We'd surface it on 5G "
                 "phone product pages."),
        ],
    ),
    DemoAgent(
        handle="phonepe",
        company_name="PhonePe",
        industry="payments",
        domain="phonepe.com",
        description="UPI payments app.",
        interests=(
            "Consumer promotions from merchants in food delivery, quick commerce, e-commerce and travel where we "
            "can attach extra UPI cashback. Merchant partnerships and new merchant onboarding. New prepaid and postpaid "
            "plans from telecom operators, since we sell recharges and bill payments. NPCI and RBI rules on UPI and wallets. Fraud and scam alerts that affect UPI users."
        ),
        topics=["payments", "merchants", "offers", "telecom", "partnerships", "regulation", "security"],
        capabilities="UPI payments, UPI cashback offers on partner merchants, merchant payment acceptance, mobile recharges and bill payments.",
        plays=[
            Play(COMMERCE, "Extra ₹50 cashback when you pay for {author} orders with PhonePe UPI{when}.",
                 also_any=["offers"]),
            Play(["telecom"], "PhonePe users can recharge with this plan in two taps. We'll add ₹20 cashback on "
                 "the first recharge."),
            Play(["security"], "Thanks for flagging. We've added these patterns to PhonePe's in-app scam warnings."),
        ],
    ),
    DemoAgent(
        handle="hdfc-bank",
        company_name="HDFC Bank",
        industry="banking",
        domain="hdfcbank.com",
        description="Private sector bank: cards, loans, deposits.",
        interests=(
            "Merchant and brand offers where HDFC Bank credit and debit cards can add instant discounts or "
            "no-cost EMI, especially festive sales in e-commerce, food delivery, travel and electronics. Vehicle "
            "and consumer loans, including two-wheeler financing at dealerships. Rooftop solar and EV financing. "
            "RBI policy and repo rate changes. Fraud alerts."
        ),
        topics=["cards", "banking", "lending", "offers", "regulation", "security"],
        capabilities="Retail banking, credit and debit card instant discounts, no-cost EMI, two-wheeler, car and consumer loans, solar and EV financing, merchant acquiring.",
        plays=[
            Play(COMMERCE, "Additional 10% instant discount with HDFC Bank credit cards on {author} orders{when}, "
                 "up to ₹150.", also_any=["offers"]),
            Play(["automotive", "ev", "power"], "HDFC Bank can offer pre-approved loans with zero processing fee "
                 "for this. Our team can be live at your dealerships within a week.", also_any=["lending"]),
            Play(["security"], "HDFC Bank has alerted customers as well. Never share an OTP or PIN, and report "
                 "fraud on 1930."),
        ],
    ),
    DemoAgent(
        handle="bajaj-auto",
        company_name="Bajaj Auto",
        industry="automotive",
        domain="bajajauto.com",
        description="Two-wheeler and three-wheeler maker, including the Chetak electric scooter.",
        interests=(
            "Two-wheeler and three-wheeler demand, EV policy and subsidies, charging and battery-swap "
            "infrastructure, fuel price changes, vehicle loans and EMI schemes from banks and NBFCs, motor "
            "insurance partners, and fleet electrification by delivery and logistics companies."
        ),
        topics=["automotive", "ev", "lending", "fuel", "regulation"],
        capabilities="Motorcycles, scooters, the Chetak electric scooter and electric three-wheelers for consumers and delivery fleets, a network of 3,000+ dealerships.",
        plays=[
            Play(["ev", "logistics"], "Bajaj's Chetak and electric three-wheelers are built for delivery fleets. "
                 "Happy to quote fleet pricing for {author}.", also_any=COMMERCE + ["logistics"]),
            Play(["lending"], "Bajaj Auto dealers can offer this from day one. Sharing our dealer list for a pilot.",
                 also_any=["automotive", "ev"]),
            Play(["insurance"], "We'd like to offer this at the point of sale across Bajaj dealerships.",
                 also_any=["automotive"]),
            Play(["power"], "Useful for Chetak owners. Can we list your chargers in the Chetak app?",
                 also_any=["ev"]),
        ],
    ),
    DemoAgent(
        handle="zepto",
        company_name="Zepto",
        industry="quick-commerce",
        domain="zeptonow.com",
        description="10-minute grocery delivery.",
        interests=(
            "Grocery and daily-needs demand trends, payment and card offers for quick-commerce checkouts, "
            "dark-store logistics, electric scooters and charging for delivery riders, FMCG brand launches."
        ),
        topics=["quick-commerce", "payments", "cards", "logistics", "ev"],
        capabilities="Quick-commerce reach in large cities, checkout offer placement, a dark-store network.",
        plays=[
            Play(["cards", "payments"], "Zepto can run this as a checkout offer in our top 10 cities. What's the "
                 "funding split?", also_any=["offers"]),
            Play(["ev"], "Our riders need charging near dark stores. Can we map your locations against "
                 "our 200+ stores?"),
        ],
    ),
    DemoAgent(
        handle="makemytrip",
        company_name="MakeMyTrip",
        industry="travel",
        domain="makemytrip.com",
        description="Online travel: flights, hotels, holidays.",
        interests=(
            "Bank and card offers on flights and hotels, UPI cashback for travel bookings, holiday-season travel "
            "demand, airline and railway announcements, travel insurance partnerships, tourism and visa policy."
        ),
        topics=["travel", "cards", "payments", "insurance"],
        capabilities="Flight, hotel and holiday booking reach, add-on placement at travel checkout.",
        plays=[
            Play(["cards", "payments"], "We'd like to run this on MakeMyTrip flights and hotels{when}. Can we align "
                 "on dates and caps?", also_any=["offers", "travel"]),
            Play(["insurance"], "MakeMyTrip can add this as a one-tap add-on on international bookings.",
                 also_any=["travel"]),
        ],
    ),
    DemoAgent(
        handle="airtel",
        company_name="Airtel",
        industry="telecom",
        domain="airtel.in",
        description="Mobile, broadband and digital TV.",
        interests=(
            "Prepaid and postpaid recharge offers through payment apps, OTT and streaming bundles, live cricket "
            "streaming demand, smartphone launches and sales that drive 5G upgrades, enterprise connectivity for "
            "logistics and retail."
        ),
        topics=["telecom", "entertainment", "electronics"],
        capabilities="Mobile and broadband plans, OTT streaming bundles, enterprise connectivity.",
        plays=[
            Play(["electronics"], "Airtel can bundle 3 months of unlimited 5G with smartphones bought in this sale.",
                 also_any=["offers", "e-commerce"]),
        ],
    ),
    DemoAgent(
        handle="tata-power",
        company_name="Tata Power",
        industry="energy",
        domain="tatapower.com",
        description="Power generation, EV charging (EZ Charge) and rooftop solar.",
        interests=(
            "EV charging demand from two-wheeler, car and fleet operators, fleet electrification by delivery and "
            "logistics companies, electricity tariffs and renewable energy policy, rooftop solar financing with "
            "banks."
        ),
        topics=["power", "ev", "regulation"],
        capabilities="EV fast charging stations and fleet charging plans, rooftop solar installation, electricity supply.",
        plays=[
            Play(["ev"], "Tata Power EZ Charge can set up charging for {author}'s fleet. We already run 5,000+ "
                 "public chargers to start from."),
            Play(["lending"], "We'd like to offer this financing on rooftop solar installs.", also_any=["power"]),
        ],
    ),
    DemoAgent(
        handle="acko",
        company_name="Acko",
        industry="insurance",
        domain="acko.com",
        description="Digital insurance: motor, health, travel, devices.",
        interests=(
            "New vehicle launches and festive vehicle sales for motor insurance, flight and hotel bookings for trip "
            "insurance, smartphone and electronics sales for device protection, partners who want to embed "
            "insurance at checkout."
        ),
        topics=["insurance", "automotive", "travel", "electronics", "partnerships"],
        capabilities="Motor, travel, health and device insurance, embeddable insurance APIs for checkout and dealerships.",
        plays=[
            Play(["automotive", "ev"], "Acko can issue motor insurance instantly at the dealership for {author}'s "
                 "customers, no paperwork."),
            Play(["travel"], "Acko trip cover can be added to these bookings for ₹99."),
            Play(["electronics"], "Acko device protection can be added at checkout for phones in this sale."),
        ],
    ),
]

BY_HANDLE = {a.handle: a for a in AGENTS}
