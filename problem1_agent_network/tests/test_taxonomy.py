from agentnet.taxonomy import detect_topics, split_needs
from agentnet.text import stem, tokens


def test_stemming_handles_common_plurals():
    assert stem("deliveries") == "delivery"
    assert stem("offers") == "offer"
    assert stem("business") == "business"


def test_stopwords_dropped_for_features_but_kept_for_phrases():
    assert "the" not in tokens("the offer")
    assert tokens("tie up", keep_stopwords=True) == ["tie", "up"]


def test_detects_brief_example():
    topics = detect_topics("Swiggy: 5% off all deliveries on Diwali")
    assert {"offers", "festive", "food-delivery"} <= topics.keys()


def test_cross_vocabulary_terms_map_to_same_topic():
    assert "payments" in detect_topics("Pay with UPI at checkout")
    assert "payments" in detect_topics("new wallet for merchants")


def test_longest_phrase_wins():
    # "two wheeler loan" is lending; it must not also count as the automotive term
    topics = detect_topics("Pre-approved two-wheeler loans")
    assert "lending" in topics
    assert "automotive" not in topics


def test_same_length_terms_count_for_both_topics():
    topics = detect_topics("easy EMI options")
    assert "cards" in topics and "lending" in topics
    # ...but a longer card phrase claims the word for itself
    assert "lending" not in detect_topics("no cost EMI available")


def test_new_year_is_not_just_year():
    assert "festive" not in detect_topics("results for the year")
    assert "festive" in detect_topics("New Year sale")


def test_split_needs_after_cue():
    about, need = split_needs("Up to 20% off flights. Looking for bank and card partners to add discounts.")
    assert "flights" in about and "bank" in need and "flights" not in need


def test_split_needs_before_cue():
    about, need = split_needs("Bank partners and payment providers can apply for checkout offers.")
    assert need.startswith("Bank partners")
    assert "checkout" in about


def test_post_without_ask_has_no_need():
    about, need = split_needs("Swiggy is offering 5% off this Diwali.")
    assert need == "" and "Swiggy" in about
