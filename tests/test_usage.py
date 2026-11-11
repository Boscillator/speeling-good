from concurrent.futures import ThreadPoolExecutor

from speeling_good.preprocessing.usage import UsageTracker


def test_usage_normalizes_stages_and_sums_only_reported_counts():
    usage = UsageTracker()
    usage.record("early_decision", {"input_tokens": 40})
    usage.record(
        "annotation",
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    )
    usage.record("annotation", {"input_tokens": 5, "output_tokens": 2})
    usage.record("annotation", {"total_tokens": 11})

    assert usage.snapshot() == {
        "total": {
            "input_tokens": 145,
            "output_tokens": 22,
            "total_tokens": 178,
            "responses": 4,
            "responses_without_usage": 0,
        },
        "early_decision": {
            "input_tokens": 40,
            "output_tokens": 0,
            "total_tokens": 40,
            "responses": 1,
            "responses_without_usage": 0,
        },
        "annotation": {
            "input_tokens": 105,
            "output_tokens": 22,
            "total_tokens": 138,
            "responses": 3,
            "responses_without_usage": 0,
        },
    }


def test_missing_or_invalid_metadata_does_not_invent_token_counts():
    usage = UsageTracker()
    for metadata in (
        None,
        {},
        "unavailable",
        {"input_tokens": True, "output_tokens": -1, "total_tokens": "10"},
    ):
        usage.record("early_decision", metadata)
    usage.record("early_decision", {"input_tokens": 0})
    usage.record("early_decision", {"input_tokens": None, "output_tokens": 8})

    assert usage.snapshot()["early_decision"] == {
        "input_tokens": 0,
        "output_tokens": 8,
        "total_tokens": 8,
        "responses": 6,
        "responses_without_usage": 4,
    }


def test_concurrent_updates_are_not_lost_and_snapshots_remain_independent():
    usage = UsageTracker()
    before = usage.snapshot()

    def record(index):
        stage = "early_decision" if index % 2 else "annotation"
        usage.record(stage, {"input_tokens": 3, "output_tokens": 2})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(2000)))

    assert usage.snapshot()["total"] == {
        "input_tokens": 6000,
        "output_tokens": 4000,
        "total_tokens": 10000,
        "responses": 2000,
        "responses_without_usage": 0,
    }
    assert usage.snapshot()["early_decision"]["responses"] == 1000
    assert usage.snapshot()["annotation"]["responses"] == 1000
    assert before["total"]["responses"] == 0
    before["annotation"]["input_tokens"] = 999
    assert usage.snapshot()["annotation"]["input_tokens"] == 3000
