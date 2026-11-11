import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import httpx2 as httpx
import pytest
from openai import AuthenticationError, OpenAI, RateLimitError

from speeling_good.preprocessing.llm_annotation_provider import (
    LlmAnnotationProvider,
)
from speeling_good.preprocessing.types import RawPassage, SpellingAnnotation


def correction(original, corrected, before="", after=""):
    return {
        "original": original,
        "correction": corrected,
        "before": before,
        "after": after,
    }


def response(corrections):
    content = (
        corrections
        if isinstance(corrections, str)
        else json.dumps({"corrections": corrections})
    )
    return httpx.Response(
        200,
        json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


@pytest.fixture
def make_provider(monkeypatch):
    clients = []
    monkeypatch.setattr(
        "speeling_good.preprocessing.llm_annotation_provider.time.sleep", lambda _: None
    )

    def make(handler, **options):
        client = OpenAI(
            api_key="test-token",
            base_url="http://test.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return LlmAnnotationProvider(client=client, model="test-model", **options)

    yield make
    for client in clients:
        client.close()


def test_exact_context_unicode_and_sorted_spans(make_provider):
    text = "🐈 teh cat saw teh dog and wierd birds."
    provider = make_provider(
        lambda _: response(
            [
                correction("wierd", "weird"),
                correction("teh", "the", "saw ", " dog"),
                correction("teh", "the", "🐈 ", " cat"),
            ]
        )
    )
    result = provider.annotate(RawPassage(text))
    assert result.text == text
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 2, 5),
        SpellingAnnotation("teh", "the", 14, 17),
        SpellingAnnotation("wierd", "weird", 26, 31),
    ]


@pytest.mark.parametrize(
    "bad",
    [
        [correction("teh", "the")],  # Resolution failure: ambiguous context.
        '{"corrections": [',  # JSON decoding failure.
        {"unexpected": "schema"},  # Schema validation failure.
    ],
)
def test_invalid_output_recovers_with_feedback(make_provider, bad):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response(
            bad if len(calls) == 1 else [correction("teh", "the", "", " cat")]
        )

    result = make_provider(handler).annotate(RawPassage("teh cat, teh dog"))
    assert result.annotations == [SpellingAnnotation("teh", "the", 0, 3)]
    assert len(calls) == 2
    assert "Validation failed" in calls[1]["messages"][-1]["content"]
    assert result.failures
    assert all(failure.attempt == 1 for failure in result.failures)
    assert all(
        failure.response == response(bad).json()["choices"][0]["message"]["content"]
        for failure in result.failures
    )


def test_output_exhaustion_retains_partial_annotations(make_provider):
    calls = []

    def handler(request):
        calls.append(request)
        return response([correction("teh", "the"), correction("missing", "word")])

    result = make_provider(handler).annotate(RawPassage("teh cat"))
    assert len(calls) == 3
    assert result.annotations == [SpellingAnnotation("teh", "the", 0, 3)]
    assert [failure.attempt for failure in result.failures] == [1, 2, 3]
    assert all(failure.stage == "resolution" for failure in result.failures)
    assert all(
        "corrections.1" in failure.error and "missing" in failure.error
        for failure in result.failures
    )


def test_schema_error_feedback_identifies_the_invalid_field(make_provider):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            item = correction("teh", "the")
            del item["correction"]
            return response([item])
        return response([correction("teh", "the")])

    result = make_provider(handler).annotate(RawPassage("teh cat"))
    assert result.annotations == [SpellingAnnotation("teh", "the", 0, 3)]
    feedback = calls[1]["messages"][-1]["content"]
    assert "corrections.0.correction" in feedback
    assert "Field required" in feedback


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_transient_http_failure_recovers(make_provider, status):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, json={"error": {"message": "try later"}})
        return response([])

    result = make_provider(handler).annotate(RawPassage("PyTorch uses CUDA."))
    assert result.annotations == []
    assert len(calls) == 2
    assert result.failures == []


def test_connection_failure_recovers(make_provider):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("unavailable", request=request)
        return response([])

    assert make_provider(handler).annotate(RawPassage("Hello")).annotations == []
    assert len(calls) == 2


@pytest.mark.parametrize(
    "finish_reason,refusal",
    [
        ("length", None),
        ("content_filter", None),
        ("stop", "Cannot comply"),
    ],
)
def test_incomplete_or_refused_response_recovers(make_provider, finish_reason, refusal):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) > 1:
            return response([correction("teh", "the")])
        data = response([]).json()
        data["choices"][0]["finish_reason"] = finish_reason
        data["choices"][0]["message"]["refusal"] = refusal
        return httpx.Response(200, json=data)

    result = make_provider(handler).annotate(RawPassage("teh cat"))
    assert result.annotations == [SpellingAnnotation("teh", "the", 0, 3)]
    assert len(calls) == 2
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.stage == "decoding"
    assert failure.response == '{"corrections": []}'
    assert (refusal or finish_reason) in failure.error


@pytest.mark.parametrize(
    "bad",
    ['{"corrections": [', {"unexpected": "schema"}, [correction("missing", "word")]],
)
def test_exhaustion_without_usable_annotations_is_not_a_clean_passage(
    make_provider, bad
):
    failed = make_provider(lambda _: response(bad), max_attempts=2).annotate(
        RawPassage("clean text")
    )
    clean = make_provider(lambda _: response([])).annotate(RawPassage("clean text"))
    assert failed.annotations is None
    assert [failure.attempt for failure in failed.failures] == [1, 2]
    assert clean.annotations == []
    assert clean.failures == []


def test_schema_and_resolution_failures_preserve_valid_entries(make_provider):
    items = [
        correction("teh", "the"),
        {"original": "bad", "before": "", "after": ""},
        correction("absent", "word"),
    ]
    result = make_provider(lambda _: response(items), max_attempts=1).annotate(
        RawPassage("teh cat")
    )
    assert result.annotations == [SpellingAnnotation("teh", "the", 0, 3)]
    assert [failure.stage for failure in result.failures] == ["decoding", "resolution"]
    assert "corrections.1" in result.failures[0].error
    assert "corrections.2" in result.failures[1].error


@pytest.mark.parametrize("latest_count", [1, 2])
def test_best_partial_attempt_is_selected_without_merging(make_provider, latest_count):
    attempts = iter(
        [
            [
                correction("teh", "the"),
                correction("wierd", "weird"),
                correction("absent", "word"),
            ],
            [correction("speling", "spelling"), correction("wierd", "weird")][
                :latest_count
            ]
            + [correction("absent", "word")],
            '{"corrections": [',
        ]
    )
    result = make_provider(lambda _: response(next(attempts))).annotate(
        RawPassage("teh wierd speling")
    )
    expected = (
        [
            SpellingAnnotation("teh", "the", 0, 3),
            SpellingAnnotation("wierd", "weird", 4, 9),
        ]
        if latest_count == 1
        else [
            SpellingAnnotation("wierd", "weird", 4, 9),
            SpellingAnnotation("speling", "spelling", 10, 17),
        ]
    )
    assert result.annotations == expected
    assert [failure.attempt for failure in result.failures] == [1, 2, 3]


def test_valid_empty_retry_replaces_previous_partial_result(make_provider):
    attempts = iter([[correction("teh", "the"), correction("absent", "word")], []])
    result = make_provider(lambda _: response(next(attempts))).annotate(
        RawPassage("teh cat")
    )
    assert result.annotations == []
    assert len(result.failures) == 1


@pytest.mark.parametrize("missing", ["choices", "content"])
def test_missing_response_data_is_retained_as_a_decoding_failure(
    make_provider, missing
):
    data = response([]).json()
    if missing == "choices":
        data["choices"] = []
    else:
        data["choices"][0]["message"]["content"] = None
    result = make_provider(
        lambda _: httpx.Response(200, json=data), max_attempts=1
    ).annotate(RawPassage("text"))
    assert result.annotations is None
    assert len(result.failures) == 1
    assert result.failures[0].stage == "decoding"
    assert result.failures[0].response is None


def test_concurrent_passages_keep_failure_histories_separate(make_provider):
    def handler(request):
        text = json.loads(request.content)["messages"][1]["content"]
        return response([correction(text + " absent", "word")])

    provider = make_provider(handler, max_attempts=1)
    passages = [RawPassage(f"passage {i}") for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(provider.annotate, passages))
    for passage, result in zip(passages, results, strict=True):
        assert result.annotations is None
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert passage.text + " absent" in failure.error
        assert (
            json.loads(failure.response)["corrections"][0]["original"]
            == passage.text + " absent"
        )


@pytest.mark.parametrize("retry_after,expected_delay", [("2", 2.0), ("90", 30.0)])
def test_retry_after_controls_backoff(
    make_provider, monkeypatch, retry_after, expected_delay
):
    calls = []
    delays = []
    monkeypatch.setattr(
        "speeling_good.preprocessing.llm_annotation_provider.time.sleep", delays.append
    )

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": retry_after},
                json={"error": {"message": "try later"}},
            )
        return response([])

    make_provider(handler).annotate(RawPassage("Hello"))
    assert delays == [expected_delay]


@pytest.mark.parametrize(
    "status,exception,attempts",
    [
        (401, AuthenticationError, 1),
        (429, RateLimitError, 3),
    ],
)
def test_http_failure_preserves_error_and_request_budget(
    make_provider, status, exception, attempts
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "failed"}})

    with pytest.raises(exception):
        make_provider(handler).annotate(RawPassage("Hello"))
    assert len(calls) == attempts


def test_shared_provider_limits_active_requests_and_keeps_results_separate(
    make_provider,
):
    lock = Lock()
    at_capacity = Event()
    release = Event()
    active = 0
    peak = 0

    def handler(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                at_capacity.set()
        try:
            assert release.wait(5), "Requests were not released"
            text = json.loads(request.content)["messages"][1]["content"]
            return response([correction("teh", "the", text.split("teh")[0])])
        finally:
            with lock:
                active -= 1

    provider = make_provider(handler, max_inflight=2)
    passages = [RawPassage(f"{i}: teh cat") for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(provider.annotate, passage) for passage in passages]
        try:
            assert at_capacity.wait(5), "Requests did not run concurrently"
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert peak == 2
    assert [result.text for result in results] == [passage.text for passage in passages]
    assert all(
        result.annotations == [SpellingAnnotation("teh", "the", 3, 6)]
        for result in results
    )
