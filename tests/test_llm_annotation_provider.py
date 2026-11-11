import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import httpx2 as httpx
import pytest
from openai import AuthenticationError, OpenAI, RateLimitError

from speeling_good.preprocessing.llm_annotation_provider import (
    AnnotationError,
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
        [correction("teh", "the")],  # Repeated without enough context.
        [correction("teh", "the", "invented context")],
        [correction("teh", "the", "", " cat"), correction("teh cat", "the cat")],
        [correction("teh", "the", "", " cat")] * 2,
        [correction("teh", "teh", "", " cat")],
        [correction("", "the")],
        [correction("teh", "", "", " cat")],
        '{"corrections": [',
        {"unexpected": "schema"},
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


def test_output_exhaustion_does_not_return_partial_annotations(make_provider):
    calls = []

    def handler(request):
        calls.append(request)
        return response([correction("teh", "the"), correction("missing", "word")])

    with pytest.raises(AnnotationError) as error:
        make_provider(handler).annotate(RawPassage("teh cat"))
    assert len(calls) == 3
    assert isinstance(error.value.__cause__, AnnotationError)


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
