import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Event, Lock

import httpx2 as httpx
import pytest
from openai import AuthenticationError, OpenAI

from speeling_good.preprocessing.llm_annotation_provider import (
    AnnotationConfig,
    LlmAnnotationProvider,
)
from speeling_good.preprocessing.types import (
    CorrectionType,
    EarlyDecision,
    EarlyDecisionFailure,
    PassageWithEarlyDecision,
    SpellingAnnotation,
)
from speeling_good.preprocessing.usage import UsageTracker


@pytest.mark.parametrize("outcome", ["success", "recovery", "partial", "exhaustion"])
@pytest.mark.parametrize("decision", [None, EarlyDecision(False, 0.83)])
def test_early_decision_is_preserved_without_influencing_requests(
    make_provider, outcome, decision
):
    calls = []
    failures = [EarlyDecisionFailure(1, "Invalid JSON", "{")]
    passage = PassageWithEarlyDecision("teh cat", decision, failures)

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if outcome == "exhaustion" or (outcome == "recovery" and len(calls) == 1):
            return response("{")
        items = [correction("teh", "the")]
        if outcome == "partial":
            items.append(correction("absent", "word"))
        return response(items)

    result = make_provider(handler, max_attempts=2).annotate(passage)
    assert result.decision == decision
    assert result.early_decision_failures == failures
    assert result.annotations == (
        None
        if outcome == "exhaustion"
        else [
            SpellingAnnotation(
                "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
            )
        ]
    )
    assert bool(result.failures) is (outcome != "success")
    assert all(
        call["messages"][1] == {"role": "user", "content": passage.text}
        for call in calls
    )


def correction(
    original, corrected, before="", after="", correction_type=CorrectionType.SPELLING
):
    return {
        "original": original,
        "correction": corrected,
        "correction_type": correction_type,
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


@pytest.mark.parametrize("category", list(CorrectionType))
def test_predicted_type_survives_resolution_and_serialization(make_provider, category):
    result = make_provider(
        lambda _: response([correction("teh", "the", correction_type=category)])
    ).annotate(PassageWithEarlyDecision("teh cat", None))

    assert result.annotations == [SpellingAnnotation("teh", "the", 0, 3, category)]
    saved = json.loads(json.dumps(asdict(result)))
    assert saved["annotations"][0]["correction_type"] == category.value
    assert not result.failures


@pytest.mark.parametrize("invalid_type", ["missing", "unknown", None, 3])
@pytest.mark.parametrize("recover", [False, True])
def test_invalid_type_preserves_other_edits_and_retries(
    make_provider, invalid_type, recover
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        wrong = correction("teh", "the")
        if not recover or len(calls) == 1:
            if invalid_type == "missing":
                del wrong["correction_type"]
            else:
                wrong["correction_type"] = invalid_type
        return response(
            [
                wrong,
                correction("walk", "walks", correction_type=CorrectionType.INFLECTION),
            ]
        )

    result = make_provider(handler, max_attempts=2).annotate(
        PassageWithEarlyDecision("teh cat walk", None)
    )

    expected = [SpellingAnnotation("walk", "walks", 8, 12, CorrectionType.INFLECTION)]
    if recover:
        expected.insert(
            0, SpellingAnnotation("teh", "the", 0, 3, CorrectionType.SPELLING)
        )
    assert result.annotations == expected
    assert len(calls) == 2
    assert "corrections.0.correction_type" in calls[1]["messages"][-1]["content"]
    assert all(f.stage == "decoding" for f in result.failures)
    assert len(result.failures) == (1 if recover else 2)


@pytest.fixture
def make_provider(monkeypatch):
    clients = []
    monkeypatch.setattr(
        "speeling_good.preprocessing.llm_annotation_provider.time.sleep", lambda _: None
    )

    def make(handler, *, max_inflight=4, usage=None, **options):
        client = OpenAI(
            api_key="test-token",
            base_url="http://test.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return LlmAnnotationProvider(
            client=client,
            config=AnnotationConfig(
                model="test-model",
                prompt=options.pop("prompt", "Find spelling mistakes."),
                **options,
            ),
            max_inflight=max_inflight,
            usage=usage,
        )

    yield make
    for client in clients:
        client.close()


def test_usage_counts_invalid_output_but_not_transport_failures(make_provider):
    usage = UsageTracker()
    invalid = response("{").json()
    invalid["usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cost": 0.125,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    responses = iter(
        [
            httpx.Response(429, json={"error": {"message": "try later"}}),
            httpx.Response(200, json=invalid),
            response([]),
        ]
    )
    result = make_provider(lambda _: next(responses), usage=usage).annotate(
        PassageWithEarlyDecision("clean text", None)
    )

    assert result.annotations == []
    assert usage.snapshot()["annotation"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cached_tokens": 40,
        "reasoning_tokens": 5,
        "cost": 0.125,
        "responses": 2,
        "responses_without_usage": 1,
        "responses_without_cost": 1,
    }


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
    result = provider.annotate(PassageWithEarlyDecision(text, None))
    assert result.text == text
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 2, 5, correction_type=CorrectionType.SPELLING),
        SpellingAnnotation(
            "teh", "the", 14, 17, correction_type=CorrectionType.SPELLING
        ),
        SpellingAnnotation(
            "wierd", "weird", 26, 31, correction_type=CorrectionType.SPELLING
        ),
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

    result = make_provider(handler).annotate(
        PassageWithEarlyDecision("teh cat, teh dog", None)
    )
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 0, 3, correction_type=CorrectionType.SPELLING)
    ]
    assert len(calls) == 2
    assert "Validation failed" in calls[1]["messages"][-1]["content"]
    assert calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": response(bad).json()["choices"][0]["message"]["content"],
    }
    assert result.failures
    assert all(failure.attempt == 1 for failure in result.failures)
    assert all(
        failure.response == response(bad).json()["choices"][0]["message"]["content"]
        for failure in result.failures
    )


@pytest.mark.parametrize("temperature", [None, 0.7])
@pytest.mark.parametrize("reasoning_effort", ["none", "minimal", "max"])
def test_custom_prompt_and_sampling_are_preserved_on_retry(
    make_provider, temperature, reasoning_effort
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response("{" if len(calls) == 1 else [])

    prompt = "Find spelling errors.\nPreserve technical terms."
    provider = make_provider(
        handler,
        prompt=prompt,
        temperature=temperature,
        top_p=0.9,
        max_completion_tokens=256,
        reasoning_effort=reasoning_effort,
        seed=0,
    )
    result = provider.annotate(PassageWithEarlyDecision("clean text", None))
    assert result.annotations == []
    assert len(calls) == 2
    for call in calls:
        assert call["messages"][0] == {"role": "system", "content": prompt}
        assert call["model"] == "test-model"
        if temperature is None:
            assert "temperature" not in call
        else:
            assert call["temperature"] == temperature
        assert call["top_p"] == 0.9
        assert call["max_completion_tokens"] == 256
        assert call["seed"] == 0
        assert call["reasoning"] == {"effort": reasoning_effort}
        assert call["response_format"]["type"] == "json_schema"
    assert "Validation failed" in calls[1]["messages"][-1]["content"]


def test_optional_sampling_parameters_are_omitted_by_default(make_provider):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response([])

    make_provider(handler).annotate(PassageWithEarlyDecision("clean text", None))
    assert calls[0]["temperature"] == 0
    assert {"top_p", "max_completion_tokens", "seed", "reasoning"}.isdisjoint(calls[0])


def test_output_exhaustion_retains_partial_annotations(make_provider):
    calls = []

    def handler(request):
        calls.append(request)
        return response([correction("teh", "the"), correction("missing", "word")])

    result = make_provider(handler).annotate(PassageWithEarlyDecision("teh cat", None))
    assert len(calls) == 3
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 0, 3, correction_type=CorrectionType.SPELLING)
    ]
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

    result = make_provider(handler).annotate(PassageWithEarlyDecision("teh cat", None))
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 0, 3, correction_type=CorrectionType.SPELLING)
    ]
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

    result = make_provider(handler).annotate(
        PassageWithEarlyDecision("PyTorch uses CUDA.", None)
    )
    assert result.annotations == []
    assert len(calls) == 2
    assert len(result.failures) == 1
    assert result.failures[0].stage == "transport"


def test_connection_failure_recovers(make_provider):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("unavailable", request=request)
        return response([])

    assert (
        make_provider(handler)
        .annotate(PassageWithEarlyDecision("Hello", None))
        .annotations
        == []
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "finish_reason,refusal",
    [
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

    result = make_provider(handler).annotate(PassageWithEarlyDecision("teh cat", None))
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 0, 3, correction_type=CorrectionType.SPELLING)
    ]
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
        PassageWithEarlyDecision("clean text", None)
    )
    clean = make_provider(lambda _: response([])).annotate(
        PassageWithEarlyDecision("clean text", None)
    )
    assert failed.annotations is None
    assert [failure.attempt for failure in failed.failures] == [1, 2]
    assert clean.annotations == []
    assert clean.failures == []


def test_schema_and_resolution_failures_preserve_valid_entries(make_provider):
    items = [
        correction("teh", "the"),
        {
            "original": "bad",
            "before": "",
            "after": "",
            "correction_type": "spelling",
        },
        correction("absent", "word"),
    ]
    result = make_provider(lambda _: response(items), max_attempts=1).annotate(
        PassageWithEarlyDecision("teh cat", None)
    )
    assert result.annotations == [
        SpellingAnnotation("teh", "the", 0, 3, correction_type=CorrectionType.SPELLING)
    ]
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
        PassageWithEarlyDecision("teh wierd speling", None)
    )
    expected = (
        [
            SpellingAnnotation(
                "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
            ),
            SpellingAnnotation(
                "wierd", "weird", 4, 9, correction_type=CorrectionType.SPELLING
            ),
        ]
        if latest_count == 1
        else [
            SpellingAnnotation(
                "wierd", "weird", 4, 9, correction_type=CorrectionType.SPELLING
            ),
            SpellingAnnotation(
                "speling", "spelling", 10, 17, correction_type=CorrectionType.SPELLING
            ),
        ]
    )
    assert result.annotations == expected
    assert [failure.attempt for failure in result.failures] == [1, 2, 3]


def test_valid_empty_retry_replaces_previous_partial_result(make_provider):
    attempts = iter([[correction("teh", "the"), correction("absent", "word")], []])
    result = make_provider(lambda _: response(next(attempts))).annotate(
        PassageWithEarlyDecision("teh cat", None)
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
    ).annotate(PassageWithEarlyDecision("text", None))
    assert result.annotations is None
    assert len(result.failures) == 1
    assert result.failures[0].stage == "decoding"
    assert result.failures[0].response is None


def test_concurrent_passages_keep_failure_histories_separate(make_provider):
    def handler(request):
        text = json.loads(request.content)["messages"][1]["content"]
        return response([correction(text + " absent", "word")])

    provider = make_provider(handler, max_attempts=1)
    passages = [PassageWithEarlyDecision(f"passage {i}", None) for i in range(8)]
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

    make_provider(handler).annotate(PassageWithEarlyDecision("Hello", None))
    assert delays == [expected_delay]


@pytest.mark.parametrize(
    "status,exception,attempts",
    [
        (401, AuthenticationError, 1),
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
        make_provider(handler).annotate(PassageWithEarlyDecision("Hello", None))
    assert len(calls) == attempts


@pytest.mark.parametrize("status", [None, 429, 503])
@pytest.mark.parametrize("partial", [False, True])
def test_transport_exhaustion_retains_partial_results_and_diagnostics(
    make_provider, status, partial
):
    calls = []

    def handler(request):
        calls.append(request)
        if partial and len(calls) == 1:
            return response([correction("teh", "the"), correction("absent", "word")])
        if status is None:
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(status, json={"error": {"message": "unavailable"}})

    decision = EarlyDecision(True, 0.9)
    result = make_provider(handler, max_attempts=3).annotate(
        PassageWithEarlyDecision("teh cat", decision)
    )
    assert len(calls) == 3
    assert result.annotations == (
        [
            SpellingAnnotation(
                "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
            )
        ]
        if partial
        else None
    )
    assert result.decision == decision
    assert [failure.attempt for failure in result.failures] == [1, 2, 3]
    assert result.failures[-1].stage == "transport"


@pytest.mark.parametrize("partial", [False, True])
def test_token_limit_stops_repeated_capped_calls_and_preserves_best_result(
    make_provider, partial
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if partial and len(calls) == 1:
            return response([correction("teh", "the"), correction("absent", "word")])
        data = response('{"corrections": [').json()
        data["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=data)

    result = make_provider(handler, max_completion_tokens=256).annotate(
        PassageWithEarlyDecision("teh cat", None)
    )
    assert len(calls) == (2 if partial else 1)
    assert result.annotations == (
        [
            SpellingAnnotation(
                "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
            )
        ]
        if partial
        else None
    )
    assert "length" in result.failures[-1].error


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
    passages = [PassageWithEarlyDecision(f"{i}: teh cat", None) for i in range(8)]
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
        result.annotations
        == [
            SpellingAnnotation(
                "teh", "the", 3, 6, correction_type=CorrectionType.SPELLING
            )
        ]
        for result in results
    )
