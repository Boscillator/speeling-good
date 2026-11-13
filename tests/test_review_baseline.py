import copy
from typing import Any

import pytest

from speeling_good.review.baseline import baseline_record, proposal_tasks, python_offset


def test_proposal_identity_preserves_occurrences_and_detects_source_changes():
    record: dict[str, Any] = {
        "text": "teh teh",
        "annotations": [
            {
                "original": "teh",
                "corrected": "the",
                "span_start": start,
                "span_end": start + 3,
                "correction_type": "spelling",
            }
            for start in (0, 4)
        ],
    }
    tasks = proposal_tasks(record, "run", "part-00000.jsonl", 1)
    assert tasks[0]["passage_id"] == tasks[1]["passage_id"]
    assert tasks[0]["proposal_id"] != tasks[1]["proposal_id"]
    changed = copy.deepcopy(record)
    changed["annotations"][0]["corrected"] = "ten"
    new = proposal_tasks(changed, "run", "part-00000.jsonl", 1)[0]
    assert new["proposal_id"] == tasks[0]["proposal_id"]
    assert new["source_hash"] != tasks[0]["source_hash"]
    assert (
        proposal_tasks(record, "other-run", "part-00000.jsonl", 1)[0]["passage_id"]
        != tasks[0]["passage_id"]
    )


def test_utf16_offsets_preserve_unicode_and_reject_split_characters():
    assert python_offset("é😀\tword", 4) == 3
    with pytest.raises(ValueError):
        python_offset("é😀\tword", 2)
    with pytest.raises(ValueError):
        python_offset("é😀", 4)


def test_in_scope_replacement_requires_text_or_explicit_deletion():
    data = proposal_tasks(
        {
            "text": "word",
            "annotations": [
                {
                    "original": "word",
                    "corrected": "words",
                    "span_start": 0,
                    "span_end": 4,
                    "correction_type": "style",
                }
            ],
        },
        "run",
        "part-00000.jsonl",
        1,
    )[0]
    results = [
        {"from_name": name, "value": {"choices": [value]}}
        for name, value in (
            ("inclusion", "in_scope"),
            ("type", "unsure"),
            ("span", "correct"),
            ("replacement", "incorrect"),
            ("delete_replacement", "delete"),
        )
    ]
    exported = baseline_record(
        {"data": data, "annotations": [{"id": 1, "result": results}]}
    )
    assert exported["proposal"]["corrected"] == "words"
    assert exported["reviews"][0]["corrected"]["replacement"] == ""
    assert exported["reviews"][0]["judgments"]["inclusion"] == "in_scope"
    assert exported["reviews"][0]["judgments"]["type"] == "unsure"
    results.pop()
    with pytest.raises(ValueError):
        baseline_record({"data": data, "annotations": [{"id": 1, "result": results}]})
    assert baseline_record({"data": data, "annotations": []})["reviews"] == []
    canceled = baseline_record(
        {
            "data": data,
            "annotations": [{"id": 2, "result": results, "was_cancelled": True}],
        }
    )
    assert canceled["reviews"][0]["status"] == "canceled"


@pytest.mark.parametrize("inclusion", ["outside_scope", "not_an_error", "unsure"])
@pytest.mark.parametrize("stale_replacement", [False, True])
def test_non_in_scope_reviews_need_no_replacement_and_preserve_raw_answers(
    inclusion, stale_replacement
):
    data = dict.fromkeys(
        (
            "passage_id",
            "proposal_id",
            "source",
            "source_hash",
            "source_record",
            "text",
            "proposal",
        )
    )
    results = [
        {"from_name": name, "value": {"choices": [value]}}
        for name, value in (
            ("inclusion", inclusion),
            ("type", "correct"),
            ("span", "correct"),
        )
    ]
    if stale_replacement:
        results.append(
            {"from_name": "replacement", "value": {"choices": ["incorrect"]}}
        )
        results.append(
            {"from_name": "correct_replacement", "value": {"text": ["old answer"]}}
        )
    review = baseline_record(
        {"data": data, "annotations": [{"id": 1, "result": results}]}
    )["reviews"][0]
    assert review["judgments"]["replacement"] is None
    assert "replacement" not in review["corrected"]
    assert review["raw_results"] == results


def test_interface_layout_changes_preserve_annotation_schema():
    from speeling_good.cli.review_corrections import interface_schema

    before = '<View><Text name="passage" value="$text"/><Choices name="span" toName="passage"><Choice value="correct"/></Choices></View>'
    after = '<View><Choices name="span" toName="passage"><Choice value="correct"/></Choices><View visibleWhen="choice-selected"><Text name="passage" value="$text"/></View></View>'
    assert interface_schema(before) == interface_schema(after)
    assert interface_schema(before) != interface_schema(
        after.replace('value="correct"', 'value="incorrect"')
    )
    assert interface_schema(before) != interface_schema(
        after.replace('toName="passage"', 'toName="other"')
    )
