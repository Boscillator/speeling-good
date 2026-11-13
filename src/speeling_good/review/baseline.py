"""Lossless proposal identities and tool-independent human review records."""

import hashlib
import html
import json
from typing import Any

from speeling_good.preprocessing.types import CorrectionType


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def proposal_tasks(record: dict, source: str, shard: str, row: int) -> list[dict]:
    text = record["text"]
    passage_id = digest([source, shard, row])
    tasks = []
    for index, proposal in enumerate(record["annotations"] or []):
        start, end = proposal["span_start"], proposal["span_end"]
        if not 0 <= start < end <= len(text) or text[start:end] != proposal["original"]:
            raise ValueError(f"Invalid source span: {shard}:{row}:{index}")
        CorrectionType(proposal["correction_type"])
        highlighted = (
            '<div style="white-space:pre-wrap">'
            + html.escape(text[:start])
            + "<mark>"
            + html.escape(text[start:end])
            + "</mark>"
            + html.escape(text[end:])
            + "</div>"
        )
        tasks.append(
            {
                "passage_id": passage_id,
                "proposal_id": digest([passage_id, index]),
                "source": {
                    "id": source,
                    "shard": shard,
                    "row": row,
                    "annotation_index": index,
                },
                "source_hash": digest(record),
                "source_record": record,
                "text": text,
                "proposal": proposal,
                "highlighted": highlighted,
                "original": proposal["original"],
                "replacement": proposal["corrected"],
                "predicted_type": proposal["correction_type"],
            }
        )
    return tasks


def python_offset(text: str, offset: int) -> int:
    """Label Studio Text offsets count JavaScript UTF-16 code units."""
    if not 0 <= offset <= len(text.encode("utf-16-le")) // 2:
        raise ValueError("Text offset is outside the passage")
    try:
        return len(text.encode("utf-16-le")[: offset * 2].decode("utf-16-le"))
    except UnicodeDecodeError as error:
        raise ValueError("Offset splits a Unicode character") from error


def baseline_record(task: dict) -> dict:
    data = task["data"]
    reviews = []
    for annotation in task.get("annotations", []):
        values = {}
        for result in annotation.get("result", []):
            name = result["from_name"]
            if name in values:
                raise ValueError(
                    f"Multiple values for {name} in annotation {annotation['id']}"
                )
            values[name] = result["value"]
        canceled = annotation.get("was_cancelled", False)
        judgments = {}
        for name, allowed in (
            ("inclusion", {"in_scope", "outside_scope", "not_an_error", "unsure"}),
            ("type", {"correct", "incorrect", "unsure"}),
            ("span", {"correct", "incorrect", "unsure"}),
            ("replacement", {"correct", "incorrect", "unsure"}),
        ):
            if (
                name == "replacement"
                and judgments["inclusion"] != "in_scope"
                and not canceled
            ):
                judgments[name] = None
                continue
            choices = values.get(name, {}).get("choices", [])
            value = choices[0] if len(choices) == 1 else None
            if not canceled and value not in allowed:
                raise ValueError(
                    f"Missing/invalid {name} in annotation {annotation['id']}"
                )
            judgments[name] = value
        corrected = {}
        if not canceled and judgments["type"] == "incorrect":
            choices = values.get("correct_type", {}).get("choices", [])
            if len(choices) != 1:
                raise ValueError("Incorrect type requires a corrected type")
            corrected["correction_type"] = CorrectionType(choices[0]).value
        if not canceled and judgments["span"] == "incorrect":
            span = values.get("correct_span")
            if not span:
                raise ValueError("Incorrect span requires one corrected span")
            text = data["text"]
            start, end = (
                python_offset(text, span["start"]),
                python_offset(text, span["end"]),
            )
            if not 0 <= start < end <= len(text) or text[start:end] != span["text"]:
                raise ValueError("Corrected span does not match original passage")
            corrected["span"] = {
                "span_start": start,
                "span_end": end,
                "original": text[start:end],
            }
        if not canceled and judgments["replacement"] == "incorrect":
            deletion = values.get("delete_replacement", {}).get("choices") == ["delete"]
            replacement = values.get("correct_replacement", {}).get("text", [])
            if not deletion and len(replacement) != 1:
                raise ValueError(
                    "Incorrect replacement requires text or explicit deletion"
                )
            if deletion and replacement:
                raise ValueError("Choose replacement text or deletion, not both")
            corrected["replacement"] = "" if deletion else replacement[0]
        reviews.append(
            {
                "annotation_id": annotation["id"],
                "reviewer": annotation.get("completed_by"),
                "created_at": annotation.get("created_at"),
                "updated_at": annotation.get("updated_at"),
                "status": "canceled" if canceled else "completed",
                "judgments": judgments,
                "corrected": corrected,
                "notes": values.get("notes", {}).get("text", []),
                "raw_results": annotation.get("result", []),
            }
        )
    return {
        "schema_version": 1,
        **{
            key: data[key]
            for key in (
                "passage_id",
                "proposal_id",
                "source",
                "source_hash",
                "source_record",
                "text",
                "proposal",
            )
        },
        "reviews": reviews,
    }
