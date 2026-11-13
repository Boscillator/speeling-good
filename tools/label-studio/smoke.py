"""Live API smoke check. Run with the main project's uv environment, not pytest.

Requires LABEL_STUDIO_API_TOKEN and a running service. Creates a uniquely named
project and leaves it available for visual inspection. Never calls a model.
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from speeling_good.cli.review_corrections import (
    LabelStudio,
    export_reviews,
    import_proposals,
)


def main():
    client = LabelStudio()
    title = f"spelling-smoke-{uuid4().hex[:8]}"
    project = client.project(title, create=True)
    assert client.project(title, create=True)["id"] == project["id"]
    text = "  Café 😀\t teh cat\nwas verry verry nice.  "
    proposals = []
    for original, replacement, kind in (
        ("teh", "the", "spelling"),
        ("verry", "very", "grammar"),
        ("nice", "pleasant", "style"),
    ):
        start = text.index(original)
        proposals.append(
            {
                "original": original,
                "corrected": replacement,
                "span_start": start,
                "span_end": start + len(original),
                "correction_type": kind,
            }
        )
    source: dict[str, Any] = {
        "text": text,
        "annotations": proposals,
        "failures": [],
        "decision": None,
        "early_decision_failures": [],
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "part-00000.jsonl"
        path.write_text(json.dumps(source, ensure_ascii=False) + "\n")
        import_proposals(client, project["id"], path, title)
        tasks = list(client.tasks(project["id"]))
        assert len(tasks) == 3
        for index, task in enumerate(tasks):

            def choice(name, value):
                return {
                    "from_name": name,
                    "to_name": "passage",
                    "type": "choices",
                    "value": {"choices": [value]},
                }

            results = [
                choice("inclusion", "outside_scope" if index == 2 else "in_scope"),
                choice("type", "incorrect" if index == 1 else "correct"),
                choice("span", "incorrect" if index == 1 else "correct"),
            ]
            if index != 2:
                results.append(
                    choice("replacement", "incorrect" if index == 1 else "unsure")
                )
            if index == 1:
                start = text.index("verry")
                end = start + len("verry verry")
                results.extend(
                    [
                        choice("correct_type", "doubled_word"),
                        {
                            "from_name": "correct_span",
                            "to_name": "passage",
                            "type": "labels",
                            "value": {
                                "start": len(text[:start].encode("utf-16-le")) // 2,
                                "end": len(text[:end].encode("utf-16-le")) // 2,
                                "text": text[start:end],
                                "labels": ["corrected_span"],
                            },
                        },
                        {
                            "from_name": "correct_replacement",
                            "to_name": "passage",
                            "type": "textarea",
                            "value": {"text": [" very\t"]},
                        },
                    ]
                )
            client.request(
                "POST", f"/api/tasks/{task['id']}/annotations/", {"result": results}
            )
        before = list(client.tasks(project["id"]))
        import_proposals(client, project["id"], path, title)
        after = list(client.tasks(project["id"]))
        assert before == after
        output = Path("data/reviews") / f"{title}.jsonl"
        export_reviews(client, project["id"], output)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(records) == 3
        for index, record in enumerate(records):
            assert record["text"] == text
            assert record["source_record"] == source
            assert record["proposal"] == proposals[index]
        review = records[1]["reviews"][0]
        assert review["corrected"]["correction_type"] == "doubled_word"
        assert review["corrected"]["replacement"] == " very\t"
        assert review["corrected"]["span"]["original"] == "verry verry"
        assert records[2]["reviews"][0]["judgments"]["inclusion"] == "outside_scope"
        assert records[2]["reviews"][0]["judgments"]["replacement"] is None
        source["text"] += "changed"
        path.write_text(json.dumps(source) + "\n")
        try:
            import_proposals(client, project["id"], path, title)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed source was accepted")
    print(
        f"PASS: {title}; inspect {client.url}/projects/{project['id']}; export {output}"
    )


if __name__ == "__main__":
    main()
