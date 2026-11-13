"""Manage a local Label Studio proposal review project."""

import argparse
import fcntl
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from loguru import logger

from speeling_good.preprocessing.types import CorrectionType
from speeling_good.review.baseline import baseline_record, proposal_tasks


def interface() -> str:
    template = files("speeling_good.review").joinpath("interface.xml").read_text()
    return template.replace(
        "<!-- CORRECTION_TYPES -->",
        "\n".join(f'<Choice value="{kind.value}"/>' for kind in CorrectionType),
    )


def interface_schema(config: str) -> dict:
    """Compare named controls and options while allowing layout-only changes."""
    return {
        element.attrib["name"]: (
            element.tag,
            element.attrib,
            sorted(
                (child.tag, tuple(sorted(child.attrib.items())))
                for child in element
                if child.tag in {"Choice", "Label"}
            ),
        )
        for element in ElementTree.fromstring(config).iter()
        if "name" in element.attrib
    }


class LabelStudio:
    def __init__(self) -> None:
        self.url = os.environ.get("LABEL_STUDIO_URL", "http://127.0.0.1:8080").rstrip(
            "/"
        )
        self.token = os.environ["LABEL_STUDIO_API_TOKEN"]
        self.authorization = f"Token {self.token}"
        if self.token.count(".") == 2:
            refreshed = self.request(
                "POST", "/api/token/refresh/", {"refresh": self.token}
            )
            self.authorization = f"Bearer {refreshed['access']}"

    def request(self, method: str, path: str, body=None):
        request = Request(
            self.url + path,
            data=None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": self.authorization,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=120) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as error:
            raise RuntimeError(
                f"Label Studio {method} {path}: HTTP {error.code}"
            ) from None

    def project(self, title: str, create: bool = False) -> dict:
        page = 1
        matches = []
        while True:
            result = self.request("GET", f"/api/projects/?page={page}&page_size=100")
            projects = result["results"]
            matches.extend(p for p in projects if p["title"] == title)
            if page * 100 >= result["count"]:
                break
            page += 1
        if len(matches) > 1:
            raise ValueError(f"Multiple projects named {title}; use a unique title")
        if not matches:
            if not create:
                raise ValueError(f"Project {title} does not exist; run setup")
            return self.request(
                "POST", "/api/projects/", {"title": title, "label_config": interface()}
            )
        project = self.request("GET", f"/api/projects/{matches[0]['id']}/")
        if project["label_config"].strip() != interface().strip():
            if not create:
                raise ValueError(
                    "Project interface differs; run setup to apply compatible layout changes"
                )
            if interface_schema(project["label_config"]) != interface_schema(
                interface()
            ):
                raise ValueError(
                    "Project interface schema differs; use a new project title"
                )
            project = self.request(
                "PATCH",
                f"/api/projects/{project['id']}/",
                {"label_config": interface()},
            )
            logger.info("Updated project layout; annotation controls are unchanged")
        return project

    def tasks(self, project_id: int):
        page = 1
        while True:
            result = self.request(
                "GET", f"/api/tasks/?project={project_id}&page={page}&page_size=100"
            )
            for task in result["tasks"]:
                yield self.request("GET", f"/api/tasks/{task['id']}/")
            if page * 100 >= result["total"]:
                break
            page += 1


def import_proposals(
    client: LabelStudio, project_id: int, path: Path, source: str
) -> None:
    existing = {}
    for task in client.tasks(project_id):
        key = task["data"]["proposal_id"]
        if key in existing:
            raise ValueError(f"Duplicate existing proposal ID: {key}")
        existing[key] = task["data"]
    paths = sorted(path.glob("part-*.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("No part-*.jsonl shards found")
    pending = []
    seen = set()
    # Validate the whole input before importing any tasks.
    for shard in paths:
        with shard.open(encoding="utf-8") as stream:
            for row, line in enumerate(stream, 1):
                for data in proposal_tasks(json.loads(line), source, shard.name, row):
                    key = data["proposal_id"]
                    if key in seen:
                        raise ValueError(f"Duplicate input proposal ID: {key}")
                    seen.add(key)
                    if key in existing:
                        if existing[key] != data:
                            raise ValueError(
                                f"Source changed for proposal {key}; use a new source ID"
                            )
                    else:
                        pending.append({"data": data})
    for offset in range(0, len(pending), 100):
        client.request(
            "POST", f"/api/projects/{project_id}/import", pending[offset : offset + 100]
        )
    logger.info(
        "Imported {} proposals; skipped {} existing proposals",
        len(pending),
        len(seen) - len(pending),
    )


def export_reviews(client: LabelStudio, project_id: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            seen = set()
            for task in client.tasks(project_id):
                try:
                    record = baseline_record(task)
                except ValueError as error:
                    raise ValueError(f"Task {task['id']}: {error}") from error
                if record["proposal_id"] in seen:
                    raise ValueError("Duplicate proposal IDs in export")
                seen.add(record["proposal_id"])
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Exported {} proposals to {}", count, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "import", "export"):
        sub = subcommands.add_parser(name)
        sub.add_argument("--project", default="spelling-baseline")
        if name == "import":
            sub.add_argument("--source-id", required=True)
            sub.add_argument("path", type=Path)
        if name == "export":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Serialize repository-local setup/import/export; released automatically on crash.
    lock_path = Path("data/label-studio-review.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = LabelStudio()
        project = client.project(args.project, create=args.command == "setup")
        if args.command == "import":
            import_proposals(client, project["id"], args.path, args.source_id)
        elif args.command == "export":
            export_reviews(client, project["id"], args.output)
        else:
            logger.info("Project ready: {}/projects/{}", client.url, project["id"])


if __name__ == "__main__":
    main()
