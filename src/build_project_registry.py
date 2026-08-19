from __future__ import annotations

import random
from pathlib import Path

from src.smoke_test.config import SmokeTestConfig
from src.smoke_test.io import read_jsonl
from src.smoke_test.schemas import ProjectRecord
from src.smoke_test.stage_utils import is_culturally_sensitive, is_time_sensitive


def build_project_registry(config: SmokeTestConfig, *, limit: int | None = None, subjects: list[str] | None = None) -> list[ProjectRecord]:
    rows = read_jsonl(config.dataset.path)
    target_subjects = subjects or config.dataset.subjects
    if len(target_subjects) > 5:
        raise ValueError("smoke_001 cannot use more than 5 subjects")

    selected: list[ProjectRecord] = []
    rng = random.Random(config.run.seed)
    for subject in target_subjects:
        candidates = [
            row
            for row in rows
            if row.get("language") == config.dataset.language
            and row.get("subject") == subject
            and (not config.dataset.exclude_time_sensitive or not is_time_sensitive(row))
            and (not config.dataset.exclude_culturally_sensitive or not is_culturally_sensitive(row))
        ]
        candidates.sort(key=lambda row: str(row.get("sample_id", "")))
        rng.shuffle(candidates)
        take = min(config.dataset.projects_per_subject, len(candidates))
        if take < config.dataset.projects_per_subject:
            raise ValueError(f"subject {subject} has only {take} eligible projects")
        for row in candidates[:take]:
            selected.append(to_project_record(row))

    if limit is not None:
        if limit > 100:
            raise ValueError("smoke_001 limit cannot exceed 100")
        selected = selected[:limit]
    if len(selected) > config.dataset.total_projects:
        selected = selected[: config.dataset.total_projects]
    if len({project.project_id for project in selected}) != len(selected):
        raise ValueError("duplicate project_id in smoke registry")
    return selected


def to_project_record(row: dict) -> ProjectRecord:
    sample_id = str(row["sample_id"])
    metadata = {
        "required_knowledge": row.get("required_knowledge"),
        "time_sensitive": row.get("time_sensitive"),
        "cultural_sensitivity_label": row.get("cultural_sensitivity_label"),
        "source_dataset_split": row.get("source_dataset_split"),
    }
    return ProjectRecord(
        project_id=sample_id,
        sample_id=sample_id,
        language=str(row.get("language", "en")),
        subject=str(row["subject"]),
        subject_category=str(row.get("subject_category", "")),
        question=str(row["question"]),
        options={str(k): str(v) for k, v in dict(row["options"]).items()},
        answer=str(row["answer"]),
        source_dataset_split=str(row.get("source_dataset_split", "")),
        metadata=metadata,
    )


def main() -> None:
    from src.smoke_test.config import load_config
    from src.smoke_test.io import write_jsonl

    config = load_config("configs/smoke_test.yaml")
    rows = build_project_registry(config)
    write_jsonl(Path(config.run.output_dir) / "projects.jsonl", rows)
    print(f"wrote {len(rows)} projects")


if __name__ == "__main__":
    main()
