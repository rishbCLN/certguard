import csv
import json
import sys
from pathlib import Path

import pytest

import certguard.cli as cli


class Result:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.failed = 0

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode}


class Pipeline:
    calls: list[tuple[Path, dict[str, object]]] = []
    init_kwargs: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        self.init_kwargs.append(kwargs)

    def analyze(self, source: Path, **kwargs) -> Result:
        self.calls.append((source, kwargs))
        return Result("single")


class BatchProcessor:
    calls: list[tuple[list[Path], bool, str | None, dict | None]] = []

    def __init__(self, _pipeline) -> None:
        pass

    def analyze(self, sources, *, recursive, expected_credential_title, manifest) -> Result:
        self.calls.append((sources, recursive, expected_credential_title, manifest))
        return Result("batch")


@pytest.fixture(autouse=True)
def replace_processors(monkeypatch):
    Pipeline.calls.clear()
    Pipeline.init_kwargs.clear()
    BatchProcessor.calls.clear()
    monkeypatch.setattr(cli, "CertGuardPipeline", Pipeline)
    monkeypatch.setattr(cli, "BatchProcessor", BatchProcessor)


def test_single_document_keeps_existing_analysis_options(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "certificate.pdf"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "certguard",
            str(source),
            "--offline",
            "--submission-id",
            "submission-1",
            "--expected-recipient",
            "Alice Example",
        ],
    )

    assert cli.main() == 0

    assert json.loads(capsys.readouterr().out) == {"mode": "single"}
    assert Pipeline.calls == [
        (
            source,
            {
                "submission_id": "submission-1",
                "expected_recipient": "Alice Example",
                "expected_credential_title": None,
            },
        )
    ]


def test_multiple_documents_create_batch_output(monkeypatch, tmp_path) -> None:
    first = tmp_path / "one.pdf"
    second = tmp_path / "two.png"
    output = tmp_path / "reports" / "batch.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "certguard",
            str(first),
            str(second),
            "--expected-credential-title",
            "Python Basics",
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0

    assert json.loads(output.read_text(encoding="utf-8")) == {"mode": "batch"}
    assert BatchProcessor.calls == [([first, second], False, "Python Basics", None)]


def test_failed_batch_returns_nonzero(monkeypatch, tmp_path) -> None:
    first = tmp_path / "one.pdf"
    second = tmp_path / "two.png"

    class FailedBatchProcessor(BatchProcessor):
        def analyze(self, *args, **kwargs):
            result = super().analyze(*args, **kwargs)
            result.failed = 1
            return result

    monkeypatch.setattr(cli, "BatchProcessor", FailedBatchProcessor)
    monkeypatch.setattr(sys, "argv", ["certguard", str(first), str(second)])

    assert cli.main() == 1


def test_manifest_is_loaded_and_passed_to_the_batch(monkeypatch, tmp_path) -> None:
    first = tmp_path / "one.pdf"
    second = tmp_path / "two.png"
    manifest = tmp_path / "class.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["filename", "student_id", "expected_recipient", "expected_credential_title"]
        )
        writer.writerow(["one.pdf", "CS21001", "Alice Example", "Python Basics"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(first), str(second), "--manifest", str(manifest)],
    )

    assert cli.main() == 0

    (_, _, _, loaded) = BatchProcessor.calls[0]
    assert loaded["one.pdf"].student_id == "CS21001"


def test_manifest_with_a_single_document_is_rejected(monkeypatch, tmp_path) -> None:
    source = tmp_path / "one.pdf"
    manifest = tmp_path / "class.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        stream.write("filename,student_id,expected_recipient\n")
        stream.write("one.pdf,CS21001,Alice\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(source), "--manifest", str(manifest)],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_batch_rejects_a_shared_recipient(monkeypatch, tmp_path) -> None:
    first = tmp_path / "one.pdf"
    second = tmp_path / "two.pdf"
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(first), str(second), "--expected-recipient", "Alice"],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_brave_search_uses_environment_key(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "certificate.pdf"
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(source), "--search", "brave"],
    )

    assert cli.main() == 0

    assert json.loads(capsys.readouterr().out) == {"mode": "single"}
    assert Pipeline.init_kwargs[0]["search_enabled"] is True
    assert Pipeline.init_kwargs[0]["search_client"].api_key == "secret-key"


def test_brave_search_requires_environment_key(monkeypatch, tmp_path) -> None:
    source = tmp_path / "certificate.pdf"
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(source), "--search", "brave"],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_onnx_forgery_model_is_passed_to_pipeline(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "certificate.pdf"
    model_path = tmp_path / "forgery.onnx"
    model = object()
    monkeypatch.setattr(cli, "OnnxForgeryModel", lambda path: model if path == model_path else None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(source), "--forgery-model", str(model_path)],
    )

    assert cli.main() == 0

    assert json.loads(capsys.readouterr().out) == {"mode": "single"}
    assert Pipeline.init_kwargs[0]["forgery_model"] is model


def test_grammar_root_is_passed_to_pipeline(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "certificate.pdf"
    grammar_root = tmp_path / "grammar"
    monkeypatch.setattr(
        sys,
        "argv",
        ["certguard", str(source), "--grammar-root", str(grammar_root)],
    )

    assert cli.main() == 0

    assert json.loads(capsys.readouterr().out) == {"mode": "single"}
    assert Pipeline.init_kwargs[0]["grammar_root"] == grammar_root


def test_benchmark_reports_latency_without_enforcing_threshold(
    monkeypatch, tmp_path, capsys
) -> None:
    source = tmp_path / "certificate.png"
    source.touch()

    class BenchmarkReport:
        checks = [type("Check", (), {"name": "semantic_structural_dissonance", "duration_ms": 4})()]

    monkeypatch.setattr(Pipeline, "analyze", lambda _self, _source: BenchmarkReport())
    monkeypatch.setattr(sys, "argv", ["certguard", "benchmark", str(source), "--iterations", "2"])

    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["iterations"] == 2
    assert output["ssdd_ms"] == {"min": 4.0, "median": 4.0, "p95": 4.0, "max": 4.0}
    assert output["threshold_enforced"] is False


def test_invalid_grammar_configuration_is_a_cli_error(monkeypatch, tmp_path) -> None:
    source = tmp_path / "certificate.png"

    def invalid_pipeline(**_kwargs):
        raise cli.GrammarProfileError("invalid signed grammar")

    monkeypatch.setattr(cli, "CertGuardPipeline", invalid_pipeline)
    monkeypatch.setattr(sys, "argv", ["certguard", str(source), "--grammar-root", "grammar"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_build_grammar_subcommand_passes_explicit_inputs(monkeypatch, tmp_path, capsys) -> None:
    template = tmp_path / "template.png"
    manifest = tmp_path / "manifest.yaml"
    output = tmp_path / "grammar" / "example.yaml"
    key = tmp_path / "grammar.key"
    calls = []
    monkeypatch.setenv("CERTGUARD_GRAMMAR_HMAC_KEY_FILE", str(key))
    monkeypatch.setattr(
        cli, "build_grammar_profile", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "certguard",
            "build-grammar",
            "--issuer",
            "nptel",
            "--variant",
            "v1",
            "--template",
            str(template),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0
    assert calls == [(("nptel", "v1", template, manifest, output), {"key_file": key})]
    assert capsys.readouterr().out.strip() == str(output)
