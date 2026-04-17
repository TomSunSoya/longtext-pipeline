from src.longtext_pipeline.manifest import ManifestManager


def test_create_manifest_initializes_audit_as_not_started(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")

    manifest = ManifestManager().create_manifest(str(input_file), "a" * 64)

    assert manifest.stages["audit"].status == "not_started"


def test_update_stage_skipped_does_not_override_degraded_manifest_status(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    manager = ManifestManager()
    manifest = manager.create_manifest(str(input_file), "a" * 64)
    manifest.status = "completed_with_issues"

    manager.update_stage(manifest, "audit", "skipped")

    assert manifest.stages["audit"].status == "skipped"
    assert manifest.status == "completed_with_issues"


def test_completed_stage_helpers_treat_successful_with_warnings_as_complete(tmp_path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("body", encoding="utf-8")
    manager = ManifestManager()
    manifest = manager.create_manifest(str(input_file), "a" * 64)
    manager.update_stage(manifest, "audit", "successful_with_warnings")

    assert manager.is_stage_complete(manifest, "audit") is True
    assert "audit" in manager.get_completed_stages(manifest)
