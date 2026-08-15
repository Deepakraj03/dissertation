import csv

from finalize_dtm_estimation_corpus import find_shard_manifests, merge_shard_manifests


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "status", "patches_saved", "roughness"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_find_shard_manifests_matches_real_naming_pattern(tmp_path):
    (tmp_path / "manifest_shard0of3.csv").write_text("product_id,status,patches_saved,roughness\n")
    (tmp_path / "manifest_shard1of3.csv").write_text("product_id,status,patches_saved,roughness\n")
    (tmp_path / "manifest_shard2of3.csv").write_text("product_id,status,patches_saved,roughness\n")
    (tmp_path / "manifest.csv").write_text("stale merged output from a previous run\n")
    (tmp_path / "unrelated.csv").write_text("not a shard manifest\n")

    found = find_shard_manifests(tmp_path)

    assert {p.name for p in found} == {
        "manifest_shard0of3.csv", "manifest_shard1of3.csv", "manifest_shard2of3.csv",
    }


def test_merge_shard_manifests_combines_all_rows_and_finds_productive_products(tmp_path):
    _write_manifest(tmp_path / "manifest_shard0of2.csv", [
        {"product_id": "A", "status": "ok", "patches_saved": 4, "roughness": 0.1},
        {"product_id": "B", "status": "ok", "patches_saved": 0, "roughness": 0.2},
    ])
    _write_manifest(tmp_path / "manifest_shard1of2.csv", [
        {"product_id": "C", "status": "dtm_download_failed", "patches_saved": 0, "roughness": None},
        {"product_id": "D", "status": "ok", "patches_saved": 12, "roughness": 0.3},
    ])

    all_rows, products_with_patches = merge_shard_manifests(
        find_shard_manifests(tmp_path))

    assert {r["product_id"] for r in all_rows} == {"A", "B", "C", "D"}
    assert len(all_rows) == 4
    assert set(products_with_patches) == {"A", "D"}


def test_merge_shard_manifests_empty_dir_returns_empty(tmp_path):
    all_rows, products_with_patches = merge_shard_manifests(
        find_shard_manifests(tmp_path))
    assert all_rows == []
    assert products_with_patches == []
