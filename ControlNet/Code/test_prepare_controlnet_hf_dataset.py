from prepare_controlnet_hf_dataset import list_pairs, write_loading_script


def test_list_pairs_matches_target_and_condition_files(tmp_path):
    (tmp_path / "P1_target.jpg").write_bytes(b"t1")
    (tmp_path / "P1_condition.png").write_bytes(b"c1")
    (tmp_path / "P2_target.jpg").write_bytes(b"t2")
    (tmp_path / "P2_condition.png").write_bytes(b"c2")

    pairs = list_pairs(tmp_path)

    assert len(pairs) == 2
    assert pairs[0] == (tmp_path / "P1_target.jpg", tmp_path / "P1_condition.png")
    assert pairs[1] == (tmp_path / "P2_target.jpg", tmp_path / "P2_condition.png")


def test_list_pairs_skips_target_without_matching_condition(tmp_path):
    (tmp_path / "P1_target.jpg").write_bytes(b"t1")
    (tmp_path / "P1_condition.png").write_bytes(b"c1")
    (tmp_path / "P2_target.jpg").write_bytes(b"t2")  # no matching condition

    pairs = list_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0][0].name == "P1_target.jpg"


def test_list_pairs_empty_dir_returns_empty_list(tmp_path):
    assert list_pairs(tmp_path) == []


def test_write_loading_script_creates_file_matching_dir_name(tmp_path):
    corpus_dir = tmp_path / "paired_controlnet_corpus"
    corpus_dir.mkdir()

    script_path = write_loading_script(corpus_dir)

    assert script_path == corpus_dir / "paired_controlnet_corpus.py"
    assert script_path.exists()
    assert "GeneratorBasedBuilder" in script_path.read_text()
