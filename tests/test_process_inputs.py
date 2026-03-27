from types import SimpleNamespace

from boltz import main


def test_process_inputs_skips_existing_without_override(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    records_dir = out_dir / "processed" / "records"
    records_dir.mkdir(parents=True)
    (records_dir / "same.json").write_text("{}")

    input_same = tmp_path / "same.yaml"
    input_same.write_text("version: 1\nsequences: []\n")
    input_new = tmp_path / "new.yaml"
    input_new.write_text("version: 1\nsequences: []\n")

    calls = []

    monkeypatch.setattr(main, "tqdm", lambda x, total=None: x)
    monkeypatch.setattr(main, "load_canonicals", lambda _: {})
    monkeypatch.setattr(main.Record, "load", lambda path: SimpleNamespace(id=path.stem))
    monkeypatch.setattr(main.Manifest, "dump", lambda self, path: None)

    def fake_process_input(path, **kwargs):
        calls.append((path.stem, kwargs["override_existing_processed"]))

    monkeypatch.setattr(main, "process_input", fake_process_input)

    main.process_inputs(
        data=[input_same, input_new],
        out_dir=out_dir,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        msa_server_url="",
        msa_pairing_strategy="greedy",
        boltz2=True,
        override=False,
    )

    assert calls == [("new", False)]


def test_process_inputs_reprocesses_existing_with_override(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    records_dir = out_dir / "processed" / "records"
    records_dir.mkdir(parents=True)
    (records_dir / "same.json").write_text("{}")

    input_same = tmp_path / "same.yaml"
    input_same.write_text("version: 1\nsequences: []\n")
    input_new = tmp_path / "new.yaml"
    input_new.write_text("version: 1\nsequences: []\n")

    calls = []

    monkeypatch.setattr(main, "tqdm", lambda x, total=None: x)
    monkeypatch.setattr(main, "load_canonicals", lambda _: {})
    monkeypatch.setattr(main.Record, "load", lambda path: SimpleNamespace(id=path.stem))
    monkeypatch.setattr(main.Manifest, "dump", lambda self, path: None)

    def fake_process_input(path, **kwargs):
        calls.append((path.stem, kwargs["override_existing_processed"]))

    monkeypatch.setattr(main, "process_input", fake_process_input)

    main.process_inputs(
        data=[input_same, input_new],
        out_dir=out_dir,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        msa_server_url="",
        msa_pairing_strategy="greedy",
        boltz2=True,
        override=True,
    )

    assert set(calls) == {("same", True), ("new", True)}
