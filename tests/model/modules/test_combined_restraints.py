from types import SimpleNamespace

import torch
from scipy import optimize

from boltz.data import const
from boltz.model.modules.combined_restraints import CombinedRestraints


def _make_test_feats():
    residues = ["GLY", "ALA"]
    atom_names = []
    atom_to_token_idx = []
    ref_space_uid = []
    res_type_idx = []

    for resid0, res_name in enumerate(residues):
        names = const.ref_atoms[res_name]
        atom_names.extend(names)
        atom_to_token_idx.extend([resid0] * len(names))
        ref_space_uid.extend([resid0] * len(names))
        res_type_idx.append(const.token_ids[res_name])

    num_atoms = len(atom_names)
    num_tokens = len(residues)

    atom_to_token = torch.nn.functional.one_hot(
        torch.tensor(atom_to_token_idx), num_classes=num_tokens
    ).float().unsqueeze(0)
    res_type = torch.nn.functional.one_hot(
        torch.tensor(res_type_idx), num_classes=const.num_tokens
    ).float().unsqueeze(0)

    feats = {
        "asym_id": torch.tensor([[0, 0]], dtype=torch.long),
        "atom_pad_mask": torch.ones((1, num_atoms), dtype=torch.float),
        "atom_to_token": atom_to_token,
        "mol_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "record": [
            SimpleNamespace(
                chains=[
                    SimpleNamespace(
                        chain_id=0,
                        chain_name="A",
                        mol_type=const.chain_type_ids["PROTEIN"],
                    )
                ]
            )
        ],
        "ref_atom_name_chars": torch.zeros((1, num_atoms, 4, 64), dtype=torch.float),
        "ref_element": torch.zeros((1, num_atoms, const.num_elements), dtype=torch.float),
        "ref_space_uid": torch.tensor([ref_space_uid], dtype=torch.long),
        "res_type": res_type,
        "ref_conformer_restraint": torch.zeros((1, num_atoms), dtype=torch.long),
    }
    return feats


def test_combined_restraints_reset_clears_state():
    restr = CombinedRestraints.get_instance()
    restr.reset()
    restr.set_config(
        {
            "verbose": "debug",
            "distance_restraints_config": [
                {
                    "atom_selection1": "(resid 1 and name O)",
                    "atom_selection2": "(resid 2)",
                    "flat-bottomed1": {"target_distance1": 10},
                }
            ],
        }
    )

    assert len(restr.distance_data) == 1
    assert restr.log_level == "debug"

    restr.reset()

    assert restr.config == {}
    assert restr.distance_data == []
    assert restr.active_sites == []
    assert restr.sites == []
    assert restr.log_level == "quiet"

    restr.set_config({})
    assert restr.distance_data == []


def test_combined_restraints_logs_resolved_distance_selection(capsys):
    restr = CombinedRestraints.get_instance()
    restr.reset()
    restr.set_config(
        {
            "verbose": "debug",
            "distance_restraints_config": [
                {
                    "atom_selection1": "(resid 1 and name O)",
                    "atom_selection2": "(resid 2)",
                    "flat-bottomed1": {"target_distance1": 10},
                }
            ],
        }
    )

    feats = _make_test_feats()
    restr.setup(feats, nbatch=1)

    output = capsys.readouterr().out
    dist = restr.distance_data[0]

    assert len(dist.selected_atoms1) == 1
    assert len(dist.selected_atoms2) == 5
    assert dist.selection_kind() == "group_com"
    assert "distance[1] kind=group_com matched=(1, 5)" in output
    assert "sel1: (resid 1 and name O)" in output
    assert "sel2: (resid 2)" in output

    restr.reset()


def test_combined_restraints_quiet_mode_suppresses_selection_summary(capsys):
    restr = CombinedRestraints.get_instance()
    restr.reset()
    restr.set_config(
        {
            "verbose": False,
            "distance_restraints_config": [
                {
                    "atom_selection1": "(resid 1 and name O)",
                    "atom_selection2": "(resid 2)",
                    "flat-bottomed1": {"target_distance1": 10},
                }
            ],
        }
    )

    restr.setup(_make_test_feats(), nbatch=1)
    output = capsys.readouterr().out

    assert "distance restraint 1:" not in output
    assert "=== start restr ===" not in output

    restr.reset()


def test_combined_restraints_reverts_nonfinite_optimizer_output(monkeypatch):
    restr = CombinedRestraints.get_instance()
    restr.reset()
    restr.set_config(
        {
            "verbose": "info",
            "distance_restraints_config": [
                {
                    "atom_selection1": "(resid 1 and name O)",
                    "atom_selection2": "(resid 2)",
                    "flat-bottomed1": {"target_distance1": 10},
                }
            ],
        }
    )
    restr.setup(_make_test_feats(), nbatch=1)

    coords = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    expected = coords.clone()

    def fake_minimize(*args, **kwargs):
        return optimize.OptimizeResult(
            x=torch.full((18,), float("nan")).numpy(),
            success=False,
            status=3,
            message="forced failure",
        )

    monkeypatch.setattr(optimize, "minimize", fake_minimize)

    restr.minimize(coords, istep=0, sigma_t=0.5)

    assert torch.equal(coords, expected)

    restr.reset()
