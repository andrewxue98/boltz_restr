from boltz.model.modules.selection import AtomSelector


def test_name_selection_matches_single_atom():
    selector = AtomSelector("(resid 121 and name O)")

    assert selector.eval({"chain": "A", "resid": 121, "name": "O", "index": 1000})
    assert not selector.eval({"chain": "A", "resid": 121, "name": "N", "index": 997})
    assert not selector.eval({"chain": "A", "resid": 120, "name": "O", "index": 996})


def test_name_keyword_can_select_multiple_names():
    selector = AtomSelector("name N O")

    assert selector.eval({"chain": "A", "resid": 1, "name": "N", "index": 0})
    assert selector.eval({"chain": "A", "resid": 1, "name": "O", "index": 3})
    assert not selector.eval({"chain": "A", "resid": 1, "name": "CA", "index": 1})
