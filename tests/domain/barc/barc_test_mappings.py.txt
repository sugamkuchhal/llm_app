from domain.barc.barc_mappings import resolve_genre

def test_resolve_genre_ebn():
    assert resolve_genre("English Business News performance") == "EBN"

def test_resolve_genre_hbn():
    assert resolve_genre("hbn weekly reach") == "HBN v1"

def test_resolve_genre_none():
    assert resolve_genre("sports news") is None
