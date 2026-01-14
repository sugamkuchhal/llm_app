from domain.barc.barc_rules import resolve_time_window, has_dead_hours_filter

def test_time_window_last_4_weeks():
    text = "Analyze ratings for the last 4 weeks"
    assert resolve_time_window(text) == "Last 4 Weeks"

def test_time_window_none():
    assert resolve_time_window("Analyze ratings") is None

def test_dead_hours_true():
    assert has_dead_hours_filter("exclude dead hours from analysis") is True

def test_dead_hours_false():
    assert has_dead_hours_filter("standard analysis") is False
