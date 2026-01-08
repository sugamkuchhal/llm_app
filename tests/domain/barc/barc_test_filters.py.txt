from domain.barc.barc_filters import build_filters

def test_build_filters_defaults_and_extraction():
    metric = {
        "definition": "English Business News weekly performance"
    }

    sql = """
    SELECT *
    FROM table
    WHERE region = 'India'
      AND target = 'NCCS AB Male 22+'
      AND channel = 'CNBC TV18'
    """

    planner_text = "Analyze last 4 weeks performance"
    filters = build_filters(metric, sql, planner_text)

    assert "Genre: EBN" in filters
    assert "Region: India" in filters
    assert "Target: NCCS AB Male 22+" in filters
    assert "Channel: CNBC TV18" in filters
    assert "Time Window: Last 4 Weeks" in filters


def test_time_window_default_applied():
    filters = build_filters(
        {"definition": "English Business News performance"},
        "SELECT * FROM table",
        planner_text=""
    )
    assert "Time Window: Last 4 Weeks" in filters
