import unittest


from domain.barc.barc_rules import (
    NO_FILTER_SENTINEL,
    infer_user_specified_region_target,
    resolve_time_window_value,
    choose_default_with_constraints,
    sql_has_dim_filter,
)


class TestBarcFilterResolution(unittest.TestCase):
    def test_infer_user_specified_region_disambiguates_genre_code(self):
        candidates = [
            {"region": "HSM", "target": "NCCS All 15+"},
            {"region": "India", "target": "NCCS AB 15+"},
        ]
        # If the only region-like match is a genre code and there's no explicit hint,
        # we treat it as NOT specifying region.
        region, target = infer_user_specified_region_target(
            question="hindi news performance HSM",
            candidates=candidates,
            inferred_genre="HSM",
        )
        self.assertFalse(region)
        self.assertFalse(target)

    def test_infer_user_specified_region_honors_locative_phrase_for_non_genre_token(self):
        candidates = [
            {"region": "India", "target": "NCCS All 15+"},
        ]
        region, target = infer_user_specified_region_target(
            question="top channels in India",
            candidates=candidates,
            inferred_genre="HSM",
        )
        self.assertTrue(region)
        self.assertFalse(target)

    def test_infer_user_specified_region_does_not_promote_genre_code_in_locative_phrase(self):
        candidates = [
            {"region": "HSM", "target": "NCCS All 15+"},
        ]
        region, target = infer_user_specified_region_target(
            question="top performing timebands for NDTV in HSM",
            candidates=candidates,
            inferred_genre="HSM",
        )
        self.assertFalse(region)
        self.assertFalse(target)

    def test_infer_user_specified_region_disambiguates_factual_code(self):
        candidates = [
            {"region": "FACTUAL", "target": "NCCS All 15+"},
        ]
        region, target = infer_user_specified_region_target(
            question="factual performance FACTUAL",
            candidates=candidates,
            inferred_genre="FACTUAL",
        )
        self.assertFalse(region)
        self.assertFalse(target)

    def test_resolve_time_window_defaults_when_user_did_not_specify(self):
        value, source = resolve_time_window_value(
            planner_value=NO_FILTER_SENTINEL,
            question="show top channels by AMA",
        )
        self.assertEqual(value, "Last 4 Weeks")
        self.assertEqual(source, "default")

    def test_resolve_time_window_flags_missing_constraint_when_user_specified(self):
        value, source = resolve_time_window_value(
            planner_value=NO_FILTER_SENTINEL,
            question="last 2 weeks show top channels by AMA",
        )
        self.assertEqual(value, NO_FILTER_SENTINEL)
        self.assertEqual(source, "explicit")

    def test_choose_default_with_constraints_can_infer_unique_value(self):
        allowed_rows = [
            {"genre": "HSM", "region": "HSM", "target": "NCCS All 15+", "channel": "Aaj Tak"},
            {"genre": "English News", "region": "India", "target": "NCCS AB 15+", "channel": "CNN News18"},
        ]
        chosen, src = choose_default_with_constraints(
            dim="region",
            selected_default_dimensions={"region": "India"},
            allowed_rows=allowed_rows,
            constraints={"genre": "HSM", "channel": None, "target": None},
        )
        self.assertEqual(chosen, "HSM")
        self.assertEqual(src, "inferred")

    def test_sql_has_dim_filter_matches_simple_equals(self):
        sql = "SELECT * FROM t WHERE Region = 'HSM' AND Target = 'X'"
        self.assertTrue(sql_has_dim_filter(sql, col="region", value="HSM"))


if __name__ == "__main__":
    unittest.main()

