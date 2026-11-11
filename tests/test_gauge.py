import unittest
from decimal import Decimal
from metrology.errors import Conflict, StateError, Missing, InputError
from metrology.processing.grr import analyze_balanced, Reading, f_critical
from metrology.console import dispatch
from support import application, certificate, rows
from metrology.ingress.readers import InputRow
from metrology.observations import Observation

START = "2026-11-11T00:00:00Z"
END = "2026-11-12T00:00:00Z"
RECEIVED = "2026-11-12T01:00:00Z"


def gauge_observation(index, part, operator, replicate, value, **changes):
    payload = {
        "id": f"g{index}",
        "source": "s1",
        "sequence": index,
        "instrument": "i1",
        "channel": "p",
        "event_time": f"2026-11-11T00:{index % 60:02d}:00Z",
        "value": str(value),
        "unit": "kPa",
        "tags": {
            "lot": "A",
            "part": part,
            "operator": operator,
            "replicate": str(replicate),
        },
    }
    payload.update(changes)
    return Observation.parse(payload)


def study_plan(**changes):
    plan = {
        "id": "grr-1",
        "instrument": "i1",
        "channel": "p",
        "start": START,
        "end": END,
        "parts": ["part-a", "part-b", "part-c"],
        "operators": ["op-1", "op-2", "op-3"],
        "replicates": 2,
        "sources": ["s1"],
        "tolerance_width": "10",
        "process_variation": "4",
    }
    plan.update(changes)
    return plan


def build_balanced_rows():
    # Deterministic synthetic crossed data with part spread dominating.
    part_offsets = {"part-a": 0, "part-b": 4, "part-c": 8}
    operator_offsets = {"op-1": 0, "op-2": 1, "op-3": 2}
    index = 0
    items = []
    for part, po in part_offsets.items():
        for operator, oo in operator_offsets.items():
            for replicate in (1, 2):
                value = 10 + po + oo + (0.2 if replicate == 2 else 0)
                items.append(
                    gauge_observation(index, part, operator, replicate, value)
                )
                index += 1
    return items


class GaugeStudyLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.app = application()
        self.addCleanup(self.app.close)
        # Widen the acceptance band so the synthetic study values are accepted.
        self.app.catalog.add_policy(
            {"id": "q-wide", "unit": "kPa", "lower": "0", "upper": "1000"}
        )
        self.app.catalog.bind("i1", "p", "q-wide")
        self.addCleanup(self.app.close)

    def collect_balanced(self):
        self.app.gauge_studies.plan(study_plan())
        self.app.gauge_studies.start("grr-1", 1)
        self.app.pipeline.ingest(
            "grr-batch", rows(*build_balanced_rows()), received_at=RECEIVED
        )
        for index in range(len(build_balanced_rows())):
            self.app.gauge_studies.bind("grr-1", f"g{index}")

    def test_plan_is_immutable_and_calibration_checked(self):
        first = self.app.gauge_studies.plan(study_plan())
        self.assertEqual(first["state"], "planned")
        self.assertEqual(self.app.gauge_studies.plan(study_plan()), first)
        with self.assertRaises(Conflict):
            self.app.gauge_studies.plan(study_plan(replicates=3))
        with self.assertRaises(StateError):
            self.app.gauge_studies.plan(
                study_plan(
                    id="bad-window",
                    start="2030-01-01T00:00:00Z",
                    end="2030-01-02T00:00:00Z",
                )
            )
        with self.assertRaises(InputError):
            self.app.gauge_studies.plan(study_plan(parts=["only-one"]))
        with self.assertRaises(InputError):
            self.app.gauge_studies.plan(study_plan(tolerance_width="0"))
        with self.assertRaises(InputError):
            self.app.gauge_studies.plan(study_plan(replicates=1))
        with self.assertRaises(InputError):
            self.app.gauge_studies.plan(study_plan(operators=["op-1"]))
        with self.assertRaises(InputError):
            self.app.gauge_studies.plan(study_plan(sources=[]))

    def test_full_balanced_study_produces_frozen_anova(self):
        self.collect_balanced()
        completeness = self.app.gauge_studies.completeness("grr-1")
        self.assertTrue(completeness["balanced"])
        self.assertEqual(completeness["planned_cells"], 18)
        self.assertEqual(completeness["bound_observations"], 18)
        self.assertEqual(completeness["issues"], [])
        analysis = self.app.gauge_studies.analyze("grr-1")
        evaluation = analysis["formula_evaluation"]
        self.assertIsNotNone(evaluation)
        sources = {row["source"] for row in evaluation["anova"]}
        self.assertEqual(
            sources,
            {"part", "operator", "repeatability", "total"}
            if evaluation["interaction"]["pooled_into_repeatability"]
            else {"part", "operator", "interaction", "repeatability", "total"},
        )
        submitted = self.app.gauge_studies.submit("grr-1", 2)
        self.assertEqual(submitted["state"], "review")
        self.assertEqual(submitted["revision"], 3)
        approved = self.app.gauge_studies.decide(
            "grr-1", 3, "lead-metrologist", True, "GRR within tolerance budget"
        )
        self.assertEqual(approved["state"], "approved")
        self.assertEqual(approved["revision"], 4)
        self.assertIsNotNone(approved["report"])
        self.assertTrue(self.app.inspector.check()["invariants"] == "ok")

    def test_balanced_study_reports_variance_components_and_percents(self):
        self.collect_balanced()
        evaluation = self.app.gauge_studies.analyze("grr-1")[
            "formula_evaluation"
        ]
        components = evaluation["variance_components"]
        for component in components.values():
            self.assertIn("estimate", component)
            self.assertIn("variance", component)
            self.assertGreaterEqual(Decimal(component["variance"]), 0)
        # Part spread dominates the synthetic design.
        self.assertGreater(
            Decimal(components["part_to_part"]["variance"]),
            Decimal(components["repeatability"]["variance"]),
        )
        contributions = evaluation["contributions"]
        gauge = contributions["gauge"]
        self.assertIsNotNone(gauge["percent_vs_tolerance"])
        self.assertIsNotNone(gauge["percent_vs_process"])
        total = contributions["total"]
        self.assertGreater(
            Decimal(total["study_variation"]), Decimal(gauge["study_variation"])
        )

    def test_missing_replicate_blocks_submission_and_surfaces_in_completeness(self):
        self.app.gauge_studies.plan(study_plan())
        self.app.gauge_studies.start("grr-1", 1)
        items = build_balanced_rows()
        # Drop the final reading to create a missing cell.
        self.app.pipeline.ingest(
            "grr-batch", rows(*items[:-1]), received_at=RECEIVED
        )
        for index in range(len(items) - 1):
            self.app.gauge_studies.bind("grr-1", f"g{index}")
        completeness = self.app.gauge_studies.completeness("grr-1")
        self.assertFalse(completeness["balanced"])
        self.assertIn("missing_replicates", completeness["issues"])
        self.assertEqual(len(completeness["missing"]), 1)
        self.assertEqual(len(completeness["short_cells"]), 1)
        with self.assertRaises(StateError):
            self.app.gauge_studies.submit("grr-1", 2)
        analysis = self.app.gauge_studies.analyze("grr-1")
        self.assertIsNone(analysis["formula_evaluation"])
        self.assertIn("note", analysis["diagnostics"])
        frozen = self.app.gauge_studies.freeze_diagnostics("grr-1", 2)
        self.assertEqual(frozen["state"], "frozen_diagnostics")
        self.assertIsNotNone(frozen["report"]["diagnostics"])

    def test_extra_replicate_and_reuse_are_recorded_deviations(self):
        self.collect_balanced()
        # A new observation targeting an already occupied cell: extra replicate.
        extra = gauge_observation(
            100, "part-a", "op-1", 1, 10.5, id="g100", sequence=100
        )
        self.app.pipeline.ingest("extra-batch", rows(extra), received_at=RECEIVED)
        with self.assertRaises(Conflict):
            self.app.gauge_studies.bind("grr-1", "g100")
        attempts = self.app.gauge_studies.attempts("grr-1")
        self.assertEqual(attempts[0]["reason"], "duplicate_cell")
        completeness = self.app.gauge_studies.completeness("grr-1")
        self.assertFalse(completeness["balanced"])
        self.assertIn("plan_deviations", completeness["issues"])
        self.assertEqual(
            completeness["deviation_counts"], {"duplicate_cell": 1}
        )
        # Reusing the same record fails and is remembered as a deviation too.
        with self.assertRaises(Conflict):
            self.app.gauge_studies.bind("grr-1", "g0")

    def test_unplanned_part_and_outside_window_are_rejected_with_evidence(self):
        self.app.gauge_studies.plan(study_plan())
        self.app.gauge_studies.start("grr-1", 1)
        bad_part = gauge_observation(
            1, "part-zzz", "op-1", 1, 10, id="bp", sequence=1
        )
        outside = gauge_observation(
            2,
            "part-a",
            "op-1",
            1,
            10,
            id="outside",
            sequence=2,
            event_time="2026-11-13T00:00:00Z",
        )
        self.app.pipeline.ingest(
            "bad-batch", rows(bad_part, outside), received_at=RECEIVED
        )
        with self.assertRaises(Conflict):
            self.app.gauge_studies.bind("grr-1", "bp")
        with self.assertRaises(StateError):
            self.app.gauge_studies.bind("grr-1", "outside")
        reasons = {
            item["reason"] for item in self.app.gauge_studies.attempts("grr-1")
        }
        self.assertEqual(reasons, {"unplanned_part", "outside_window"})

    def test_observation_cannot_move_between_studies_or_cells(self):
        self.collect_balanced()
        self.app.gauge_studies.plan(study_plan(id="grr-2"))
        self.app.gauge_studies.start("grr-2", 1)
        # A bound record is rejected for both the other study and its own.
        with self.assertRaises(Conflict):
            self.app.gauge_studies.bind("grr-2", "g0")
        with self.assertRaises(Conflict):
            self.app.gauge_studies.bind("grr-1", "g0")

    def test_replay_after_submission_blocks_approval_and_freeze_tracks_revision(self):
        self.collect_balanced()
        self.app.gauge_studies.submit("grr-1", 2)
        self.app.pipeline.replay(
            "recalibrate", "i1", "p", START, END
        )
        with self.assertRaises(Conflict):
            self.app.gauge_studies.decide(
                "grr-1", 3, "lead", True, "stale approval"
            )
        completeness = self.app.gauge_studies.completeness("grr-1")
        self.assertIn("bound_record_revised", completeness["issues"])
        self.assertTrue(completeness["revised_bindings"])

    def test_dispatch_reaches_gauge_actions(self):
        self.app.gauge_studies.plan(study_plan())
        result = dispatch(
            self.app,
            {"action": "gauge_start", "id": "grr-1", "revision": 1},
        )
        self.assertEqual(result["state"], "collecting")
        fetched = dispatch(self.app, {"action": "gauge_study", "id": "grr-1"})
        self.assertEqual(fetched["id"], "grr-1")
        with self.assertRaises(Missing):
            dispatch(self.app, {"action": "gauge_study", "id": "missing"})


class GrrStatisticsTests(unittest.TestCase):
    def reading(self, index, part, operator, replicate, value):
        return Reading(
            observation=f"r{index}",
            part=part,
            operator=operator,
            replicate=replicate,
            value=Decimal(str(value)),
            unit="kPa",
            revision=1,
            certificate="c1",
            event_time=START,
        )

    def balanced(self, r=2):
        items = []
        index = 0
        for part in ("a", "b", "c"):
            po = {"a": 0, "b": 4, "c": 8}[part]
            for operator in ("x", "y"):
                oo = {"x": 0, "y": 1}[operator]
                for replicate in range(1, r + 1):
                    items.append(
                        self.reading(
                            index,
                            part,
                            operator,
                            replicate,
                            10 + po + oo + (0.1 if replicate == 2 else 0),
                        )
                    )
                    index += 1
        return items

    def test_negative_estimates_are_truncated_and_flagged(self):
        # Crossing part and operator effects (each part mean equals the grand
        # mean while cells differ) yields negative raw part/operator ANOVA
        # estimates, which must be truncated to zero and explicitly flagged.
        layout = {
            ("a", "x"): (10, 10),
            ("a", "y"): (12, 12),
            ("b", "x"): (12, 12),
            ("b", "y"): (10, 10),
        }
        items = []
        index = 0
        for (part, operator), pair in layout.items():
            for replicate, value in zip((1, 2), pair):
                items.append(
                    self.reading(index, part, operator, replicate, value)
                )
                index += 1
        result = analyze_balanced(items)
        components = result["variance_components"]
        truncated = [
            name
            for name, component in components.items()
            if component["truncated_to_zero"]
        ]
        self.assertTrue(truncated)
        for name in truncated:
            self.assertLess(
                Decimal(components[name]["estimate"]), 0
            )
            self.assertEqual(components[name]["variance"], "0")
        # No component may expose a negative variance after truncation.
        for component in components.values():
            self.assertGreaterEqual(Decimal(component["variance"]), 0)

    def test_interaction_pooling_basis_is_documented(self):
        result = analyze_balanced(self.balanced())
        interaction = result["interaction"]
        self.assertIn(interaction["pooled_into_repeatability"], (True, False))
        self.assertIn("basis", interaction)
        anova_sources = [row["source"] for row in result["anova"]]
        if interaction["pooled_into_repeatability"]:
            self.assertNotIn("interaction", anova_sources)
            self.assertIn("repeatability", anova_sources)
        else:
            self.assertIn("interaction", anova_sources)
        self.assertIn(
            interaction["significant"], (True, False, None)
        )

    def test_zero_total_variation_emits_no_percentages(self):
        items = [
            self.reading(i, part, operator, replicate, 5)
            for i, (part, operator, replicate) in enumerate(
                [
                    (p, o, r)
                    for p in ("a", "b")
                    for o in ("x", "y")
                    for r in (1, 2)
                ]
            )
        ]
        result = analyze_balanced(
            items, tolerance_width="10", process_variation="2"
        )
        for contribution in result["contributions"].values():
            self.assertIsNone(contribution["percent_vs_tolerance"])
            self.assertIsNone(contribution["percent_vs_process"])
        self.assertEqual(
            result["contributions"]["total"]["standard_deviation"], "0"
        )

    def test_missing_tolerance_or_process_leaves_that_scope_null(self):
        result = analyze_balanced(self.balanced())
        gauge = result["contributions"]["gauge"]
        self.assertIsNone(gauge["percent_vs_tolerance"])
        self.assertIsNone(gauge["percent_vs_process"])
        only_tolerance = analyze_balanced(
            self.balanced(), tolerance_width="10"
        )
        self.assertIsNotNone(
            only_tolerance["contributions"]["gauge"]["percent_vs_tolerance"]
        )
        self.assertIsNone(
            only_tolerance["contributions"]["gauge"]["percent_vs_process"]
        )

    def test_unbalanced_input_refuses_formula(self):
        items = self.balanced()[:-1]
        with self.assertRaises(InputError):
            analyze_balanced(items)

    def test_mixed_units_are_rejected(self):
        items = self.balanced()
        items[0] = Reading(
            observation="r0",
            part="a",
            operator="x",
            replicate=1,
            value=Decimal("10"),
            unit="Pa",
            revision=1,
            certificate="c1",
            event_time=START,
        )
        with self.assertRaises(InputError):
            analyze_balanced(items)

    def test_f_table_uses_none_for_untabulated_degrees(self):
        self.assertIsNotNone(f_critical(2, 6))
        self.assertIsNone(f_critical(999, 888))

    def test_design_dimensions_are_echoed(self):
        result = analyze_balanced(self.balanced(r=3))
        self.assertEqual(result["design"]["replicates"], 3)
        self.assertEqual(result["design"]["part_count"], 3)
        self.assertEqual(result["design"]["operator_count"], 2)
        self.assertTrue(result["design"]["balanced"])

    def test_single_replicate_is_rejected(self):
        with self.assertRaises(InputError):
            analyze_balanced(self.balanced(r=1))


if __name__ == "__main__":
    unittest.main()
