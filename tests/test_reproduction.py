"""Packaging/archive tests; never launch model experiments."""
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.build_tables import build, stats
from reproduction.reproduce import GROUPS, REPO, job_plan, result_errors, validate_result, write_parameter_cache
from reproduction.verify_archive import verify


class ReproductionTests(unittest.TestCase):
    def test_archive_integrity(self):
        report = verify()
        self.assertEqual(report["status"], "archive_integrity_passed")
        self.assertFalse(report["raw_data_rerun_verified"])

    def test_plan_unique_outputs(self):
        jobs = job_plan()
        self.assertEqual(len({j["name"] for j in jobs}), len(jobs))
        self.assertEqual(len({j["expected"] for j in jobs}), len(jobs))
        self.assertEqual(set(j["group"] for j in jobs), set(GROUPS))
        for j in jobs:
            self.assertTrue((REPO / j["argv"][0]).is_file())
            self.assertNotIn("two_stage", " ".join(j["argv"]))
            self.assertNotIn("admm_kriging", j["argv"])

    def test_group_filter(self):
        self.assertEqual([j["name"] for j in job_plan(["guangzhou"], ["scalability"])], ["scalability"])
        self.assertEqual(job_plan(["guangzhou"], ["timeseries"]), [])

    def test_missing_metrics_not_zero(self):
        self.assertIsNone(stats([None, float("nan")])["mean"])
        self.assertEqual(stats([1, 3])["mean"], 2)

    def test_failed_rows_are_errors(self):
        self.assertTrue(result_errors({"results": [{"error": "failure"}]}))
        self.assertTrue(result_errors({"mae_spatial": float("nan")}))
        self.assertFalse(result_errors({"mae_spatial": 1., "mae_fault": float("nan")}))

    def test_incomplete_result_is_not_success(self):
        self.assertTrue(validate_result("faults", {}))
        self.assertTrue(validate_result("timeseries", {}))
        rows = [{"mae_spatial": 1., "mae_all": 1.} for _ in range(15)]
        self.assertFalse(validate_result("faults", {"results": rows}))

    def test_recovered_cache_has_no_invented_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_parameter_cache(root, ["guangzhou"])
            obj = json.loads((root / "results/hpo_cache/hpo_admm_2b.json").read_text())
            self.assertEqual(obj["best_params"]["rank"], 15)
            self.assertEqual(obj["all_results"], [])
            self.assertIn("not recovered", obj["provenance"])

    def test_table_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build(Path(tmp))
            self.assertEqual(report["tables_exported"], 3)
            self.assertEqual(len(list(Path(tmp).glob("*.csv"))), 3)


if __name__ == "__main__":
    unittest.main()
