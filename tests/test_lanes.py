"""Lanes: the first lane, each loop step, deterministic checks first, and completion that is earned.

Every rule from the orchestration pattern is a test here: one request for all questions with an
'other' escape; low confidence goes up, not down; failing checks and scope never reach Jev;
escalation moves one lane at a time and ends at a person; complete is refused when the facts
say otherwise.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _decide_fakes import Scripted, TempHome  # noqa: E402
from jevkit import lanes, policy as policies  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class Classify(TempHome):
    def test_all_questions_go_in_one_request_with_an_other_option(self):
        fake = Scripted({"lane": "small", "security_sensitive": 0.05, "underspecified": 0.1})
        out = lanes.classify("rename parse_row to parse_line in utils.py", transport=fake)
        self.assertEqual(len(fake.requests), 1)
        questions = fake.requests[0]["questions"]
        self.assertEqual(sorted(questions), ["lane", "security_sensitive", "underspecified"])
        self.assertIn("other", questions["lane"]["criteria"])
        self.assertEqual(out["lane"], "small")
        self.assertEqual(out["target"], {"agent": "jev-lane-small", "model": "haiku", "effort": "low"})

    def test_a_small_pick_without_confidence_goes_up_to_medium(self):
        fake = Scripted({"lane": "small"}, confidence=0.6, rest=0.1)
        self.assertEqual(lanes.classify("tidy the readme", transport=fake)["lane"], "medium")

    def test_a_medium_pick_below_half_confidence_goes_to_high(self):
        fake = Scripted({"lane": "medium"}, confidence=0.4, rest=0.15)
        self.assertEqual(lanes.classify("fix the flaky sync", transport=fake)["lane"], "high")

    def test_security_sensitive_work_is_at_least_high(self):
        fake = Scripted({"lane": "small", "security_sensitive": 0.9})
        self.assertEqual(lanes.classify("rotate the webhook secret", transport=fake)["lane"], "high")

    def test_escalate_needs_confidence(self):
        self.assertEqual(lanes.classify("x", transport=Scripted({"lane": "escalate"}))["lane"], "escalate")
        unsure = Scripted({"lane": "escalate"}, confidence=0.5, rest=0.1)
        self.assertEqual(lanes.classify("x", transport=unsure)["lane"], "high")

    def test_other_keeps_the_current_model(self):
        out = lanes.classify("ask Steve which plan he wants", transport=Scripted({"lane": "other"}))
        self.assertEqual(out["lane"], "keep_current")
        self.assertIsNone(out["target"])

    def test_code_decides_first_and_nothing_is_sent(self):
        fake = Scripted({"lane": "small"})
        self.assertEqual(lanes.classify("x", facts={"person_named_model": True}, transport=fake)["lane"], "keep_current")
        self.assertEqual(lanes.classify("x", facts={"prior_failed_attempts": 2}, transport=fake)["lane"], "escalate")
        self.assertEqual(lanes.classify("x", facts={"security_paths": True}, transport=fake)["lane"], "high")
        self.assertEqual(fake.requests, [])

    def test_jev_down_keeps_the_current_model(self):
        out = lanes.classify("anything", transport=Scripted(fail="timeout"))
        self.assertEqual(out["lane"], "keep_current")
        self.assertTrue(out["decision"]["fallback_used"])

    def test_hermes_targets_and_a_local_override(self):
        self.assertEqual(lanes.targets("hermes")["escalate"]["model"], "gpt-6-astra")
        (self.home / "jev").mkdir(exist_ok=True)
        (self.home / "jev" / "lanes.json").write_text(json.dumps(
            {"hermes": {"small": {"model": "gpt-reserve"}}, "claude-code": {"nonsense": {"model": "x"}}}))
        mapped = lanes.targets("hermes")
        self.assertEqual(mapped["small"], {"provider": "openai-codex", "model": "gpt-reserve", "effort": "medium"})
        self.assertEqual(sorted(lanes.targets("claude-code")), sorted(lanes.LANES))


class Step(TempHome):
    GOOD = {"checks_run": True, "checks_available": True, "checks_failed": 0, "out_of_scope_files": 0,
            "diff_empty": False, "expects_changes": True, "security_changed": False}

    def test_a_failing_check_is_retried_without_asking_jev(self):
        fake = Scripted()
        out = lanes.step("x", lane="small", attempts=1, facts={**self.GOOD, "checks_failed": 1}, transport=fake)
        self.assertEqual((out["action"], out["lane"], out["source"]), ("retry", "small", "code"))
        self.assertEqual(fake.requests, [])

    def test_failing_again_escalates_one_lane(self):
        out = lanes.step("x", lane="small", attempts=2, facts={**self.GOOD, "checks_failed": 1}, transport=Scripted())
        self.assertEqual((out["action"], out["lane"]), ("escalate", "medium"))
        self.assertEqual(out["target"]["model"], "sonnet")
        same = lanes.step("x", lane="medium", attempts=1, facts={**self.GOOD, "checks_failed": 2},
                          same_failure_repeated=True, transport=Scripted())
        self.assertEqual(same["lane"], "high")

    def test_the_top_lane_escalates_to_a_person(self):
        out = lanes.step("x", lane="escalate", attempts=3, facts={**self.GOOD, "checks_failed": 1}, transport=Scripted())
        self.assertEqual((out["action"], out["lane"], out["target"]), ("escalate", "person", None))

    def test_scope_and_unrun_checks_are_code_decisions(self):
        fake = Scripted()
        self.assertEqual(lanes.step("x", lane="medium", facts={**self.GOOD, "out_of_scope_files": 2},
                                    transport=fake)["action"], "retry")
        self.assertEqual(lanes.step("x", lane="medium", facts={**self.GOOD, "checks_run": False},
                                    transport=fake)["action"], "verify")
        self.assertEqual(lanes.step("x", lane="medium", facts={**self.GOOD, "diff_empty": True},
                                    transport=fake)["action"], "continue")
        self.assertEqual(lanes.step("x", lane="small", facts={**self.GOOD, "security_changed": True},
                                    transport=fake)["lane"], "medium")
        self.assertEqual(fake.requests, [])

    def test_complete_when_checks_pass_and_jev_agrees(self):
        fake = Scripted({"implemented": 0.95, "in_scope": 0.92, "next": "complete"})
        out = lanes.step("add --json flag", lane="small", facts=self.GOOD,
                         state={"diff_stat": "cli.py | 4 +", "checks": "$ pytest\nexit 0\n3 passed"}, transport=fake)
        self.assertEqual(out["action"], "complete")
        sent = fake.requests[0]["state"]
        self.assertEqual(sorted(sent), ["checks", "diff_stat", "task"])

    def test_unsure_jev_escalates_and_undecided_verifies(self):
        low = Scripted({"implemented": 0.6, "in_scope": 0.9, "next": "needs_stronger_model"})
        self.assertEqual(lanes.step("x", lane="medium", facts=self.GOOD, transport=low)["lane"], "high")
        middle = Scripted({"implemented": 0.6, "in_scope": 0.9, "next": "complete"})
        self.assertEqual(lanes.step("x", lane="medium", facts=self.GOOD, transport=middle)["action"], "verify")

    def test_jev_down_means_verify(self):
        self.assertEqual(lanes.step("x", lane="high", facts=self.GOOD, transport=Scripted(fail="timeout"))["action"],
                         "verify")

    def test_complete_is_refused_when_the_facts_disagree(self):
        # A local override with no pre-rules: the guard in step() must still hold.
        rules = json.loads((REPO / "jevkit" / "policies" / "loop-step.json").read_text())
        rules["pre_rules"] = []
        (self.home / "jev" / "policies").mkdir(parents=True)
        (self.home / "jev" / "policies" / "loop-step.json").write_text(json.dumps(rules))
        fake = Scripted({"implemented": 0.99, "in_scope": 0.99, "next": "complete"})
        out = lanes.step("x", lane="small", facts={**self.GOOD, "checks_failed": 1}, transport=fake)
        self.assertEqual(out["action"], "verify")
        self.assertEqual(out["complete_refused"], "a check is failing")


class Evidence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
        for command in (["git", "init", "-q"], ["git", "commit", "-q", "--allow-empty", "-m", "base"]):
            subprocess.run(command, cwd=self.repo, check=True, env=env)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "app.py").write_text("print('hi')\n")
        (self.repo / "notes.txt").write_text("stray\n")
        (self.repo / ".env.local").write_text("X=1\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_scope_sensitive_paths_and_check_tails(self):
        long_output = "python3 -c \"import sys; [print(i) for i in range(500)]; sys.exit(3)\""
        found = lanes.evidence(self.repo, runs=["true", long_output], scope=["src/*"])
        facts = found["facts"]
        self.assertEqual(facts["checks_failed"], 1)
        self.assertEqual(facts["out_of_scope_files"], 2)
        self.assertTrue(facts["security_changed"])
        self.assertEqual(found["sensitive"], [".env.local"])
        failing = found["checks"][1]
        self.assertEqual(failing["exit"], 3)
        self.assertEqual(failing["tail"].splitlines()[-1], "499")
        self.assertLessEqual(len(failing["tail"].splitlines()), lanes.TAIL_LINES)
        self.assertIn("src/app.py", found["state"]["diff_stat"])

    def test_no_changes_and_no_checks(self):
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=empty, check=True)
        facts = lanes.evidence(empty)["facts"]
        self.assertTrue(facts["diff_empty"])
        self.assertFalse(facts["checks_run"])
        self.assertFalse(facts["checks_available"])


class ClaudeAgents(unittest.TestCase):
    def test_each_lane_agent_matches_the_lane_map(self):
        folder = REPO / "claude" / "agents"
        for lane, target in lanes.TARGETS["claude-code"].items():
            text = (folder / f"{target['agent']}.md").read_text()
            head = text.split("---")[1]
            self.assertIn(f"name: {target['agent']}", head)
            self.assertRegex(head, rf"(?m)^model: {target['model']}$")
            self.assertRegex(head, rf"(?m)^effort: {target['effort']}$")
            self.assertIn("hermes-jev-skills", head)

    def test_the_policies_lint(self):
        for name in ("lane", "loop-step"):
            loaded = policies.load(name)
            self.assertLessEqual(len(loaded["questions"]), 8)


class ClaudeInstall(unittest.TestCase):
    """The installer's Claude Code target: additive, backed up, delimited, removable."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("jev_install_lanes", REPO / "install.py")
        self.install = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.install)
        self._tmp = tempfile.TemporaryDirectory()
        self.claude = Path(self._tmp.name) / ".claude"
        self.claude.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_block_is_added_once_backed_up_and_removed_cleanly(self):
        mine = "# My rules\n\nAlways use tabs.\n"
        (self.claude / "CLAUDE.md").write_text(mine)
        first = self.install.install_claude(self.claude, check=False)
        self.assertEqual(first["claude_md_change"], "added")
        self.assertTrue(Path(first["claude_md_backup"]).read_text() == mine)
        text = (self.claude / "CLAUDE.md").read_text()
        self.assertTrue(text.startswith(mine.rstrip()))
        self.assertEqual(text.count(self.install.BLOCK_BEGIN), 1)
        again = self.install.install_claude(self.claude, check=False)
        self.assertEqual(again["claude_md_change"], "unchanged")
        self.assertEqual(len(list(self.claude.glob("CLAUDE.md.bak-jev-*"))), 1)
        self.assertEqual(sorted(p.name for p in (self.claude / "agents").iterdir()),
                         [f"jev-lane-{lane}.md" for lane in sorted(lanes.LANES)])
        gone = self.install.uninstall_claude(self.claude)
        self.assertEqual(len(gone["agents_removed"]), 4)
        self.assertEqual((self.claude / "CLAUDE.md").read_text(), mine)

    def test_a_persons_own_agent_file_is_left_alone(self):
        (self.claude / "agents").mkdir()
        own = self.claude / "agents" / "jev-lane-small.md"
        own.write_text("---\nname: jev-lane-small\nmodel: sonnet\n---\nmine\n")
        out = self.install.install_claude(self.claude, check=False, with_block=False)
        self.assertIn(str(own), out["agents_left_alone"])
        self.assertIn("mine", own.read_text())
        self.assertFalse((self.claude / "CLAUDE.md").exists())
        self.install.uninstall_claude(self.claude)
        self.assertTrue(own.exists())

    def test_check_changes_nothing_and_a_fresh_file_is_deleted_on_uninstall(self):
        out = self.install.install_claude(self.claude, check=True)
        self.assertEqual(out["claude_md_change"], "added")
        self.assertFalse((self.claude / "CLAUDE.md").exists())
        self.assertFalse((self.claude / "agents").exists())
        self.install.install_claude(self.claude, check=False)
        self.install.uninstall_claude(self.claude)
        self.assertFalse((self.claude / "CLAUDE.md").exists())


if __name__ == "__main__":
    unittest.main()
