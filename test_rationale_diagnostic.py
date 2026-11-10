"""CPU contracts for the separate explanation diagnostic."""
import unittest
from rationale_diagnostic import facts_for, messages_for, parse, seed_for, specification, state_for, summarize


class RationaleDiagnosticTests(unittest.TestCase):
    def test_explanation_parser_rejects_missing_tags_and_extra_actions(self):
        self.assertTrue(parse("I will move down toward A's goal.</rationale><answer>move down</answer>", "brief_rationale")["format_ok"])
        for text in ["move down", "</rationale><answer>wait</answer>",
                     "Reason.</rationale><answer>wait || move down</answer>",
                     "Reason.</rationale><answer>wait</answer>extra"]:
            self.assertFalse(parse(text, "brief_rationale")["format_ok"])

    def test_action_only_has_no_bare_action_fallback(self):
        self.assertFalse(parse("move right", "action_only")["format_ok"])
        self.assertEqual(parse("move right</answer>", "action_only")["action"], "move right")

    def test_prompts_share_state_without_historical_model_answers(self):
        for fixture in specification()["fixtures"]:
            env = state_for(fixture)
            action, a_prefix = messages_for(fixture, "action_only")
            reason, r_prefix = messages_for(fixture, "brief_rationale")
            self.assertIn(env.render(), action[1]["content"])
            self.assertIn(env.render(), reason[1]["content"])
            self.assertEqual(len(action), 2)
            self.assertEqual(len(reason), 2)
            self.assertNotIn("Do not add an explanation.", reason[0]["content"])
            self.assertEqual((a_prefix, r_prefix), ("<answer>", "<rationale>"))

    def test_local_block_delays_opponent_and_facts_do_not_mutate_state(self):
        fixture = specification()["fixtures"][0]
        facts = facts_for(fixture)
        before = facts["shortest_distances_before_action"]["B"]
        after = facts["legal_action_effects"]["block right"]["after_A_before_B"]["distances"]["B"]
        self.assertGreater(after, before)
        self.assertEqual(facts["observation"], state_for(fixture).render())

    def test_seed_uniqueness_across_distinct_states_and_repetitions(self):
        spec = specification()
        seeds = [seed_for(f["id"], i) for f in spec["fixtures"] for i in range(3)]
        self.assertEqual(len(set(seeds)), 18)

    def test_mirrored_failed_block_does_not_delay_b(self):
        fixture = next(f for f in specification()["fixtures"] if f["id"] == "mirror_pre_block")
        facts = facts_for(fixture)
        self.assertEqual(facts["shortest_distances_before_action"]["B"], 3)
        self.assertEqual(facts["legal_action_effects"]["block right"]["after_A_before_B"]["distances"]["B"], 3)

    def test_completion_requires_all_unique_cells(self):
        spec = specification()
        rows = [{"policy": p, "condition": c, "fixture_id": f["id"], "repetition": i,
                 "format_ok": True, "action_is_legal": True, "action": "wait", "token_budget_exhausted": False}
                for p in spec["policies"] for c in spec["conditions"] for f in spec["fixtures"] for i in range(3)]
        self.assertTrue(summarize(rows, spec)["complete"])
        self.assertFalse(summarize(rows[:-1]+rows[:1], spec)["complete"])
        rows[0]["format_ok"] = False
        self.assertEqual(summarize(rows, spec)["action_changes"][0]["both_valid_pairs"], 17)


if __name__ == "__main__":
    unittest.main()
