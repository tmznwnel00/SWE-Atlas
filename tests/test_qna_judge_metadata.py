"""The judge grades the criterion; the evaluator supplies its known metadata."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "data/qa").glob("*/tests/evaluate_answer.py"))
PROMPTS = sorted((ROOT / "data/qa").glob("*/tests/system_prompt.txt"))


class QnaJudgeMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FILES:
            raise RuntimeError("No QnA evaluators found")
        spec = importlib.util.spec_from_file_location("qna_metadata_test", FILES[0])
        cls.evaluator = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=None)}):
            spec.loader.exec_module(cls.evaluator)

    def evaluate(self, rating, title='1.6: Reports a "quoted" value and preserves “Unicode”.'):
        calls = []
        text = rating if isinstance(rating, str) else json.dumps({"ratings": [rating]})

        def create(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])

        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        with patch.object(self.evaluator, "MAX_RETRIES", 1), contextlib.redirect_stderr(io.StringIO()) as errors:
            result = self.evaluator.evaluate_single_rubric(
                client, "judge-model", "system", "{problem_statement}\n{model_answer}\n{title}",
                "problem", "answer", {"id": "known-id", "title": title},
            )
        return result, calls, errors.getvalue()

    def test_all_copies_match_and_evaluators_compile(self):
        self.assertEqual(len(FILES), 124)
        self.assertEqual(len(PROMPTS), 124)
        for paths in [FILES, PROMPTS]:
            self.assertEqual(len({hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), 1)
        for path in FILES:
            compile(path.read_bytes(), str(path), "exec")

    def test_output_example_is_valid_json_without_statement_echo(self):
        examples = re.findall(r"```json\s*\n(.*?)\n```", PROMPTS[0].read_text(), re.DOTALL)
        self.assertEqual(len(examples), 2)
        for example in examples:
            json.loads(example)
        rating = json.loads(examples[-1])["ratings"][0]
        self.assertEqual(set(rating), {"status", "score", "justification"})
        self.assertEqual(set(re.findall(r"\b(?:YES|NO)\b", rating["status"])), {"YES", "NO"})
        self.assertEqual(set(re.findall(r"\b[01]\b", rating["score"])), {"0", "1"})
        self.assertIn("do not output the placeholders themselves", PROMPTS[0].read_text())

    def test_caller_attaches_trusted_title_when_judge_omits_it(self):
        result, calls, errors = self.evaluate({"status": "YES", "score": "1", "justification": "Evidence exists."})
        self.assertEqual(result, {
            "rubric_statement": 'Reports a "quoted" value and preserves “Unicode”.',
            "rubric_id": "known-id", "status": "YES", "score": "1", "justification": "Evidence exists.",
        })
        self.assertEqual(len(calls), 1)
        self.assertEqual(errors, "")

    def test_caller_uses_trusted_metadata_even_if_judge_echoes_other_text(self):
        result, _, _ = self.evaluate({"rubric_statement": "Different generated metadata", "status": "NO", "score": 0})
        self.assertEqual(result["rubric_statement"], 'Reports a "quoted" value and preserves “Unicode”.')
        self.assertEqual(result["rubric_id"], "known-id")
        self.assertEqual(result["status"], "NO")
        self.assertEqual(result["score"], 0)

    def test_criterion_input_and_request_parameters_are_preserved(self):
        _, calls, _ = self.evaluate({"status": "YES", "score": 1})
        request = calls[0]
        self.assertEqual(request["model"], "judge-model")
        self.assertEqual(request["max_tokens"], 2048)
        self.assertEqual(request["messages"][0], {"role": "system", "content": "system"})
        title_json = request["messages"][1]["content"].split("\n", 2)[2]
        self.assertEqual(json.loads(title_json), 'Reports a "quoted" value and preserves “Unicode”.')

    def test_canonical_status_score_and_negative_behavior_are_preserved(self):
        for status, score, raw in [("YES", 0, "1"), ("NO", 1, "0"), (None, 1, "1"), (None, 0, "0")]:
            with self.subTest(status=status, score=score):
                result, calls, errors = self.evaluate({"status": status, "score": score, "justification": "evidence"})
                self.assertEqual(len(calls), 1)
                self.assertEqual(errors, "")
                for rubric_type, expected in [("positive", raw), ("negative", "0" if raw == "1" else "1")]:
                    canonical = self.evaluator._canonicalize_judge_result(result, rubric_type)
                    self.assertEqual(canonical["score"], expected)
                    self.assertTrue(self.evaluator._is_scored(canonical))

    def test_unescaped_quotes_are_still_rejected(self):
        malformed = ('{"ratings":[{"rubric_statement":"A "quoted" criterion",'
                     '"status":"YES","score":"1","justification":"evidence"}]}')
        result, calls, errors = self.evaluate(malformed)
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)
        self.assertIn("invalid response", errors)

    def test_missing_or_invalid_score_and_status_are_still_rejected(self):
        for rating in [{"justification": "evidence"}, {"status": "MAYBE", "score": 0.5}]:
            with self.subTest(rating=rating):
                result, _, errors = self.evaluate(rating)
                self.assertIsNone(result)
                self.assertIn("invalid response", errors)


if __name__ == "__main__":
    unittest.main()
