"""Offline tests for sf311.vision_llm. No network, no API key: the Anthropic client is mocked.

    set PYTHONPATH=src
    python -m unittest tests.test_vision_llm -v      (or: pytest tests/test_vision_llm.py)
"""

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sf311 import vision_llm as v
from sf311.dataset import LABELS, OTHER


def fake_response(label="Encampment", conf=0.9, stop="end_turn", text=None, cache_read=900):
    body = text if text is not None else json.dumps({"label": label, "confidence": conf, "reason": "tents on sidewalk"})
    return SimpleNamespace(
        stop_reason=stop,
        model="claude-sonnet-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=body)],
        usage=SimpleNamespace(input_tokens=1100, cache_creation_input_tokens=0,
                              cache_read_input_tokens=cache_read, output_tokens=120),
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def make_jpeg(path: Path, size=(2000, 1500)):
    from PIL import Image

    Image.new("RGB", size, (120, 120, 120)).save(path, format="JPEG")


class TestPrompt(unittest.TestCase):
    def test_every_label_described(self):
        for lab in LABELS + [OTHER]:
            self.assertIn(f"- {lab}:", v.SYSTEM_PROMPT)
        self.assertEqual(v.OUTPUT_SCHEMA["properties"]["label"]["enum"], LABELS + [OTHER])

    def test_prompt_hash_stable(self):
        self.assertEqual(v.prompt_hash(), v.prompt_hash())
        self.assertEqual(len(v.prompt_hash()), 12)

    def test_build_request_caches_system_and_uses_schema(self):
        req = v.build_request("claude-sonnet-5-5", "AAAA", "image/jpeg", "low", fallback=True)
        self.assertEqual(req["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(req["output_config"]["format"]["schema"], v.OUTPUT_SCHEMA)
        self.assertEqual(req["output_config"]["effort"], "low")
        self.assertNotIn("tool_choice", req)  # forced tool_choice 400s on Sonnet 5.5
        self.assertEqual(req["fallbacks"], "default")
        img = req["messages"][0]["content"][0]
        self.assertEqual(img["source"]["type"], "base64")

    def test_no_fallback_for_haiku(self):
        req = v.build_request("claude-haiku-5-5", "AAAA", "image/jpeg", "low", fallback=True)
        self.assertNotIn("fallbacks", req)
        self.assertNotIn("betas", req)


class TestParsingAndCost(unittest.TestCase):
    def test_parse_ok(self):
        r = v.parse_response(fake_response())
        self.assertEqual(r["pred"], "Encampment")
        self.assertAlmostEqual(r["confidence"], 0.9)
        self.assertIsNone(r["error"])
        self.assertEqual(r["cache_read_input_tokens"], 900)

    def test_parse_bad_json(self):
        r = v.parse_response(fake_response(text="not json"))
        self.assertIsNone(r["pred"])
        self.assertTrue(r["error"].startswith("bad json"))

    def test_parse_refusal(self):
        r = v.parse_response(fake_response(stop="refusal", text=""))
        self.assertEqual(r["pred"], OTHER)
        self.assertEqual(r["error"], "refusal")

    def test_confidence_clipped_and_unknown_label(self):
        r = v.parse_response(fake_response(label="Potholes", conf=3))
        self.assertEqual(r["pred"], OTHER)  # schema enum should prevent this; defensive anyway
        self.assertEqual(r["confidence"], 1.0)

    def test_cost(self):
        u = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 1_000_000, "output_tokens": 1_000_000}
        self.assertAlmostEqual(v.cost_usd("claude-sonnet-5-5", u), 2.00 + 0.20 + 10.00)
        self.assertAlmostEqual(v.cost_usd("claude-haiku-5-5", u), 0.10 + 0.01 + 0.50)
        self.assertTrue(math.isnan(v.cost_usd("unknown-model", u)))

    def test_estimate_haiku_cheaper(self):
        s = v.estimate("claude-sonnet-5-5", 500, 1049)
        h = v.estimate("claude-haiku-5-5", 500, 1049)
        self.assertGreater(s.cost, h.cost * 10)
        self.assertGreater(v.estimate("claude-sonnet-5-5", 1300, 1049).cost, s.cost)


class TestMetrics(unittest.TestCase):
    def test_metrics(self):
        y = ["A", "A", "B", "B"]
        p = ["A", "B", "B", "B"]
        m = v.metrics(y, p, labels=["A", "B", "C"])
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["per_class"]["A"]["recall"], 0.5)
        self.assertAlmostEqual(m["per_class"]["B"]["recall"], 1.0)
        f1a, f1b = 2 * 1 * 0.5 / 1.5, 2 * (2 / 3) * 1 / (5 / 3)
        self.assertAlmostEqual(m["macro_f1"], (f1a + f1b) / 2)  # C absent -> excluded
        self.assertTrue(math.isnan(m["per_class"]["C"]["recall"]))


class TestSampling(unittest.TestCase):
    def df(self):
        labs = ["Street and Sidewalk Cleaning"] * 600 + ["Parking Enforcement"] * 300 + ["Sewer"] * 8 + [OTHER] * 92
        return pd.DataFrame({"service_request_id": [str(i) for i in range(len(labs))], "label": labs})

    def test_size_strata_determinism(self):
        d = self.df()
        s = v.stratified_sample(d, 100)
        self.assertEqual(len(s), 100)
        self.assertEqual(s["service_request_id"].nunique(), 100)
        self.assertGreaterEqual((s["label"] == "Sewer").sum(), 5)  # rare class floor
        self.assertEqual(s["service_request_id"].tolist(), v.stratified_sample(d, 100)["service_request_id"].tolist())

    def test_n_larger_than_df(self):
        self.assertEqual(len(v.stratified_sample(self.df(), 5000)), 1000)


class TestImage(unittest.TestCase):
    def test_encode_downscales_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jpg"
            make_jpeg(p)
            before = p.read_bytes()
            _, media, size = v.encode_image(p, 1024)
            self.assertEqual(media, "image/jpeg")
            self.assertEqual(max(size), 1024)
            self.assertEqual(p.read_bytes(), before)  # original untouched
            _, _, size0 = v.encode_image(p, 0)
            self.assertEqual(size0, (2000, 1500))

    def test_image_tokens(self):
        self.assertEqual(v.image_tokens(1024, 768), math.ceil(1024 * 768 / 750))
        self.assertLessEqual(v.image_tokens(8000, 6000), math.ceil(2576 * 1932 / 750) + 1)


class TestRunLoop(unittest.TestCase):
    def args(self, **kw):
        a = dict(model="claude-sonnet-5-5", concurrency=2, retries=1, effort="low", max_side=512, fallback=True)
        a.update(kw)
        return SimpleNamespace(**a)

    def todo(self, d: Path, n=3):
        rows = []
        for i in range(n):
            p = d / f"{i}.jpg"
            make_jpeg(p, (800, 600))
            rows.append({"service_request_id": str(i), "split": "test", "label": "Encampment",
                         "service_subtype": "encampment", "path": str(p)})
        return pd.DataFrame(rows)

    def test_classify_all_with_mock_client_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            client = FakeClient([fake_response()])
            out = []
            asyncio.run(v.classify_all(client, self.todo(d), self.args(), "h1", out.append))
            self.assertEqual(len(out), 3)
            self.assertEqual(len(client.calls), 3)
            self.assertTrue(all(r["pred"] == "Encampment" for r in out))
            self.assertTrue(all(r["cost_usd"] > 0 for r in out))
            # cache round trip with dedupe on (id, prompt_hash)
            v.save_cache("m", out, pred_dir=d)
            v.save_cache("m", out[:1], pred_dir=d)
            c = v.load_cache("m", pred_dir=d)
            self.assertEqual(len(c), 3)
            self.assertNotIn("data", c.columns)  # no image bytes stored
            print("\n" + v.summarize(c, "m"))

    def test_bad_json_retried(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient([fake_response(text="oops"), fake_response()])
            out = []
            asyncio.run(v.classify_all(client, self.todo(Path(d), 1), self.args(), "h", out.append))
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(out[0]["pred"], "Encampment")

    def test_non_retryable_error_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient([ValueError("boom")])
            out = []
            asyncio.run(v.classify_all(client, self.todo(Path(d), 2), self.args(), "h", out.append))
            self.assertEqual(out, [])


class TestDryRunMakesNoCall(unittest.TestCase):
    def test_dry_run(self):
        import contextlib
        import io

        import anthropic

        orig = anthropic.AsyncAnthropic

        def boom(*a, **k):
            raise AssertionError("dry run must not construct a client")

        anthropic.AsyncAnthropic = boom
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                v.main(["--dry-run"])
            self.assertIn("claude-haiku-5-5", buf.getvalue())
            self.assertIn("DRY RUN", buf.getvalue())
        finally:
            anthropic.AsyncAnthropic = orig


if __name__ == "__main__":
    unittest.main()
