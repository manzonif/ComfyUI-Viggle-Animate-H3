"""Unit and executor-level checks for the two-stage hires extend nodes.

Run with the ComfyUI venv's Python:
    PYTHONPATH=<comfyui> python tests/test_hires_loop.py
"""
import importlib.util
import types
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK.parents[1]))
sys.path.insert(0, str(PACK / "tests"))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.sample  # noqa: F401  (attribute-patched in the executor test)

pkg = types.ModuleType("viggle_pack")
pkg.__path__ = [str(PACK)]
sys.modules["viggle_pack"] = pkg
nodes_spec = importlib.util.spec_from_file_location("viggle_pack.nodes", PACK / "nodes.py")
viggle = importlib.util.module_from_spec(nodes_spec)
sys.modules["viggle_pack.nodes"] = viggle
nodes_spec.loader.exec_module(viggle)
loop_spec = importlib.util.spec_from_file_location("viggle_pack.loop_nodes", PACK / "loop_nodes.py")
loop = importlib.util.module_from_spec(loop_spec)
sys.modules["viggle_pack.loop_nodes"] = loop
loop_spec.loader.exec_module(loop)

import folder_paths
import nodes as comfy_nodes
import comfy.model_patcher
import comfy.model_sampling
from comfy.nested_tensor import NestedTensor
from comfy_extras import nodes_custom_sampler as core

from test_chaining import conditioning
from test_loop import (RecordingVAE, TestCondSet, TestModelLoader, TestVAE,
                       fake_sample_record, make_model_patcher)

FPS = 24
AUDIO_FPS = 40


def make_plan(total=328, chunk=124, overlap=5, canvas=(64, 64), with_audio=True):
    spans = viggle.plan_spans(total, chunk, overlap)
    cond_set = {"spans": spans, "conds": [conditioning() for _ in spans],
                "total_frames": total, "canvas": canvas}
    if with_audio:
        total_a = round(total / FPS * AUDIO_FPS)
        cond_set["audio_latent"] = (torch.arange(total_a, dtype=torch.float32)
                                    .reshape(1, 1, 1, total_a)
                                    .expand(1, 32, 2, total_a).contiguous())
        cond_set["audio_digest"] = b"hires-digest"
    return cond_set


class FakeDyn:
    def get_display_node_id(self, unique_id):
        return unique_id


def rows_for(a, b):
    return round((b + 1) / FPS * AUDIO_FPS) - round(a / FPS * AUDIO_FPS)


class StartTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.start = loop.ViggleHiresChunkStart()
        self.dyn = FakeDyn()

    def test_first_chunk_outputs(self):
        loop_tag, state, noise, cond, latent, status = \
            self.start.start(self.plan, 42, 0, 0, None, self.dyn, "6")
        self.assertEqual(noise.seed, 42)
        self.assertIs(cond, self.plan["conds"][0])
        video, audio = latent["samples"].unbind()
        a, b, _, latn = self.plan["spans"][0]
        self.assertEqual(tuple(video.shape), (1, 24, latn, 4, 4))
        self.assertEqual(audio.shape[-1], rows_for(a, b))
        self.assertTrue(torch.equal(video, torch.zeros_like(video)))
        self.assertTrue(torch.equal(audio, self.plan["audio_latent"][..., :rows_for(a, b)]))
        self.assertEqual(state["index"], 0)
        self.assertIsNone(state["previous"])

    def test_chunk_seeds_and_conditioning_advance(self):
        state = self.start.start(self.plan, 42, 0, 0, None, self.dyn, "6")[1]
        # Start reads the advancing index from the state Store passes on.
        state1 = {**state, "index": 1}
        _, _, noise2, cond2, _, _ = self.start.start(self.plan, 42, 0, 0, state1, self.dyn, "6")
        self.assertEqual(noise2.seed, 43)
        self.assertIs(cond2, self.plan["conds"][1])
        self.assertEqual(state["index"], 0)  # start does not mutate the incoming state

    def test_rerender_override_applies_only_to_target_chunk(self):
        state1 = self.start.start(self.plan, 42, 2, 999, None, self.dyn, "6")[1]
        state1 = {**state1, "index": 1}
        _, _, noise, _, _, _ = self.start.start(self.plan, 42, 2, 999, state1, self.dyn, "6")
        self.assertEqual(noise.seed, 999)
        state0 = self.start.start(self.plan, 42, 2, 999, None, self.dyn, "6")[1]
        _, _, noise0, _, _, _ = self.start.start(self.plan, 42, 2, 999, state0, self.dyn, "6")
        self.assertEqual(noise0.seed, 42)


class PinTests(unittest.TestCase):
    def setUp(self):
        self.pin = loop.ViggleHiresChunkPin()
        self.dyn = FakeDyn()
        self.plan = make_plan(with_audio=False)
        self.state0 = loop.ViggleHiresChunkStart().start(self.plan, 42, 0, 0, None, self.dyn, "6")[1]
        latn = self.plan["spans"][0][3]
        self.rows0 = rows_for(*self.plan["spans"][0][:2])
        # Per-latent markers make carry placement check non-trivial.
        self.vid = (torch.arange(latn, dtype=torch.float32).reshape(1, 1, -1, 1, 1)
                    .expand(1, 24, latn, 8, 8).contiguous())
        self.aud = torch.full((1, 32, 2, self.rows0), 2.5)

    def latent(self, video, audio=None):
        if audio is None:
            return {"samples": video, "latent_format_version_0": torch.empty(0)}
        return {"samples": NestedTensor((video, audio)), "latent_format_version_0": torch.empty(0)}

    def test_first_chunk_passthrough_without_mask(self):
        out, state = self.pin.pin(self.latent(self.vid, self.aud), self.state0, self.dyn, "12")
        self.assertIs(state, self.state0)
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video, self.vid))
        self.assertTrue(torch.equal(audio, self.aud))
        self.assertNotIn("noise_mask", out)

    def test_first_chunk_plain_video_gets_generated_audio(self):
        out, _ = self.pin.pin(self.latent(self.vid), self.state0, self.dyn, "12")
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video, self.vid))
        self.assertEqual(audio.shape, (1, 32, 2, self.rows0))
        self.assertTrue(torch.equal(audio, torch.zeros_like(audio)))
        # Nothing pinned and no clean audio: the sampler generates everything.
        self.assertNotIn("noise_mask", out)

    def second_state(self, previous_tail):
        a, b, lat0, latn = self.plan["spans"][0]
        return {**self.state0, "index": 1,
                "previous": {"span": [a, b, lat0, latn], "video": previous_tail}}

    def test_carry_written_and_masked(self):
        state = self.second_state(self.vid[:, :, -2:])
        latn1 = self.plan["spans"][1][3]
        rows1 = rows_for(*self.plan["spans"][1][:2])
        vid2 = torch.full((1, 24, latn1, 8, 8), 3.5)
        aud2 = torch.full((1, 32, 2, rows1), 4.5)
        out, _ = self.pin.pin(self.latent(vid2, aud2), state, self.dyn, "12")
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video[:, :, :2], self.vid[:, :, -2:]))
        self.assertTrue(torch.equal(video[:, :, 2:], vid2[:, :, 2:]))
        self.assertTrue(torch.equal(audio, aud2))
        mask_v, mask_a = out["noise_mask"].unbind()
        self.assertTrue(torch.equal(mask_v[:, :, :2], torch.zeros_like(mask_v[:, :, :2])))
        self.assertTrue(torch.equal(mask_v[:, :, 2:], torch.ones_like(mask_v[:, :, 2:])))
        self.assertTrue(torch.equal(mask_a, torch.ones_like(mask_a)))

    def test_clean_audio_overrides_and_masks_zero(self):
        plan = make_plan()  # with driving audio
        state = loop.ViggleHiresChunkStart().start(plan, 42, 0, 0, None, self.dyn, "6")[1]
        a, b, _, _ = plan["spans"][0]
        out, _ = self.pin.pin(self.latent(self.vid, self.aud), state, self.dyn, "12")
        video, audio = out["samples"].unbind()
        self.assertTrue(torch.equal(audio, plan["audio_latent"][..., :rows_for(a, b)]))
        mask_v, mask_a = out["noise_mask"].unbind()
        self.assertTrue(torch.equal(mask_v, torch.ones_like(mask_v)))  # first chunk: no carry
        self.assertTrue(torch.equal(mask_a, torch.zeros_like(mask_a)))

    def test_temporal_mismatch_raises(self):
        bad = self.vid.clone()
        bad = torch.cat([bad, bad[:, :, :1]], dim=2)
        with self.assertRaisesRegex(ValueError, "temporal"):
            self.pin.pin(self.latent(bad, self.aud), self.state0, self.dyn, "12")

    def test_tail_resolution_mismatch_raises(self):
        state = self.second_state(self.vid[:, :, -2:, :4, :4])
        with self.assertRaisesRegex(ValueError, "tail"):
            self.pin.pin(self.latent(self.vid, self.aud), state, self.dyn, "12")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = loop.ViggleHiresChunkStore()
        self.dyn = FakeDyn()
        self.plan = make_plan()
        self.spans = self.plan["spans"]
        self.start = loop.ViggleHiresChunkStart()

    def run_chunk(self, state, value):
        a, b, lat0, latn = self.spans[state["index"]]
        rows = rows_for(a, b)
        video = torch.full((1, 24, latn, 8, 8), value)
        audio = torch.full((1, 32, 2, rows), value)
        latent = {"samples": NestedTensor((video, audio))}
        return self.store.store(latent, state, self.dyn, "16")

    @staticmethod
    def marker(i, latn):
        """Distinct value per latent position so carry placement is checkable."""
        return i + 1 + 0.001 * torch.arange(latn, dtype=torch.float32).reshape(1, 1, latn, 1, 1)

    def test_master_assembly_and_carry_chain(self):
        state = self.start.start(self.plan, 42, 0, 0, None, self.dyn, "6")[1]
        master_lat = None
        for i in range(len(self.spans)):
            a, b, _, latn = self.spans[i]
            rows = rows_for(a, b)
            new = self.marker(i, latn).expand(1, 24, latn, 8, 8).contiguous()
            audio = torch.full((1, 32, 2, rows), float(i + 1))
            # Pin first (as in the graph); the fake sampler then fills mask-1
            # regions with `new` and keeps mask-0 regions at their input value.
            pin_out, state = loop.ViggleHiresChunkPin().pin(
                {"samples": NestedTensor((new, audio)), "latent_format_version_0": torch.empty(0)},
                state, self.dyn, "12")
            zin, zina = pin_out["samples"].unbind()
            if "noise_mask" in pin_out:
                mv, ma = pin_out["noise_mask"].unbind()
                pv = new * mv + zin * (1 - mv)
                pa = audio * ma + zina * (1 - ma)
            else:
                pv, pa = new, audio
            state, master_lat, _ = self.store.store(
                {"samples": NestedTensor((pv, pa))}, state, self.dyn, "16")
        self.assertEqual(state["index"], len(self.spans))
        total_lat = self.spans[-1][2] + self.spans[-1][3]
        self.assertEqual(tuple(state["master_v"].shape), (1, 24, total_lat, 8, 8))
        # Expected master: each window writes [lat0, lat0+latn); the carry region
        # of window i+1 equals the tail of window i (fake sampler keeps mask-0).
        expected = torch.empty(total_lat)
        for i in range(len(self.spans)):
            lat0, latn = self.spans[i][2], self.spans[i][3]
            expected[lat0:lat0 + latn] = self.marker(i, latn).reshape(-1)
        for i in range(1, len(self.spans)):
            prev_lat0, prev_latn = self.spans[i - 1][2], self.spans[i - 1][3]
            lat0 = self.spans[i][2]
            carry = prev_lat0 + prev_latn - lat0
            expected[lat0:lat0 + carry] = self.marker(i - 1, prev_latn).reshape(-1)[-carry:]
        self.assertTrue(torch.allclose(
            state["master_v"][0, 0, :, 0, 0], expected))
        a = state["master_a"]
        self.assertEqual(a.shape[-1], round(self.plan["total_frames"] / FPS * AUDIO_FPS))
        # Audio is the clean plan slice (Pin overrides, mask-0 keeps it).
        self.assertTrue(torch.equal(a, self.plan["audio_latent"]))
        self.assertTrue(torch.isfinite(master_lat["samples"].unbind()[0]).all())

    def test_canvas_mismatch_raises(self):
        state = self.start.start(self.plan, 42, 0, 0, None, self.dyn, "6")[1]
        state, _, _ = self.run_chunk(state, 1.0)
        a, b, _, latn = self.spans[1]
        bad = torch.full((1, 24, latn, 10, 10), 2.0)
        latent = {"samples": NestedTensor((bad, torch.zeros(1, 32, 2, rows_for(a, b))))}
        with self.assertRaisesRegex(ValueError, "canvas"):
            self.store.store(latent, state, self.dyn, "16")

    def test_nan_raises(self):
        state = self.start.start(self.plan, 42, 0, 0, None, self.dyn, "6")[1]
        a, b, _, latn = self.spans[0]
        video = torch.full((1, 24, latn, 8, 8), float("nan"))
        latent = {"samples": NestedTensor((video, torch.zeros(1, 32, 2, rows_for(a, b))))}
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.store.store(latent, state, self.dyn, "16")

    def test_plain_video_output_allowed(self):
        state = self.start.start(self.plan, 42, 0, 0, None, self.dyn, "6")[1]
        a, b, _, latn = self.spans[0]
        video = torch.full((1, 24, latn, 8, 8), 1.0)
        state, master_lat, _ = self.store.store({"samples": video}, state, self.dyn, "16")
        video_out, audio_out = master_lat["samples"].unbind()
        self.assertTrue(torch.equal(video_out[:, :, 0:latn], video))
        self.assertTrue(torch.equal(audio_out, torch.zeros_like(audio_out)))


class LoopEndMasterTests(unittest.TestCase):
    def test_terminal_returns_master(self):
        plan = make_plan()
        spans = plan["spans"]
        total_lat = spans[-1][2] + spans[-1][3]
        total_a = round(plan["total_frames"] / FPS * AUDIO_FPS)
        master_v = torch.full((1, 24, total_lat, 8, 8), 7.0)
        master_a = torch.full((1, 32, 2, total_a), 8.0)
        state = {"plan": plan, "run_name": "hires_test", "index": len(spans),
                 "entries": [{"span": list(s)} for s in spans],
                 "master_v": master_v, "master_a": master_a, "previous": None}
        out = loop.ViggleChunkLoopEnd().finish(("6", 0), state, torch.ones(1, 8, 8, 3),
                                               FakeDyn(), "17")
        collection, status, master = out
        self.assertEqual(collection["run_name"], "hires_test")
        self.assertIsNotNone(master)
        v, a = master["samples"].unbind()
        self.assertTrue(torch.equal(v, master_v))
        self.assertTrue(torch.equal(a, master_a))

    def test_legacy_state_returns_none_master(self):
        plan = make_plan()
        spans = plan["spans"]
        state = {"plan": plan, "run_name": "legacy", "index": len(spans),
                 "entries": [{"span": list(s)} for s in spans], "previous": None}
        out = loop.ViggleChunkLoopEnd().finish(("6", 0), state, torch.ones(1, 8, 8, 3),
                                               FakeDyn(), "17")
        self.assertIsNone(out[2])


class FakeSeparate:
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    def go(self, latent):
        v, a = latent["samples"].unbind()
        return ({"samples": v, "latent_format_version_0": torch.empty(0)},
                {"samples": a, "latent_format_version_0": torch.empty(0)})


class FakeUpscale:
    """Stand-in for MinimaxH3LatentUpscaler3D: 2x spatial, preserves T and domain."""
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    def go(self, latent):
        v = latent["samples"]
        # 5D interpolate scales the last three dims: keep T, double H and W.
        v = F.interpolate(v, size=(v.shape[2], v.shape[3] * 2, v.shape[4] * 2), mode="nearest")
        return ({"samples": v, "latent_format_version_0": torch.empty(0)},)


class FakeConcat:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_latent": ("LATENT",), "audio_latent": ("LATENT",)}}

    def go(self, video_latent, audio_latent):
        v, a = video_latent["samples"], audio_latent["samples"]
        return ({"samples": NestedTensor((v, a)), "latent_format_version_0": torch.empty(0)},)


class TestSigmasLow:
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def go(self):
        return (torch.tensor([1.0, 0.85714285]),)


class TestSigmasHigh:
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def go(self):
        return (torch.tensor([0.85714285, 0.6, 0.3, 0.0]),)


class MasterCapture:
    captured = None
    RETURN_TYPES = ("STRING",)
    FUNCTION = "go"
    CATEGORY = "_viggle_test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"master": ("LATENT",)}}

    def go(self, master):
        MasterCapture.captured = master
        return ("captured",)


class HiresLoopExecutorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)
        self.record = []
        self.model = make_model_patcher()
        self.plan = make_plan()
        self.n_chunks = len(self.plan["spans"])
        TestModelLoader.model = self.model
        TestCondSet.cond_set = self.plan
        TestVAE.vae = RecordingVAE(self.record)
        MasterCapture.captured = None
        self._mappings = patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {
            "TestModelLoader": TestModelLoader, "TestCondSet": TestCondSet,
            "TestVAE": TestVAE, "BasicGuider": core.BasicGuider,
            "KSamplerSelect": core.KSamplerSelect,
            "SamplerCustomAdvanced": core.SamplerCustomAdvanced,
            "FakeSeparate": FakeSeparate, "FakeUpscale": FakeUpscale, "FakeConcat": FakeConcat,
            "TestSigmasLow": TestSigmasLow, "TestSigmasHigh": TestSigmasHigh,
            "MasterCapture": MasterCapture,
            **loop.NODE_CLASS_MAPPINGS})
        self._mappings.start()
        self._out = patch.object(folder_paths, "get_output_directory", lambda: str(self.out_dir))
        self._out.start()

    def tearDown(self):
        self._out.stop()
        self._mappings.stop()
        self._tmp.cleanup()

    def prompt(self):
        return {
            "1": {"class_type": "TestModelLoader", "inputs": {}},
            "2": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "3": {"class_type": "TestSigmasLow", "inputs": {}},
            "4": {"class_type": "TestSigmasHigh", "inputs": {}},
            "5": {"class_type": "TestCondSet", "inputs": {}},
            "6": {"class_type": "ViggleHiresChunkStart",
                  "inputs": {"cond_set": ["5", 0], "seed": 42,
                             "rerender_chunk": 0, "rerender_seed": 0}},
            "7": {"class_type": "SamplerCustomAdvanced",
                  "inputs": {"noise": ["6", 2], "guider": ["8", 0], "sampler": ["2", 0],
                             "sigmas": ["3", 0], "latent_image": ["6", 4]}},
            "8": {"class_type": "BasicGuider",
                  "inputs": {"model": ["1", 0], "conditioning": ["6", 3]}},
            "9": {"class_type": "FakeSeparate", "inputs": {"latent": ["7", 0]}},
            "10": {"class_type": "FakeUpscale", "inputs": {"latent": ["9", 0]}},
            "11": {"class_type": "FakeConcat",
                   "inputs": {"video_latent": ["10", 0], "audio_latent": ["9", 1]}},
            "12": {"class_type": "ViggleHiresChunkPin",
                   "inputs": {"latent": ["11", 0], "state": ["6", 1]}},
            "13": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["6", 2], "guider": ["8", 0], "sampler": ["2", 0],
                              "sigmas": ["4", 0], "latent_image": ["12", 0]}},
            "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["15", 0]}},
            "15": {"class_type": "TestVAE", "inputs": {}},
            "16": {"class_type": "ViggleHiresChunkStore",
                   "inputs": {"latent": ["13", 0], "state": ["12", 1]}},
            "17": {"class_type": "ViggleChunkLoopEnd",
                   "inputs": {"loop": ["6", 0], "chunk": ["16", 0], "images": ["14", 0]}},
            "18": {"class_type": "MasterCapture", "inputs": {"master": ["17", 2]}},
        }

    def execute(self):
        import execution
        from server import PromptServer
        server = type("S", (), {"client_id": None, "send_sync": lambda *a, **k: None})()
        executor = execution.PromptExecutor(
            server, cache_args={"ram": 16.0, "ram_inactive": 16.0, "lru": 0})
        with patch.object(PromptServer, "instance", types.SimpleNamespace(
                             client_id="test-client", send_sync=lambda *a, **k: None),
                          create=True), \
             patch.object(comfy.model_management, "load_models_gpu", lambda *a, **k: None), \
             patch.object(core.Guider_Basic, "sample", fake_sample_record(self.record)), \
             patch.object(comfy.sample, "fix_empty_latent_channels", lambda *a, **k: a[1]), \
             patch.object(viggle.latent_preview, "prepare_callback", lambda *a, **k: None):
            executor.execute(self.prompt(), "test-prompt", execute_outputs=["18"])
        return executor

    def test_full_hires_loop(self):
        executor = self.execute()
        self.assertTrue(executor.success, executor.status_messages)
        samples = [e for e in self.record if e[0] == "sample"]
        self.assertEqual(len(samples), self.n_chunks * 2)
        # Both stages of a chunk share its noise object, as in the single-shot.
        self.assertEqual([e[1] for e in samples],
                         [seed for seed in (42 + i for i in range(self.n_chunks)) for _ in range(2)])
        master = MasterCapture.captured
        self.assertIsNotNone(master)
        video, audio = master["samples"].unbind()
        total_lat = self.plan["spans"][-1][2] + self.plan["spans"][-1][3]
        total_a = round(self.plan["total_frames"] / FPS * AUDIO_FPS)
        self.assertEqual(tuple(video.shape), (1, 24, total_lat, 8, 8))
        self.assertEqual(tuple(audio.shape), (1, 32, 2, total_a))
        # Stage 2 ran on the high-res canvas for every chunk.
        decodes = [e for e in self.record if e[0] == "decode"]
        self.assertEqual(len(decodes), self.n_chunks)
        self.assertEqual(decodes[0][1], (1, 24, 37, 8, 8))
        # Clean driving audio survived the whole loop (mask-0 rows).
        self.assertTrue(torch.equal(audio, self.plan["audio_latent"]))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
