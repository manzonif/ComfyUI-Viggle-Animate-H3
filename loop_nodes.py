"""Experimental graph-expanded chunk sampling with disk checkpoints."""
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import tempfile

import safetensors
import safetensors.torch
import torch

import folder_paths
import nodes as comfy_nodes
import comfy.model_management
from comfy_execution.graph_utils import GraphBuilder, is_link

from comfy.nested_tensor import NestedTensor
from comfy_extras import nodes_custom_sampler as core_sampler

from .nodes import FPS, ViggleChunkedSampler, _hash_cache_value, _validate_sigmas, _send_progress, _encode_anchor

CHECKPOINT_VERSION = 1


def _run_dir(run_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("Viggle: run_name must be 1–64 letters, digits, underscores or hyphens.")
    if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])", run_name):
        raise ValueError("Viggle: choose a run_name that is not a reserved Windows filename.")
    root = Path(folder_paths.get_output_directory()).resolve()
    directory = (root / "viggle_chunks" / run_name).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Viggle: checkpoint directory must remain inside ComfyUI's output directory.")
    return directory


def _checkpoint_path(directory, filename):
    if not re.fullmatch(r"chunk_[0-9]+_[0-9a-f]{16}\.latent", filename):
        raise ValueError("Viggle: invalid checkpoint filename.")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Viggle: checkpoint path escapes its run directory.")
    return path


def _atomic_write(path, writer):
    fd, temporary = tempfile.mkstemp(prefix=".viggle-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sampling_graph_signature(dynprompt, unique_id):
    """Stable across node IDs/restarts; include loader file identity and node code."""
    memo = {}
    linked_files = {}
    loader_inputs = {"unet_name": "diffusion_models", "lora_name": "loras",
                     "ckpt_name": "checkpoints", "vae_name": "vae"}

    def describe(node_id):
        memo_key = node_id
        if memo_key in memo:
            return memo[memo_key]
        node = dynprompt.get_node(node_id)
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node["class_type"])
        inputs = {name: [describe(value[0]), value[1]] if is_link(value) else value
                  for name, value in sorted(node.get("inputs", {}).items())}
        files = []
        for name, category in loader_inputs.items():
            value = node.get("inputs", {}).get(name)
            if isinstance(value, str):
                path = folder_paths.get_full_path(category, value)
                if path is not None:
                    stat = os.stat(path)
                    files.append((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns))
            elif is_link(value):
                # The graph does not expose the evaluated filename. Track the
                # category's file metadata conservatively, without loading weights
                # or executing arbitrary filename-producing nodes ourselves.
                if category not in linked_files:
                    identities = []
                    for filename in sorted(folder_paths.get_filename_list(category)):
                        path = folder_paths.get_full_path(category, filename)
                        if path is not None:
                            stat = os.stat(path)
                            identities.append((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns))
                    linked_files[category] = identities
                files.append((category, linked_files[category]))
        source = inspect.getsourcefile(cls) if cls is not None else None
        code = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None
        data = (node["class_type"], inputs, files, code)
        h = hashlib.sha256()
        if not _hash_cache_value(h, data):
            raise ValueError("Viggle: sampling graph contains runtime objects that cannot be recorded for resume.")
        memo[memo_key] = h.hexdigest()
        return memo[memo_key]

    node = dynprompt.get_node(unique_id)
    roots = [node["inputs"][key] for key in ("guider", "sampler", "sigmas")]
    if "vae" in node["inputs"]:
        roots.append(node["inputs"]["vae"])
    data = [(describe(value[0]), value[1]) for value in roots]
    data += [hashlib.sha256(Path(inspect.getsourcefile(core_sampler.Noise_RandomNoise)).read_bytes()).hexdigest()]
    # Changes to our carry/serialization implementation invalidate saved sampling.
    data += [hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             hashlib.sha256(Path(inspect.getsourcefile(ViggleChunkedSampler)).read_bytes()).hexdigest()]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def _read_checkpoint(directory, entry, canvas):
    path = _checkpoint_path(directory, entry["file"])
    a, b, _, latn = entry["span"]
    ch, cw = canvas
    expected = {"latent_tensor": [1, 24, latn, ch // 16, cw // 16],
                "audio": [1, 32, 2, round((b + 1) / FPS * 40) - round(a / FPS * 40)]}
    with safetensors.safe_open(path, framework="pt", device="cpu") as file:
        saved = json.loads(file.metadata()["viggle"])
        if saved != {"version": CHECKPOINT_VERSION, "entry": entry, "canvas": list(canvas)}:
            raise ValueError(f"Viggle: checkpoint metadata mismatch: {path.name}")
        for name, shape in expected.items():
            if file.get_slice(name).get_shape() != shape:
                raise ValueError(f"Viggle: invalid {name} shape in {path.name}")
        video, audio = file.get_tensor("latent_tensor"), file.get_tensor("audio")
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError(f"Viggle: NaN/Inf in saved checkpoint {path.name}")
    return video, audio


class ViggleChunkLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cond_set": ("VIGGLE_COND_SET",),
            "run_name": ("STRING", {"default": "viggle_take_01", "tooltip": "Checkpoints go to output/viggle_chunks/<run_name>. Use a different name for separate projects."}),
            "resume": ("BOOLEAN", {"default": True, "tooltip": "Load matching saved chunks; changed settings get new checkpoints. Completed chunks are decoded/saved again."}),
        }, "optional": {"initial_state": ("VIGGLE_LOOP_STATE", {"forceInput": True})}}

    RETURN_TYPES = ("VIGGLE_LOOP", "VIGGLE_LOOP_STATE", "STRING")
    RETURN_NAMES = ("loop", "state", "status")
    FUNCTION = "start"
    CATEGORY = "sampling/viggle/experimental"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def start(self, cond_set, run_name, resume, initial_state=None):
        spans = cond_set["spans"]
        if not spans or len(spans) != len(cond_set["conds"]):
            raise ValueError("Viggle: conditioning windows are empty or inconsistent.")
        directory = _run_dir(run_name)
        directory.mkdir(parents=True, exist_ok=True)
        if initial_state is None:
            state = {"plan": cond_set, "run_name": run_name, "resume": resume,
                     "index": 0, "entries": [], "previous": None, "graph_signature": None}
        else:
            state = initial_state
        return ("loop", state, f"Chunk {state['index'] + 1} of {len(spans)}")


class ViggleSampleChunk:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = ViggleChunkedSampler.INPUT_TYPES()["required"]
        return {"required": {"state": ("VIGGLE_LOOP_STATE",),
                             **{name: inputs[name] for name in ("guider", "sampler", "sigmas", "seed", "rerender_chunk", "rerender_seed")}},
                "optional": {"vae": ("VAE", {"tooltip": "Connect the H3 VAE for five_frame_anchor continuation."})},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_LOOP_STATE", "LATENT", "STRING")
    RETURN_NAMES = ("chunk", "video_latent", "filename_prefix")
    FUNCTION = "sample"
    CATEGORY = "sampling/viggle/experimental"
    DESCRIPTION = "Sample and checkpoint one chunk before external VAE decoding. Use between Viggle Chunk Loop Start and End."

    def sample(self, state, guider, sampler, sigmas, seed, rerender_chunk, rerender_seed,
               dynprompt, unique_id, vae=None):
        _validate_sigmas(sigmas)
        plan, i = state["plan"], state["index"]
        continuation = plan.get("continuation", "latent_overlap")
        if continuation == "five_frame_anchor" and vae is None:
            raise ValueError("Viggle: connect the H3 VAE to Sample Chunk for five_frame_anchor continuation.")
        a, b, lat0, latn = plan["spans"][i]
        ch, cw = plan["canvas"]
        seed_i = (int(rerender_seed) if int(rerender_chunk) == i + 1 else int(seed) + i) % (1 << 64)
        signature = state["graph_signature"] or _sampling_graph_signature(dynprompt, unique_id)
        predecessor = state["entries"][-1]["key"] if i else ""
        h = hashlib.sha256()
        values = (CHECKPOINT_VERSION, signature, continuation, predecessor, plan["conds"][i], sigmas,
                  (a, b, lat0, latn), (ch, cw), seed_i, plan.get("audio_digest", b""))
        if not _hash_cache_value(h, values):
            raise ValueError("Viggle: this conditioning cannot be fingerprinted for checkpoint recovery.")
        key = h.hexdigest()
        entry = {"file": f"chunk_{i + 1:04}_{key[:16]}.latent", "key": key,
                 "previous_key": predecessor, "span": [a, b, lat0, latn], "seed": seed_i}
        directory = _run_dir(state["run_name"])
        path = _checkpoint_path(directory, entry["file"])
        restored = state["resume"] and path.exists()
        status = f"Chunk {i + 1} of {len(plan['spans'])}: frames {a}-{b}, seed {seed_i}"
        logging.info("[ViggleSampleChunk] %s [%s]", status, "restored" if restored else "sampling")
        progress_node = dynprompt.get_display_node_id(unique_id)
        _send_progress(progress_node, status + (" — restoring" if restored else " — sampling"))
        if restored:
            out_v, out_a = _read_checkpoint(directory, entry, plan["canvas"])
        else:
            dev = comfy.model_management.intermediate_device()
            a0, a1 = round(a / FPS * 40), round((b + 1) / FPS * 40)
            v = torch.zeros([1, 24, latn, ch // 16, cw // 16], device=dev)
            # Connected audio rides in as a clean condition, so it needs no carry-over.
            cond_a = plan.get("audio_latent")
            au = (cond_a[..., a0:a1].to(device=dev, dtype=torch.float32) if cond_a is not None
                  else torch.zeros([1, 32, 2, a1 - a0], device=dev))
            previous = state["previous"]
            carry = 0
            if previous is not None:
                pa, pb, pstart, plen = previous["span"]
                if continuation == "five_frame_anchor":
                    anchor = _encode_anchor(vae, previous["video"].to(dev), a - pa)
                    carry = anchor.shape[2]
                    v[:, :, :carry] = anchor.to(v)
                else:
                    carry = max(0, pstart + plen - lat0)
                    if carry:
                        v[:, :, :carry] = previous["video"][:, :, lat0 - pstart:lat0 - pstart + carry].to(dev)
                    audio_overlap = max(0, round((pb + 1) / FPS * 40) - a0)
                    audio_offset = a0 - round(pa / FPS * 40)
                    if audio_overlap and cond_a is None:
                        au[..., :audio_overlap] = previous["audio"][..., audio_offset:audio_offset + audio_overlap].to(dev)
            out_v, out_a = ViggleChunkedSampler()._sample_window(
                core_sampler.Noise_RandomNoise(seed_i), guider, sampler, sigmas, plan["conds"][i], seed_i,
                ch, cw, a, b, carry, v, au, cond_a is not None)
            out_v, out_a = out_v.detach().cpu().contiguous(), out_a.detach().cpu().contiguous()
            metadata = {"viggle": json.dumps({"version": CHECKPOINT_VERSION, "entry": entry, "canvas": list(plan["canvas"])})}
            tensors = {"latent_tensor": out_v, "audio": out_a, "latent_format_version_0": torch.empty(0)}
            _atomic_write(path, lambda temporary: safetensors.torch.save_file(tensors, temporary, metadata=metadata))
        entries = state["entries"] + [entry]
        collection = {"version": CHECKPOINT_VERSION, "run_name": state["run_name"],
                      "canvas": list(plan["canvas"]), "total_frames": plan["total_frames"],
                      "continuation": continuation, "entries": entries}
        manifest = directory / "manifest.json"
        saved_collection = collection
        if restored and manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if (all(existing.get(k) == collection[k] for k in ("version", "run_name", "canvas", "total_frames", "continuation"))
                    and existing.get("entries", [])[:len(entries)] == entries):
                # Replaying previews must not hide the already completed suffix
                # if decoding or saving fails before the loop reaches it again.
                saved_collection = existing
        _atomic_write(manifest, lambda temporary: Path(temporary).write_text(
            json.dumps(saved_collection, indent=2), encoding="utf-8"))
        next_state = {**state, "index": i + 1, "entries": entries, "graph_signature": signature,
                      "progress_node": progress_node,
                      "previous": {"span": entry["span"], "video": out_v, "audio": out_a}}
        _send_progress(progress_node, status + (" — restored; decoding/saving" if restored else " — checkpoint saved; decoding/saving"))
        prefix = f"viggle_chunks/{state['run_name']}/preview_{i + 1:04}_{key[:8]}"
        return (next_state, {"samples": out_v.clone()}, prefix)


class ViggleHiresChunkStart:
    """Loop start for the two-stage hires extend flow.

    Emits the per-chunk noise, the low-res empty AV latent (stage 1) and the
    chunk's conditioning. Actual sampling is done by the user's own
    BasicGuider + SamplerCustomAdvanced pair; see ViggleHiresChunkPin and
    ViggleHiresChunkStore for the carry/assembly glue.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cond_set": ("VIGGLE_COND_SET",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                             "tooltip": "Seed of the first chunk; later chunks use seed + chunk index."}),
            "rerender_chunk": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                       "tooltip": "0 disables. Re-running the chain uses rerender_seed for this chunk only (the whole chain re-samples, the master is in memory)."}),
            "rerender_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        }, "optional": {"initial_state": ("VIGGLE_LOOP_STATE", {"forceInput": True})},
                     "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_LOOP", "VIGGLE_LOOP_STATE", "NOISE", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("loop", "state", "noise", "conditioning", "latent", "status")
    FUNCTION = "start"
    CATEGORY = "sampling/viggle/experimental"
    DESCRIPTION = "Per-chunk noise, low-res empty AV latent and conditioning for the two-stage hires extend flow."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def start(self, cond_set, seed, rerender_chunk, rerender_seed,
              initial_state=None, dynprompt=None, unique_id=None):
        spans = cond_set["spans"]
        if not spans or len(spans) != len(cond_set["conds"]):
            raise ValueError("Viggle: conditioning windows are empty or inconsistent.")
        if initial_state is None:
            state = {"plan": cond_set, "index": 0, "entries": [], "previous": None,
                     "seed": int(seed) % (1 << 64), "rerender_chunk": int(rerender_chunk),
                     "rerender_seed": int(rerender_seed) % (1 << 64),
                     "master_v": None, "master_a": None}
        else:
            state = initial_state
        i = state["index"]
        a, b, lat0, latn = spans[i]
        ch, cw = cond_set["canvas"]
        a0, a1 = round(a / FPS * 40), round((b + 1) / FPS * 40)
        seed_i = (state["rerender_seed"] if state["rerender_chunk"] == i + 1 else state["seed"] + i) % (1 << 64)
        dev = comfy.model_management.intermediate_device()
        video = torch.zeros([1, 24, latn, ch // 16, cw // 16], device=dev)
        # Connected driving audio anchors stage 1 as well; the lock-style nodes
        # (or Pin's mask for stage 2) decide how strictly it is held.
        audio = torch.zeros([1, 32, 2, a1 - a0], device=dev)
        cond_a = cond_set.get("audio_latent")
        if cond_a is not None:
            audio = cond_a[..., a0:a1].to(device=dev, dtype=torch.float32)
        status = f"Chunk {i + 1} of {len(spans)}: frames {a}-{b}, seed {seed_i} (stage 1: {ch}x{cw})"
        _send_progress(dynprompt.get_display_node_id(unique_id), status)
        return ("loop", state, core_sampler.Noise_RandomNoise(seed_i), cond_set["conds"][i],
                {"samples": NestedTensor((video, audio)), "latent_format_version_0": torch.empty(0)}, status)


class ViggleHiresChunkPin:
    """Insert the previous chunk's hires tail and the denoise mask.

    Connect the output of your AV concat (upscaled video + audio) here, then
    the result to the high-res SamplerCustomAdvanced's latent_image. The node
    also accepts the upscaler's video-only output directly.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "Your Concat AV Latent output (upscaled video + audio), or the upscaler's video-only output."}),
            "state": ("VIGGLE_LOOP_STATE",),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("LATENT", "VIGGLE_LOOP_STATE")
    RETURN_NAMES = ("latent", "state")
    FUNCTION = "pin"
    CATEGORY = "sampling/viggle/experimental"
    DESCRIPTION = "Carry + mask preparation between the latent upscaler and the high-res SamplerCustomAdvanced."

    def pin(self, latent, state, dynprompt, unique_id):
        plan = state["plan"]
        i = state["index"]
        a, b, lat0, latn = plan["spans"][i]
        a0, a1 = round(a / FPS * 40), round((b + 1) / FPS * 40)
        rows = a1 - a0
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            video, audio = samples.unbind()
        else:
            video, audio = samples, None
        if video.shape[2] != latn:
            raise ValueError(f"Viggle: the hires latent has {video.shape[2]} temporal latents, expected {latn} "
                             f"for frames {a}-{b}. Check the upscaler's temporal settings.")
        carry = 0
        previous = state["previous"]
        if i > 0 and previous is not None and previous.get("video") is not None:
            _, _, prev_lat0, prev_latn = plan["spans"][i - 1]
            carry = max(0, prev_lat0 + prev_latn - lat0)
            tail = previous["video"]
            if tail.shape[2] < carry or tail.shape[-2:] != video.shape[-2:]:
                raise ValueError("Viggle: the previous chunk's tail does not match this window "
                                 "(the hires canvas must be identical for every chunk).")
            video = video.clone()
            video[:, :, :carry] = tail[:, :, :carry].to(video)
        cond_a = plan.get("audio_latent")
        audio_clean = cond_a is not None
        if cond_a is not None:
            # The plan's clean driving slice is canonical; it overrides whatever
            # the stage-1 branch produced.
            audio = cond_a[..., a0:a1].to(device=video.device, dtype=torch.float32)
        elif audio is None:
            audio = torch.zeros([1, 32, 2, rows], device=video.device)
        if audio.shape[-1] != rows:
            fixed = torch.zeros([1, 32, 2, rows], device=audio.device)
            fixed[..., :min(rows, audio.shape[-1])] = audio[..., :min(rows, audio.shape[-1])]
            audio = fixed
        mask = None
        if carry or audio_clean:
            mask_v = torch.ones([1, 1, latn, video.shape[3], video.shape[4]], device=video.device)
            if carry:
                mask_v[:, :, :carry] = 0
            mask_a = torch.full([1, 1, 1, rows], 0.0 if audio_clean else 1.0, device=video.device)
            mask = NestedTensor((mask_v, mask_a))
        out = dict(latent)
        out["samples"] = NestedTensor((video, audio))
        if mask is not None:
            out["noise_mask"] = mask
        _send_progress(dynprompt.get_display_node_id(unique_id),
                       f"Chunk {i + 1} of {len(plan['spans'])}: pinned {carry} carry latents at "
                       f"{video.shape[3]}x{video.shape[4]}" + (", clean audio" if audio_clean else ""))
        return (out, state)


class ViggleHiresChunkStore:
    """Collect the high-res window into the in-memory master latent."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "The high-res SamplerCustomAdvanced output of this window."}),
            "state": ("VIGGLE_LOOP_STATE",),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_LOOP_STATE", "LATENT", "STRING")
    RETURN_NAMES = ("chunk", "master", "status")
    FUNCTION = "store"
    CATEGORY = "sampling/viggle/experimental"
    DESCRIPTION = "Assembles the high-res master between the high-res sampler and Viggle Chunk Loop End."

    def store(self, latent, state, dynprompt, unique_id):
        plan = state["plan"]
        i = state["index"]
        spans = plan["spans"]
        a, b, lat0, latn = spans[i]
        a0, a1 = round(a / FPS * 40), round((b + 1) / FPS * 40)
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            video, audio = samples.unbind()
        else:
            video, audio = samples, None
        if video.shape[2] != latn:
            raise ValueError(f"Viggle: the hires output has {video.shape[2]} temporal latents, expected {latn}.")
        video = video.detach().to(torch.float32).contiguous()
        if audio is not None:
            audio = audio.detach().to(torch.float32).contiguous()
        if not torch.isfinite(video).all() or (audio is not None and not torch.isfinite(audio).all()):
            raise ValueError(f"Viggle: chunk {i + 1} contains NaN/Inf.")
        total_lat = spans[-1][2] + spans[-1][3]
        total_a = round(plan["total_frames"] / FPS * 40)
        master_v, master_a = state["master_v"], state["master_a"]
        if master_v is None:
            if i:
                raise ValueError("Viggle: the hires master was lost; restart the loop from chunk 1.")
            master_v = torch.zeros([1, 24, total_lat, video.shape[3], video.shape[4]], device="cpu")
            master_a = torch.zeros([1, 32, 2, total_a], device="cpu")
        else:
            if (video.shape[3], video.shape[4]) != master_v.shape[-2:]:
                raise ValueError("Viggle: the hires canvas changed mid-run (upscale settings differ between chunks).")
            if i != len(state["entries"]):
                raise ValueError("Viggle: chunk order out of sync with the master; restart the loop from chunk 1.")
        master_v[:, :, lat0:lat0 + latn] = video.to(master_v)
        if audio is not None and audio.shape[-1] == a1 - a0:
            master_a[..., a0:a1] = audio.to(master_a)
        carry_next = max(0, lat0 + latn - spans[i + 1][2]) if i + 1 < len(spans) else 0
        previous = {"span": [a, b, lat0, latn],
                    "video": (video[:, :, -carry_next:].clone().cpu() if carry_next else None)}
        entries = state["entries"] + [{"span": [a, b, lat0, latn]}]
        status = (f"Chunk {i + 1} of {len(spans)} stored: master "
                  f"{master_v.shape[2]}x{video.shape[3]}x{video.shape[4]}")
        _send_progress(dynprompt.get_display_node_id(unique_id), status)
        next_state = {**state, "index": i + 1, "entries": entries, "previous": previous,
                      "master_v": master_v, "master_a": master_a}
        return (next_state, {"samples": NestedTensor((master_v, master_a))}, status)


class ViggleChunkLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"loop": ("VIGGLE_LOOP", {"rawLink": True}),
                             "chunk": ("VIGGLE_LOOP_STATE",),
                             "images": ("IMAGE", {"tooltip": "Connect VAE Decode through SaveWEBM (images output) to save each chunk before advancing."})},
                "optional": {"after_save": ("*", {"tooltip": "Optional completion dependency, e.g. VHS Video Combine's filenames output."})},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIGGLE_CHUNKS", "STRING", "LATENT")
    RETURN_NAMES = ("chunks", "status", "master")
    FUNCTION = "finish"
    CATEGORY = "sampling/viggle/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = "Wait for this chunk's decode/save branch, then advance the loop. Connect loop directly from Loop Start."

    def finish(self, loop, chunk, images, dynprompt, unique_id, after_save=None):
        if not torch.isfinite(images).all():
            raise ValueError("Viggle: decoded chunk contains NaN/Inf. Its latent checkpoint is already saved.")
        if "progress_node" in chunk:
            done = chunk["index"] == len(chunk["plan"]["spans"])
            _send_progress(chunk["progress_node"],
                           f"Chunk {chunk['index']} of {len(chunk['plan']['spans'])} — "
                           + ("loop completed (final assembly/decode may follow)" if done else "decode/save finished"))
        if chunk["index"] == len(chunk["plan"]["spans"]):
            run_name = chunk.get("run_name")
            collection = {"version": CHECKPOINT_VERSION,
                          "canvas": list(chunk["plan"]["canvas"]),
                          "total_frames": chunk["plan"]["total_frames"],
                          "continuation": chunk["plan"].get("continuation", "latent_overlap"),
                          "entries": chunk["entries"]}
            if run_name is not None:
                collection["run_name"] = run_name
                message = f"Completed {chunk['index']} chunks. Saved in {_run_dir(run_name)}"
            else:
                message = f"Completed {chunk['index']} chunks (hires master kept in memory)."
            master_v = chunk.get("master_v")
            master = ({"samples": NestedTensor((master_v, chunk["master_a"]))} if master_v is not None else None)
            return (collection, message, master)

        # Copy only nodes lying between Loop Start and this End, as in ComfyUI's
        # graph-expansion loop example. External loaders/conditioning stay shared.
        children, visited = {}, set()
        pending = [unique_id]
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            for value in dynprompt.get_node(node_id).get("inputs", {}).values():
                if is_link(value):
                    children.setdefault(value[0], set()).add(node_id)
                    pending.append(value[0])
        start_id = loop[0]
        if dynprompt.get_node(start_id)["class_type"] not in ("ViggleChunkLoopStart", "ViggleHiresChunkStart"):
            raise ValueError("Viggle: connect Loop End's loop directly to Viggle Chunk Loop Start (or Hires Chunk Start).")
        contained, pending = set(), [start_id]
        while pending:
            node_id = pending.pop()
            if node_id not in contained:
                contained.add(node_id)
                pending.extend(children.get(node_id, ()))
        graph = GraphBuilder()
        copies = {}
        for number, node_id in enumerate(sorted(contained)):
            original = dynprompt.get_node(node_id)
            node = graph.node(original["class_type"], "next" if node_id == unique_id else str(number))
            node.set_override_display_id(dynprompt.get_display_node_id(node_id))
            copies[node_id] = node
        for node_id, node in copies.items():
            for name, value in dynprompt.get_node(node_id).get("inputs", {}).items():
                node.set_input(name, copies[value[0]].out(value[1]) if is_link(value) and value[0] in copies else value)
        copies[start_id].set_input("initial_state", dynprompt.get_node(unique_id)["inputs"]["chunk"])
        return {"result": (copies[unique_id].out(0), copies[unique_id].out(1), copies[unique_id].out(2)),
                "expand": graph.finalize()}


class ViggleAssembleChunkLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "run_name": ("STRING", {"default": "viggle_take_01", "tooltip": "Used when chunks is not connected; loads the saved manifest, including interrupted runs."}),
            "chunk_number": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 assembles all completed chunks. A positive number loads that one chunk for inspection."}),
        }, "optional": {"chunks": ("VIGGLE_CHUNKS",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("video_latent", "status")
    FUNCTION = "assemble"
    CATEGORY = "sampling/viggle/experimental"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(self, run_name, chunk_number, chunks=None):
        if chunks is None:
            chunks = json.loads((_run_dir(run_name) / "manifest.json").read_text(encoding="utf-8"))
        directory = _run_dir(chunks["run_name"])
        entries = chunks["entries"]
        if chunks["version"] != CHECKPOINT_VERSION or not entries:
            raise ValueError("Viggle: unsupported or empty checkpoint manifest.")
        if chunk_number:
            if chunk_number > len(entries):
                raise ValueError(f"Viggle: only {len(entries)} chunks are saved.")
            entry = entries[chunk_number - 1]
            video, _ = _read_checkpoint(directory, entry, chunks["canvas"])
            return ({"samples": video}, f"Chunk {chunk_number}: frames {entry['span'][0]}-{entry['span'][1]} (includes overlap)")
        last = entries[-1]["span"]
        ch, cw = chunks["canvas"]
        master = None
        predecessor, previous_end = "", 0
        for entry in entries:
            a, b, start, count = entry["span"]
            if entry["previous_key"] != predecessor or start > previous_end or start < 0:
                raise ValueError("Viggle: checkpoints do not form a continuous matching chain.")
            video, _ = _read_checkpoint(directory, entry, chunks["canvas"])
            if master is None:
                master = torch.empty((1, 24, last[2] + last[3], ch // 16, cw // 16), dtype=video.dtype, device="cpu")
            local_start = max(0, previous_end - start) if chunks.get("continuation") == "five_frame_anchor" else 0
            master[:, :, start + local_start:start + count] = video[:, :, local_start:]
            predecessor, previous_end = entry["key"], start + count
        complete = last[1] + 1 == chunks["total_frames"]
        return ({"samples": master}, f"{'Complete' if complete else 'Partial'}: {len(entries)} chunks, {last[1] + 1} frames. Decode once for the final video.")


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (ViggleChunkLoopStart, ViggleSampleChunk,
                                                   ViggleHiresChunkStart, ViggleHiresChunkPin,
                                                   ViggleHiresChunkStore,
                                                   ViggleChunkLoopEnd, ViggleAssembleChunkLatents)}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleChunkLoopStart": "Viggle Chunk Loop Start",
                            "ViggleSampleChunk": "Viggle Sample Chunk",
                            "ViggleHiresChunkStart": "Viggle Hires Chunk Start",
                            "ViggleHiresChunkPin": "Viggle Hires Chunk Pin",
                            "ViggleHiresChunkStore": "Viggle Hires Chunk Store",
                            "ViggleChunkLoopEnd": "Viggle Chunk Loop End",
                            "ViggleAssembleChunkLatents": "Viggle Assemble Chunk Latents"}
