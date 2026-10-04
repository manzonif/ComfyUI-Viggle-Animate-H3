# v1.3.5 — Extend the frozen text conditioning

The finetune ships a frozen 362-token text embedding (`fixed_embed_fwd_anyframe`): one global Qwen3-VL-32B (layer 50, unnormalized) sequence of the fixed prompt — not per-frame. The H3 DiT consumes variable-length text (learned projection + refiner + attention), so extra tokens from the same encoder can be appended to it.

- **Viggle Text Cond Extend** (`loaders/viggle`): takes the frozen `TEXT_COND` plus an optional `CLIP` (CLIPLoader type `minimax` — the Qwen3-VL-32B encoder your H3 workflow already loads) and an `append_text` string. Empty text is a no-op passthrough, so existing graphs are unaffected. With `replace_frozen` the frozen 362 tokens are dropped and only the encoded text is used.
- The node concatenates embeddings and modality tags (frozen first, appended text after), matching dtype/device, and rejects a clip with a different embedding dimension.

Caveat: the finetune was evaluated on the exact 362-token presentation. Appending keeps the fixed prompt and adds context (clothing, style, action modifiers) and is the sane first experiment; replacing is farther from the trained distribution. A/B against the stock prompt before trusting either.

# v1.3.4 — Two-stage hires extend loop

Three new nodes replicate the single-shot two-stage recipe — 1 step at low resolution, latent upscale, then 4 steps at high resolution with a second model/LoRA stack — inside the chunked loop, while leaving the actual sampling to ComfyUI's **native `SamplerCustomAdvanced`**. The user keeps full control of both model stacks, the upscaler and the sigma split (`SplitSigmas` etc.); the new nodes only add the per-chunk machinery.

- **Viggle Hires Chunk Start** (loop start, `sampling/viggle/experimental`): per-chunk `noise` (seed + chunk index, with `rerender_chunk`/`rerender_seed` override), the window's empty AV `latent` at the plan canvas (stage 1), the chunk's `conditioning` (connect to both guiders), plus `loop`/`state`. If the windowed conditioning has a driving audio connected, its clean slice is pre-filled into the latent's audio rows.
- **Viggle Hires Chunk Pin**: sits between the AV concat (upscaled video + audio) and the high-res `SamplerCustomAdvanced`. It writes the previous chunk's high-res tail (the overlapping latents) at the front of the window, pins the driving audio's clean slice into the audio rows (denoise mask 0), and hands the nested denoise mask to the native sampler through `latent["noise_mask"]`. The first chunk passes through unmodified, so chunk 1 is exactly the single-shot workflow.
- **Viggle Hires Chunk Store**: collects the high-res window into an in-memory master (overwriting the overlap), keeps the tail for the next carry and forwards the state to the existing **Viggle Chunk Loop End**.
- **Viggle Chunk Loop End** gains a third output, `master` (LATENT): the assembled high-res nested AV latent when the loop completes — feed it to a normal VAE Decode. Legacy loop states return `None` there, so existing graphs are unchanged.

Notes:

- The final canvas is defined by **your upscaler** and must be identical for every chunk (the nodes validate this). Stage 1 runs at the plan canvas (the windowed conditioning's `width`/`height`). To use high-res references later, move the conditioning to the high-res canvas while stage 1 stays at the low canvas.
- The master is in memory in this release (as in `Viggle Chunked Sampler`); per-chunk decode/save still runs through the Loop End `images` branch. Disk checkpoints for the hires flow follow in a later release.
- `rerender_chunk`/`rerender_seed` re-runs the whole chain with a different seed for that chunk only; earlier chunks reproduce with their unchanged seeds (the master is in memory, so the carry rebuilds from scratch).

Validation: 16 new tests (unit plus a full executor-level loop driving two native `SamplerCustomAdvanced` stages, covering per-chunk seeds, carry placement, mask contents, clean-audio passthrough, master assembly and the terminal master output). The existing 47 tests are unchanged except two Loop End tests, which now also cover the new output.

# v1.3.3 — Lip-sync from the driving audio

The H3 authors note that lip-sync improves when the model receives the driving clip's audio as a real soundtrack, encoded with the audio VAE and held as a clean latent in the target audio rows for the whole denoise. This release wires that up for the windowed and loop paths: previously the audio rows were always empty, so the model generated a throwaway track and the mouth had nothing to follow.

- **Viggle-Animate Conditioning (H3, Windowed)** gains optional `audio`, `audio_vae` and `fps` inputs (appended after `continuation`, so saved workflows keep their widget order). The driving soundtrack is encoded once, cut or zero-padded to the render's 40-rows-per-second audio grid, and sliced per window, so every chunk conditions on the rows for its own frame range — including the five anchor frames.
- Those rows are held clean (denoise mask 0) through every step, which is where the audio-conditioned chunks differ from before: mask 0 pins the rows to the supplied latent and zeroes the predicted audio velocity, so the model reads the track instead of rewriting it.
- `fps` maps a clip that was not loaded at 24 fps onto the 24 fps render timeline (24 leaves it untouched). It never changes the video timing.
- **Viggle Chunked Sampler** returns the assembled `audio_latent` as a third output, and `chunk_map` states whether the audio is the conditioned soundtrack or a generated track.
- The encoded track's fingerprint joins the chunk cache key and the loop checkpoint fingerprint, so a different soundtrack never replays a chunk rendered against another one.
- With `audio` left empty, behaviour is unchanged: empty audio rows, generated track, discarded at save time.

**After updating:** restart ComfyUI and refresh the browser. Load the MiniMax-H3 **audio** VAE (`minimax_h3_audio_vae*.safetensors`) into a VAELoader and wire the driving clip's audio plus that VAE into Windowed Conditioning; connect nothing to keep the old behaviour. The sampler's new third output means saved graphs may show one unconnected output port.

Validation: 47 Python regression tests (25 sampler/conditioning, 22 loop), including per-window audio slicing, clean-mask behaviour, timeline mapping through the audio VAE and checkpoint invalidation on a soundtrack change. Lip-sync quality still needs confirmation on real clips in ComfyUI.

# v1.3.2 — Fix long-video ending drift

Fix a final-window conditioning mismatch introduced by rounding source lengths up to the H3 frame grid. For example, a 289-frame source could supply 32 reference latents against 37 target latents in the last window. Padding the reference before VAE encoding aligns those lengths; the maintainer confirmed the fix on the affected clip.

- Restore full-length, end-aligned final windows.
- Add five-frame decoded/re-encoded continuation anchors, enabled by default. `latent_overlap` remains available for comparison.
- Preserve accepted output when final windows overlap earlier chunks.
- Trim the Chunked Sampler output to the loaded source frame count.
- Update example workflows, English/Chinese documentation, cache and checkpoint handling.

**After updating:** restart ComfyUI and refresh the browser. For older advanced loop workflows, connect the H3 VAE to Sample Chunk when using `five_frame_anchor`. Code updates invalidate automatic checkpoint reuse; saved latent files remain readable. The advanced loop's external decode still includes grid padding.

Validation: 42 Python regression tests covering reference encoding, anchor placement, assembly, caching and loop recovery. Motion quality can still vary by clip.
