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
