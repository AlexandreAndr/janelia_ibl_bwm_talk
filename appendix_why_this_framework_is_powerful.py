# %% [markdown]
# # Appendix: Why this framework is powerful
#
# *Moved out of `wheel_speed_minimal_example.ipynb` for standalone work; pending
# reintegration into that notebook.*
#
# The training pipeline in `wheel_speed_minimal_example` is four independent
# pieces: **Dataset** (*where* to sample), **Sampler** (*which* windows),
# **Transform** (*how* to process), and **DataLoader** (*how* to batch). Because
# they are decoupled, common neuroscience experiments (finer binning, more
# context, augmentation, region ablations, jittered sampling) are each a
# **one-line change that never touches the model or the training loop**.
#
# Every cell below is illustrative: it operates on a single demo window and does
# **not** modify the tuned pipeline in the main notebook.

# %% [markdown]
# # 1. Lazy loading and the time gain
#
# A `brainsets` recording is memory-mapped: `data.slice(start, end)` reads only
# the bytes for the window you ask for. You open a session in milliseconds
# regardless of its size, and each sample touches a tiny slice of it, so you
# never hold the session (or even one full modality) in RAM. That is what
# makes training over a 65-min session, or scaling to many sessions, feasible
# at all.
#
# The catch: lazy is cheap *per call*, but `IBLBrainWideMap2025.__getitem__`
# redoes the slice + bin on *every* epoch, since it does not cache. For this
# tutorial's scale (a single session, ~1 ms per window) that is a deliberate
# trade: a few extra seconds of redundant HDF5 reads and binning over a full
# training run, in exchange for a `__getitem__` with no memoization state to
# reason about. Memoizing the handful of windows a run actually samples (keyed
# on `(recording_id, start, end)`) would trade that simplicity back for speed
# if this were scaled to more sessions, more epochs, or heavier per-sample
# processing.
#
# # %%
# import os
# import time
#
# fpath = os.path.join(DATA_ROOT, "ibl_brain_wide_map_2025", f"{RECORDING_ID}.h5")
# print(f"session file on disk:       {os.path.getsize(fpath) / 1e9:5.2f} GB")
#
# t = time.time()
# _d = IBLBrainWideMap2025(root=DATA_ROOT, recording_id=RECORDING_ID, split=None)
# _r = _d.get_recording(RECORDING_ID)
# t_open = time.time() - t
#
# t = time.time()
# for _ in range(20):
#     _s = _r.slice(DEMO_T0, DEMO_T0 + 1.0)
#     _ = bin_spikes(_s.spikes, num_units=len(_s.units), bin_size=BIN_SIZE)
# t_slice = (time.time() - t) / 20
#
# t = time.time()
# _allt = np.asarray(_r.spikes.timestamps)
# t_all = time.time() - t
#
# print(f"open dataset (lazy):        {t_open * 1000:5.1f} ms   (reads no signals)")
# print(
#     f"one 1.0 s window slice+bin: {t_slice * 1000:5.1f} ms   (reads only that window)"
# )
# _all_mb = _allt.nbytes / 1e6
# print(
#     f"load ALL spike timestamps:  {t_all * 1000:5.0f} ms   ({_all_mb:.0f} MB, one modality)"
# )
#
# # %% [markdown]
# # ## Transforms: deferring computation to each slice
# #
# # That last point is exactly what a **transform** is: a small function applied to a
# # window right after the lazy slice and before the model, inside `__getitem__`.
# # Because it runs per sample on a tiny crop, it is cheap, and swapping or chaining
# # transforms reshapes the pipeline without touching the data on disk or the model.
# #
# # Binning is the clearest example. Spikes arrive as an irregular event stream, but
# # the model wants a regular grid, so each window is binned on the fly. The bin
# # width is just a parameter: change it and the model input reshapes, with no
# # reprocessing.
#
# # %%
# from torch_brain.utils import bin_spikes
#
# window = lazy_rec.slice(t0, t0 + 1.0)
# for bs in (0.02, 0.05, 0.10):
#     Xb = bin_spikes(window.spikes, num_units=len(window.units), bin_size=bs)
#     print(f"bin_size = {bs:.2f} s  ->  X shape {tuple(Xb.shape)}  (bins x units)")
#
# # %% [markdown]
# # # 2. Change the bin size
# #
# # Binning is one parameter. Finer bins give more temporal resolution but a
# # higher-dimensional, sparser input; coarser bins regularize. The model still
# # just receives `(num_bins, num_units)`.
#
# # %%
# fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
# for ax, bs in zip(axes, [0.02, 0.05, 0.1]):
#     s = demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)
#     Xd = bin_spikes(s.spikes, num_units=len(s.units), bin_size=bs)
#     ax.imshow(
#         np.log1p(Xd.T),
#         aspect="auto",
#         cmap="Greys",
#         origin="lower",
#         interpolation="nearest",
#     )
#     ax.set_title(f"bin_size = {bs} s  ({Xd.shape[0]} bins)")
#     ax.set_xlabel("time bin")
# axes[0].set_ylabel("unit")
# fig.suptitle("Same 1.0 s window, three bin sizes: one parameter, no model change")
# plt.tight_layout()
# plt.show()
#
# # %% [markdown]
# # # 3. Longer or shorter context window
# #
# # The context window is how much time each sample spans; it lives in the
# # sampler/slice length, not in the data. A 2.0 s window simply yields 100 target
# # samples (at 50 Hz) and `2.0 / bin_size` input bins.
#
# # %%
# fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# for ax, L in zip(axes, [0.5, 1.0, 2.0]):
#     s = demo_rec.slice(DEMO_T0, DEMO_T0 + L)
#     y = np.asarray(s.wheel.speed)
#     ax.plot(np.linspace(0, L, len(y)), y, color="#4C72B0")
#     ax.set_title(f"{L} s -> {len(y)} target samples")
#     ax.set_xlabel("time in window (s)")
# axes[0].set_ylabel("wheel speed")
# fig.suptitle("Same start, three context windows: only the slice length changes")
# plt.tight_layout()
# plt.show()
#
# # %% [markdown]
# # # 4. Augmentation via composable transforms
# #
# # Transforms attach with `transform=` and chain with `Compose`; they run inside
# # `__getitem__`, so augmentation is re-drawn every epoch. Here a random 0.8 s crop
# # of the same window (a different crop each call). *(Time-warping augmentations
# # like `RandomTimeScaling` also exist; we revisit those separately.)*
#
# # %%
# augment = Compose([RandomCrop(0.8)])
# base = demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)
# y_base = np.asarray(base.wheel.speed)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
# axes[0].plot(np.linspace(0, 1.0, len(y_base)), y_base, color="#333333")
# axes[0].set_title(f"raw (1.0 s, {len(y_base)} samples)")
# for _ in range(3):
#     a = augment(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0))
#     y_a = np.asarray(a.wheel.speed)
#     axes[1].plot(np.linspace(0, len(y_a) / 50.0, len(y_a)), y_a, alpha=0.7)
# axes[1].set_title("RandomCrop(0.8): a fresh crop each epoch")
# for ax in axes:
#     ax.set_xlabel("time (s)")
# axes[0].set_ylabel("wheel speed")
# fig.suptitle("A transform re-draws every epoch, combating overfitting for free")
# plt.tight_layout()
# plt.show()
#
# # %% [markdown]
# # # 5. Mask or select units
# #
# # Units are a labelled axis, so you can drop or select them declaratively.
# # `UnitDropout` keeps a random subset each sample (augmentation / regularizer);
# # `UnitFilter` keeps a fixed subset, e.g. one brain region, turning "which region
# # drives wheel decoding?" into a one-line ablation.
#
# # %%
# # (a) UnitDropout: a different random subset of units each call
# for _ in range(5):
#     before = len(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0).units)
#     after = len(
#         UnitDropout(field="spikes")(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)).units
#     )
#     print(f"UnitDropout: {before} -> {after} units kept")
#
# # (b) UnitFilter: keep only units in one brain region (cosmos atlas)
# regions = np.asarray(demo_rec.units.region_cosmos)
# labels, counts = np.unique(regions, return_counts=True)
# fig, ax = plt.subplots(figsize=(7, 3))
# ax.bar([r.decode() for r in labels], counts, color="#4C72B0")
# ax.set_ylabel("units")
# ax.set_title("Units per brain region: UnitFilter selects any subset in one line")
# plt.tight_layout()
# plt.show()
#
# th_filter = UnitFilter(
#     mask_fn=lambda u: np.asarray(u.region_cosmos) == b"TH", field="spikes"
# )
# n_before = len(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0).units)
# n_after = len(th_filter(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)).units)
# print(f"UnitFilter to thalamus: {n_before} -> {n_after} units")
#
# # %% [markdown]
# # # 6. Sampling window with jitter
# #
# # `TrialSampler` emits the same fixed windows every epoch (reproducible eval).
# # `RandomFixedWindowSampler` applies a fresh random temporal jitter each epoch, so
# # the model sees slightly shifted positions within every interval: more effective
# # data, less positional overfitting. Only the sampler changes.
#
# # %%
# jitter_sampler = RandomFixedWindowSampler(
#     sampling_intervals={RECORDING_ID: demo_iv}, window_length=1.0
# )
# fixed_sampler = TrialSampler(sampling_intervals={RECORDING_ID: demo_iv})
#
#
# def first_starts(sampler, n=6):
#     return [round(float(idx.start), 2) for idx in list(sampler)[:n]]
#
#
# print("RandomFixedWindowSampler (jitter, new positions each epoch):")
# print("  epoch 1:", first_starts(jitter_sampler))
# print("  epoch 2:", first_starts(jitter_sampler))
# print("TrialSampler (fixed windows, identical each epoch):")
# print("  epoch 1:", first_starts(fixed_sampler))
# print("  epoch 2:", first_starts(fixed_sampler))
