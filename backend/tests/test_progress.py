"""Tests for the mlx-whisper tqdm progress hook."""

from __future__ import annotations

import tqdm as tqdm_module

from app.providers.tqdm_progress import hook_tqdm, make_progress_tqdm


def test_counts_even_when_tqdm_would_be_disabled():
    seen: list[float] = []
    cls = make_progress_tqdm(seen.append)
    bar = cls(total=100)
    assert bar.disable is True, "hook must keep the worker console quiet"
    bar.update(25)
    bar.update(25)
    assert seen == [0.25, 0.5]


def test_fraction_is_clamped_to_one():
    seen: list[float] = []
    bar = make_progress_tqdm(seen.append)(total=100)
    bar.update(150)
    bar.update(50)
    assert seen == [1.0]


def test_no_total_means_no_reports():
    seen: list[float] = []
    bar = make_progress_tqdm(seen.append)(total=None)
    bar.update(5)
    assert seen == []


def test_tiny_steps_are_filtered_but_one_is_always_reported():
    seen: list[float] = []
    bar = make_progress_tqdm(seen.append)(total=1000)
    bar.update(1)  # 0.001: below the reporting step
    assert seen == []
    bar.update(4)  # 0.005: reported
    assert seen == [0.005]
    bar.update(995)  # reaches 1.0
    assert seen[-1] == 1.0


def test_hook_restores_original_tqdm():
    original = tqdm_module.tqdm
    with hook_tqdm(lambda fraction: None):
        assert tqdm_module.tqdm is not original
        bar = tqdm_module.tqdm(total=10)
        bar.update(10)
    assert tqdm_module.tqdm is original


def test_use_as_context_manager_like_mlx_whisper():
    seen: list[float] = []
    cls = make_progress_tqdm(seen.append)
    with cls(total=10) as bar:
        bar.update(3)
    assert seen == [0.3]
