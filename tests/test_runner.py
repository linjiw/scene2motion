from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene2motion.constraints import ConstraintSpec
from scene2motion.runner import ArdyRunner, _per_sample_noise


def draws(seeds):
    with _per_sample_noise(seeds, "cpu"):
        return torch.randn((len(seeds), 5)), torch.randn((len(seeds), 5))


def test_per_sample_noise_reproducible_but_advances_between_windows():
    first_a, second_a = draws([11, 29])
    first_b, second_b = draws([11, 29])

    torch.testing.assert_close(first_a, first_b)
    torch.testing.assert_close(second_a, second_b)
    assert not torch.equal(first_a, second_a)


def test_per_sample_noise_draws_are_independent_of_batch_neighbors():
    together_first, together_second = draws([11, 29])
    alone_first, alone_second = draws([29])

    torch.testing.assert_close(together_first[1], alone_first[0])
    torch.testing.assert_close(together_second[1], alone_second[0])


class _FakeScheduledModel:
    gen_horizon_len = 2
    num_frames_per_token = 1

    def __init__(self):
        self.calls = []
        self.motion_rep = _FakeMotionRep()

    def autoregressive_step(self, *, text_feat, init_history_sequence, **kwargs):
        self.calls.append({
            "num_frames": int(kwargs["num_frames"]),
            "history_frames": (
                0 if init_history_sequence is None
                else int(init_history_sequence.shape[1])
            ),
            "motion_mask_shape": (
                None if kwargs["motion_mask"] is None
                else tuple(kwargs["motion_mask"].shape)
            ),
            "observed_motion_shape": (
                None if kwargs["observed_motion"] is None
                else tuple(kwargs["observed_motion"].shape)
            ),
        })
        batch = len(text_feat)
        latent = torch.randn((batch, 1, 4), device=text_feat.device)
        prompt_code = text_feat[:, :1, :1]
        new = latent.repeat(1, 2, 1)
        new[:, :, :1] += prompt_code
        return new if init_history_sequence is None else torch.cat(
            [init_history_sequence, new], dim=1)


class _FakeMotionRep:
    nfeats_dict = {"root_pos": 3}
    motion_rep_dim = 4

    @staticmethod
    def create_conditions_from_constraints_batched(
        per_sample, lengths, *, to_normalize, device
    ):
        assert to_normalize is True
        frames = int(lengths[0].item())
        batch = len(per_sample)
        observed = torch.zeros((batch, frames, 4), device=device)
        mask = torch.ones((batch, frames, 4), device=device)
        return observed, mask


class _HistoryRewritingScheduledModel(_FakeScheduledModel):
    """Mimic ARDY's lossy history encode/decode on every continuation."""

    def autoregressive_step(self, *, text_feat, init_history_sequence, **kwargs):
        returned = super().autoregressive_step(
            text_feat=text_feat,
            init_history_sequence=init_history_sequence,
            **kwargs,
        )
        if init_history_sequence is not None:
            returned[:, : init_history_sequence.shape[1]] += 0.25
        return returned


class _HistoryMutatingScheduledModel(_FakeScheduledModel):
    def autoregressive_step(self, *, text_feat, init_history_sequence, **kwargs):
        if init_history_sequence is not None:
            init_history_sequence.add_(1.0)
        return super().autoregressive_step(
            text_feat=text_feat,
            init_history_sequence=init_history_sequence,
            **kwargs,
        )


def _fake_schedule_runner():
    runner = ArdyRunner.__new__(ArdyRunner)
    runner.device = "cpu"
    runner.model = _FakeScheduledModel()
    runner.skeleton = SimpleNamespace(root_idx=0)
    runner.encode = lambda prompts: (
        torch.tensor([[[0.0 if p == "W" else 10.0]] for p in prompts]),
        torch.ones((len(prompts), 1), dtype=torch.bool),
    )
    return runner


def test_prompt_schedule_keeps_one_advancing_noise_stream_per_row():
    runner = _fake_schedule_runner()
    # Rows 0/1 share a seed and first prompt, then fork. Row 2 shares both prompts with
    # row 0. The duplicate seeds must produce equal corresponding-window noise without
    # repeating the latent from window 0 at window 1.
    features, audit = runner.generate_prompt_schedule(
        [("W", "W"), ("W", "S"), ("W", "W")],
        [None, None, None], num_frames=4, diffusion_steps=1,
        seeds=[17, 17, 17])
    assert features.shape == (3, 4, 4)
    assert len(audit) == 2
    assert audit[0]["row_sha256"][0] == audit[0]["row_sha256"][1]
    assert audit[0]["row_sha256"][0] == audit[0]["row_sha256"][2]
    assert audit[1]["row_sha256"][0] == audit[1]["row_sha256"][1]
    assert audit[0]["row_sha256"][0] != audit[1]["row_sha256"][0]
    assert [item["global_history_start_frame"] for item in audit] == [0, 1]
    assert [item["accepted_transcript_frames_before"] for item in audit] == [0, 2]
    assert [item["input_history_frames"] for item in audit] == [0, 1]
    assert [item["model_num_frames"] for item in audit] == [4, 3]
    assert [item["transcript_frames"] for item in audit] == [2, 4]
    np.testing.assert_array_equal(features[0, :2], features[1, :2])
    np.testing.assert_array_equal(features[0], features[2])
    assert not np.array_equal(features[0, 2:], features[1, 2:])


def test_prompt_schedule_preserves_transcript_when_model_rewrites_history():
    runner = _fake_schedule_runner()
    runner.model = _HistoryRewritingScheduledModel()
    features, audit = runner.generate_prompt_schedule(
        [("W", "W"), ("W", "S")], [None, None], num_frames=4,
        diffusion_steps=1, seeds=[17, 17])

    # The model returned a rewritten prefix on window 1, but the accepted transcript keeps
    # the exact window-0 bytes.  This is the official interactive-demo append semantics.
    np.testing.assert_array_equal(features[0, :2], features[1, :2])
    assert audit[1]["returned_history_reconstruction_exact"] == [False, False]
    assert audit[1]["returned_history_reconstruction_max_abs"] == pytest.approx(
        [0.25, 0.25]
    )
    assert (
        audit[1]["input_history_row_sha256"][0]
        != audit[1]["returned_input_history_row_sha256"][0]
    )
    assert audit[1]["transcript_frames"] == 4


def test_prompt_schedule_rejects_in_place_history_mutation():
    runner = _fake_schedule_runner()
    runner.model = _HistoryMutatingScheduledModel()
    with pytest.raises(RuntimeError, match="mutated its supplied history"):
        runner.generate_prompt_schedule(
            [("W", "W")], [None], num_frames=4, diffusion_steps=1, seeds=[17]
        )


def test_prompt_schedule_slices_constraints_with_gui_default_history_window():
    runner = _fake_schedule_runner()
    spec = ConstraintSpec(
        root_xz=np.zeros((4, 2)), heading=None, root_y=None, first_heading=0.0)
    runner.generate_prompt_schedule(
        [("W", "S")], [spec], num_frames=4, diffusion_steps=1, seeds=[17])

    assert runner.model.calls == [
        {
            "num_frames": 4,
            "history_frames": 0,
            "motion_mask_shape": (1, 4, 4),
            "observed_motion_shape": (1, 4, 4),
        },
        {
            "num_frames": 3,
            "history_frames": 1,
            "motion_mask_shape": (1, 3, 4),
            "observed_motion_shape": (1, 3, 4),
        },
    ]


def test_prompt_schedule_validates_window_count_and_lengths():
    runner = _fake_schedule_runner()
    with pytest.raises(ValueError, match="must contain 2"):
        runner.generate_prompt_schedule(
            [("W",)], [None], num_frames=4, diffusion_steps=1, seeds=[1])
    with pytest.raises(ValueError, match="same positive length"):
        runner.generate_prompt_schedule(
            [("W", "W")], [None], num_frames=4, diffusion_steps=1,
            seeds=[1, 2])
