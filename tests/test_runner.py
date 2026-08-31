import torch

from scene2motion.runner import _per_sample_noise


def draws(seeds):
    with _per_sample_noise(seeds, "cpu"):
        return torch.randn((len(seeds), 5)), torch.randn((len(seeds), 5))


def test_per_sample_noise_reproducible_but_advances_between_windows():
    first_a, second_a = draws([11, 29])
    first_b, second_b = draws([11, 29])

    torch.testing.assert_close(first_a, first_b)
    torch.testing.assert_close(second_a, second_b)
    assert not torch.equal(first_a, second_a)


def test_per_sample_noise_is_independent_of_batch_neighbors():
    together_first, together_second = draws([11, 29])
    alone_first, alone_second = draws([29])

    torch.testing.assert_close(together_first[1], alone_first[0])
    torch.testing.assert_close(together_second[1], alone_second[0])
