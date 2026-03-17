import hashlib
import typing

import numpy as np
import torch
from jaxtyping import Float, Int

# Limited due to pre-calculated primes
_MAX_HALTON_DIMENSION = 1000


def create_halton_sequence(
    order: int,
    dim: int = 1,
    randomized: bool = True,
    seed: int = None,
    device=None,
    dtype=torch.float32,
) -> Float[torch.Tensor, "order dim"]:
    """Creates a 'dim' dimensional halton sequence

    Args:
        order (int): Defines the order (number of samples drawn) of the sequence.
        dim (int, optional): The dimensions of the sequence. Defaults to 1.
        randomized (bool, optional): Randomizes the sequence based on
            Owen 2017 - A randomized Halton algorithm in R. Defaults to True.
        seed (int, optional): If randomized, allow setting a seed. Defaults to None (No seed).
        device (Pytorch Device, optional): Creates the samples directly on the specified device.
        dtype (Torch Type): Specifies the data type to use. Must be a floating point one.
            Default float32.

    Returns (torch.tensor):
        Halton sequence with ``shape == (order, dim)``.
    """
    if dim < 1 or dim > _MAX_HALTON_DIMENSION:
        raise ValueError(
            "The halton sequence requires a dimension between 1 and {}. Supplied {}".format(
                _MAX_HALTON_DIMENSION, dim
            )
        )

    indices = (torch.arange(order, dtype=torch.float32, device=device) + 1).view(
        -1, 1, 1
    )
    radixes = torch.from_numpy(_PRIMES[0:dim]).to(device=device, dtype=dtype)

    max_size_by_axes = (
        torch.floor(torch.log(torch.max(indices)) / torch.log(radixes)) + 1
    )
    max_size = torch.max(max_size_by_axes)

    exponents_by_axes = (
        torch.arange(max_size, device=device).unsqueeze(0).repeat(dim, 1)
    )

    mask = exponents_by_axes < max_size_by_axes.unsqueeze(1).expand(
        -1, exponents_by_axes.shape[-1]
    )
    capped_exps = torch.where(
        mask, exponents_by_axes, torch.zeros_like(exponents_by_axes)
    )
    weights = radixes.unsqueeze(1).expand(-1, capped_exps.shape[-1]) ** capped_exps

    coeffs = torch.floor_divide(indices, weights)
    coeffs *= mask.to(coeffs.dtype)
    coeffs %= radixes.view(1, -1, 1)

    if not randomized:
        coeffs /= radixes.view(1, -1, 1)
        return (coeffs / weights.unsqueeze(0)).sum(-1)

    shuffle_seed, correction_seed = split_seed(seed, salt="HoloHaltonSequencePRNG")

    coeffs = _randomize(coeffs, radixes, device=device, seed=shuffle_seed)
    coeffs *= mask.to(coeffs.dtype)
    coeffs /= radixes.view(1, -1, 1)
    base_values = (coeffs / weights.unsqueeze(0)).sum(-1)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(correction_seed))

    zero_correction = torch.rand(
        (dim, 1), generator=generator, dtype=dtype, device=device
    )
    zero_correction /= (radixes**max_size_by_axes).view(-1, 1)

    return base_values + zero_correction.view(1, -1)


def _randomize(coeffs, radixes, seed: typing.Optional[int] = None, device=None):
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))

    initial_dtype = coeffs.dtype
    coeffs = coeffs.to(torch.int32)
    num_coeffs = coeffs.shape[-1]
    radixes = radixes.to(dtype=torch.int32).view(-1)

    perms = _get_permutations(num_coeffs, radixes, device=device, seed=seed).view(-1)

    radix_sum = radixes.sum()
    # Exclusive cumsum
    radix_offset = radixes.cumsum(0).roll(1, 0)
    radix_offset[0] = 0

    radix_offset = radix_offset.view(-1, 1)

    offsets = (
        radix_offset + torch.arange(num_coeffs, device=device).view(1, -1) * radix_sum
    )

    permuted_coeffs = torch.gather(
        perms, dim=0, index=(coeffs + offsets.unsqueeze(0)).view(-1)
    )
    return permuted_coeffs.to(initial_dtype).view(coeffs.shape)


def _get_permutations(
    num_results: int,
    dims: Int[torch.Tensor, "dim"],  # noqa: F821
    seed: typing.Optional[int] = None,
    device=None,
    dtype=torch.float32,
):
    seeds = split_seed(seed, n=dims.shape[0])

    def generate_one(dim, seed):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        rnd = torch.rand(
            (num_results, dim), generator=generator, dtype=dtype, device=device
        )
        return torch.argsort(rnd, dim=-1)

    return torch.cat([generate_one(dim, seed) for dim, seed in zip(dims, seeds)], -1)


def _primes_less_than(n):
    # From:
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """Returns sorted array of primes such that `2 <= prime < n`."""
    small_primes = np.array((2, 3, 5))
    if n <= 6:
        return small_primes[small_primes < n]
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool_)
    sieve[0] = False
    m = int(n**0.5) // 3 + 1
    for i in range(m):
        if not sieve[i]:
            continue
        k = 3 * i + 1 | 1
        sieve[k**2 // 3 :: 2 * k] = False
        sieve[(k**2 + 4 * k - 2 * k * (i & 1)) // 3 :: 2 * k] = False
    return np.r_[2, 3, 3 * np.nonzero(sieve)[0] + 1 | 1]


_PRIMES = _primes_less_than(7919 + 1)
assert len(_PRIMES) == _MAX_HALTON_DIMENSION


def split_seed(seed: int, n: int = 2, salt: typing.Optional[str] = None) -> np.ndarray:
    seed = sanitize_seed(seed, salt=salt)

    iinfo = np.iinfo(np.int32)

    prng = np.random.RandomState(seed)
    seeds = prng.randint(low=0, high=iinfo.max, size=(n,), dtype=np.int32)

    return seeds


def sanitize_seed(seed: int, salt: typing.Optional[str] = None) -> int:
    if salt is not None:
        if seed is not None:
            seed = int(
                hashlib.sha512(str((seed, salt)).encode("utf-8")).hexdigest(), 16
            ) % (2**31 - 1)
        salt = None

    prng = np.random.RandomState(seed)

    iinfo = np.iinfo(np.int32)
    seed = prng.randint(low=0, high=iinfo.max, dtype=np.int32)

    if salt is None:
        salt = int(hashlib.sha512(str(salt).encode("utf-8")).hexdigest(), 16) % (
            2**31 - 1
        )
        salt = np.int32(salt)
        seed = np.bitwise_xor(seed, salt)

    return seed
