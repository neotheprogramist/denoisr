from denoisr.scripts.generate_data import _estimate_chunk_buffer_gib


def test_estimate_chunk_buffer_gib() -> None:
    gib = 1024 * 1024 * 1024
    num_examples = 1000
    num_planes = 122
    expected = num_examples * (((num_planes * 8 * 8) + (64 * 64) + 3) * 4) / gib
    assert _estimate_chunk_buffer_gib(num_examples, num_planes) == expected
