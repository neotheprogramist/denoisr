import torch

from denoisr.training.dataset import ChessDataset


class TestChessDataset:
    def test_len(self) -> None:
        boards = torch.randn(100, 12, 8, 8)
        policies = torch.randn(100, 64, 64)
        values = torch.randn(100, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        assert len(ds) == 100

    def test_getitem_shapes(self) -> None:
        boards = torch.randn(10, 12, 8, 8)
        policies = torch.randn(10, 64, 64)
        values = torch.randn(10, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        board, policy, value = ds[0]
        assert board.shape == (12, 8, 8)
        assert policy.shape == (64, 64)
        assert value.shape == (3,)

    def test_no_augment_returns_original(self) -> None:
        boards = torch.randn(10, 12, 8, 8)
        policies = torch.randn(10, 64, 64)
        values = torch.randn(10, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        board, policy, value = ds[3]
        assert torch.equal(board, boards[3])
        assert torch.equal(policy, policies[3])
        assert torch.equal(value, values[3])

    def test_augment_flips_some_examples(self) -> None:
        """With augmentation, at least some examples should differ from originals."""
        torch.manual_seed(0)
        boards = torch.randn(100, 12, 8, 8)
        policies = torch.randn(100, 64, 64)
        values = torch.randn(100, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=True)
        flipped = 0
        for i in range(100):
            board, _, _ = ds[i]
            if not torch.equal(board, boards[i]):
                flipped += 1
        # With 50% flip probability, expect ~50 flipped (allow wide margin)
        assert 20 < flipped < 80

    def test_dataloader_integration(self) -> None:
        boards = torch.randn(32, 12, 8, 8)
        policies = torch.randn(32, 64, 64)
        values = torch.randn(32, 3)
        ds = ChessDataset(boards, policies, values, num_planes=12, augment=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        batch_boards, batch_policies, batch_values = next(iter(loader))
        assert batch_boards.shape == (8, 12, 8, 8)
        assert batch_policies.shape == (8, 64, 64)
        assert batch_values.shape == (8, 3)
