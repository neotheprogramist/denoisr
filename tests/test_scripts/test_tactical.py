import chess

from denoisr.scripts.generate_data import _is_tactical


class TestIsTactical:
    def test_starting_position_not_tactical(self) -> None:
        assert not _is_tactical(chess.Board())

    def test_endgame_is_tactical(self) -> None:
        # King + Rook vs King — only 3 pieces
        board = chess.Board("8/8/8/8/8/8/4R3/4K2k w - - 0 1")
        assert _is_tactical(board)

    def test_hanging_piece_is_tactical(self) -> None:
        # White knight on e5 attacked by black pawn on d6, not defended
        board = chess.Board("rnbqkb1r/ppp2ppp/3p4/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 3")
        # The knight on e5 is attacked by d6 pawn, check if defended
        assert board.is_attacked_by(chess.BLACK, chess.E5)
        # If not defended and knight is there, it should be tactical
        if not board.attackers(chess.WHITE, chess.E5):
            assert _is_tactical(board)

    def test_normal_middlegame_not_tactical(self) -> None:
        # Standard position after 1.e4 e5 2.Nf3 Nc6 — no hanging pieces
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        board.push_uci("g1f3")
        board.push_uci("b8c6")
        assert not _is_tactical(board)
