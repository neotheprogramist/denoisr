# src/denoisr/gui/app.py
"""Main Denoisr Chess GUI application."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING

import chess

from denoisr.gui.board_widget import BoardWidget
from denoisr.gui.types import EngineConfig, TimeControl
from denoisr.gui.uci_engine import UCIEngine

if TYPE_CHECKING:
    pass


class DenoisrApp:
    """Main GUI application with Play and Match modes."""

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("Denoisr Chess")
        self._root.resizable(False, False)

        self._board = chess.Board()
        self._engine: UCIEngine | None = None
        self._engine_thread: threading.Thread | None = None
        self._move_queue: queue.Queue[chess.Move | str] = queue.Queue()
        self._moves: list[str] = []
        self._human_color = chess.WHITE

        self._build_ui()
        self._poll_queue()

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _build_ui(self) -> None:
        # Main frame
        main = ttk.Frame(self._root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        # Left: board
        self._board_widget = BoardWidget(main, square_size=60)
        self._board_widget.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self._board_widget.set_on_move(self._on_human_move)

        # Right: controls
        ctrl = ttk.LabelFrame(main, text="Controls", padding=8)
        ctrl.grid(row=0, column=1, sticky="new")

        # Mode selector
        self._mode_var = tk.StringVar(value="play")
        ttk.Radiobutton(
            ctrl, text="Play", variable=self._mode_var, value="play",
            command=self._on_mode_change,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            ctrl, text="Match", variable=self._mode_var, value="match",
            command=self._on_mode_change,
        ).grid(row=0, column=1, sticky="w")

        # Checkpoint
        ttk.Label(ctrl, text="Checkpoint:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self._ckpt_var = tk.StringVar()
        ckpt_frame = ttk.Frame(ctrl)
        ckpt_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Entry(ckpt_frame, textvariable=self._ckpt_var, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(ckpt_frame, text="Browse", command=self._browse_ckpt).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Engine mode
        ttk.Label(ctrl, text="Engine mode:").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self._engine_mode_var = tk.StringVar(value="single")
        ttk.Combobox(
            ctrl,
            textvariable=self._engine_mode_var,
            values=["single", "diffusion"],
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))

        # Human color
        ttk.Label(ctrl, text="Play as:").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self._color_var = tk.StringVar(value="white")
        ttk.Combobox(
            ctrl,
            textvariable=self._color_var,
            values=["white", "black"],
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", pady=(8, 0))

        # Buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(
            btn_frame, text="New Game", command=self._new_game
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="Stop", command=self._stop_game
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="Flip", command=self._board_widget.flip
        ).pack(side=tk.LEFT, padx=2)

        # Bottom: status
        status_frame = ttk.LabelFrame(main, text="Game", padding=8)
        status_frame.grid(
            row=1, column=1, sticky="nsew", pady=(8, 0)
        )

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status_frame, textvariable=self._status_var, wraplength=200
        ).pack(anchor="w")

        ttk.Label(status_frame, text="Moves:").pack(
            anchor="w", pady=(8, 0)
        )
        self._moves_text = tk.Text(
            status_frame, width=28, height=12, wrap=tk.WORD, state=tk.DISABLED
        )
        self._moves_text.pack(fill=tk.BOTH, expand=True)

    def _browse_ckpt(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self._ckpt_var.set(path)

    def _on_mode_change(self) -> None:
        # Mode switching handled in future tasks
        pass

    def _new_game(self) -> None:
        self._stop_game()
        self._board = chess.Board()
        self._moves = []
        self._human_color = (
            chess.WHITE if self._color_var.get() == "white" else chess.BLACK
        )

        self._board_widget.set_board(self._board)
        self._update_moves_text()
        self._status_var.set("White to move")

        ckpt = self._ckpt_var.get().strip()
        if not ckpt:
            self._status_var.set("Set a checkpoint first")
            return

        # Build engine config
        mode = self._engine_mode_var.get()
        args = ["run", "denoisr-play", "--checkpoint", ckpt, "--mode", mode]
        config = EngineConfig(command="uv", args=tuple(args), name="Denoisr")

        try:
            self._engine = UCIEngine(config)
            self._engine.start(timeout=30.0)
        except (TimeoutError, OSError) as e:
            self._status_var.set(f"Engine failed: {e}")
            self._engine = None
            return

        # If human is black, engine moves first
        if self._human_color == chess.BLACK:
            self._board_widget.set_interactive(False)
            self._request_engine_move()
        else:
            self._board_widget.set_interactive(True)

    def _stop_game(self) -> None:
        self._board_widget.set_interactive(False)
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def _on_human_move(self, move: chess.Move) -> None:
        self._board.push(move)
        self._moves.append(move.uci())
        self._board_widget.set_board(self._board)
        self._board_widget.highlight_last_move(move)
        self._board_widget.set_interactive(False)
        self._update_moves_text()

        if self._board.is_game_over():
            self._show_game_over()
            return

        self._status_var.set("Engine thinking...")
        self._request_engine_move()

    def _request_engine_move(self) -> None:
        if self._engine is None:
            return
        tc = TimeControl(base_seconds=60.0, increment=0.0)

        def _think() -> None:
            assert self._engine is not None
            try:
                self._engine.set_position(fen=None, moves=list(self._moves))
                uci = self._engine.go(
                    time_control=tc, wtime_ms=60000, btime_ms=60000
                )
                self._move_queue.put(chess.Move.from_uci(uci))
            except Exception as e:
                self._move_queue.put(f"error:{e}")

        self._engine_thread = threading.Thread(target=_think, daemon=True)
        self._engine_thread.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._move_queue.get_nowait()
                if isinstance(item, chess.Move):
                    self._apply_engine_move(item)
                elif isinstance(item, str) and item.startswith("error:"):
                    self._status_var.set(item)
        except queue.Empty:
            pass
        self._root.after(50, self._poll_queue)

    def _apply_engine_move(self, move: chess.Move) -> None:
        self._board.push(move)
        self._moves.append(move.uci())
        self._board_widget.set_board(self._board)
        self._board_widget.highlight_last_move(move)
        self._update_moves_text()

        if self._board.is_game_over():
            self._show_game_over()
            return

        turn = "White" if self._board.turn == chess.WHITE else "Black"
        self._status_var.set(f"{turn} to move")
        self._board_widget.set_interactive(True)

    def _show_game_over(self) -> None:
        result = self._board.result()
        self._status_var.set(f"Game over: {result}")
        self._board_widget.set_interactive(False)

    def _update_moves_text(self) -> None:
        self._moves_text.config(state=tk.NORMAL)
        self._moves_text.delete("1.0", tk.END)
        # Format as numbered move pairs
        text_parts: list[str] = []
        for i in range(0, len(self._moves), 2):
            num = i // 2 + 1
            white_move = self._moves[i]
            if i + 1 < len(self._moves):
                black_move = self._moves[i + 1]
                text_parts.append(f"{num}. {white_move} {black_move}")
            else:
                text_parts.append(f"{num}. {white_move}")
        self._moves_text.insert("1.0", " ".join(text_parts))
        self._moves_text.config(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._stop_game()
        self._root.destroy()
