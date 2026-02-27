from denoisr.scripts.train_phase1 import _evaluate_stability_guard


def test_stability_guard_triggers_on_collapse_with_overflow_signal() -> None:
    decision = _evaluate_stability_guard(
        enabled=True,
        epoch_num=8,
        best_top1=0.40,
        current_top1=0.08,
        overflow_frac=0.002,
        grad_peak=20.0,
        backoff_count=0,
        cooldown_until_epoch=0,
        min_epoch=4,
        drop_ratio=0.75,
        min_drop=0.10,
        overflow_frac_threshold=5e-4,
        grad_peak_threshold=100.0,
        max_backoffs=4,
    )
    assert decision.trigger is True
    assert "top1 dropped" in decision.reason
    assert "overflow_frac" in decision.reason


def test_stability_guard_triggers_on_collapse_with_grad_spike_signal() -> None:
    decision = _evaluate_stability_guard(
        enabled=True,
        epoch_num=9,
        best_top1=0.42,
        current_top1=0.07,
        overflow_frac=0.0,
        grad_peak=250.0,
        backoff_count=0,
        cooldown_until_epoch=0,
        min_epoch=4,
        drop_ratio=0.75,
        min_drop=0.10,
        overflow_frac_threshold=5e-4,
        grad_peak_threshold=100.0,
        max_backoffs=4,
    )
    assert decision.trigger is True
    assert "grad_peak" in decision.reason


def test_stability_guard_ignores_noncatastrophic_dips() -> None:
    decision = _evaluate_stability_guard(
        enabled=True,
        epoch_num=6,
        best_top1=0.40,
        current_top1=0.35,
        overflow_frac=0.002,
        grad_peak=300.0,
        backoff_count=0,
        cooldown_until_epoch=0,
        min_epoch=4,
        drop_ratio=0.75,
        min_drop=0.10,
        overflow_frac_threshold=5e-4,
        grad_peak_threshold=100.0,
        max_backoffs=4,
    )
    assert decision.trigger is False


def test_stability_guard_respects_epoch_cooldown_and_limits() -> None:
    cooldown_decision = _evaluate_stability_guard(
        enabled=True,
        epoch_num=9,
        best_top1=0.40,
        current_top1=0.08,
        overflow_frac=0.002,
        grad_peak=120.0,
        backoff_count=0,
        cooldown_until_epoch=9,
        min_epoch=4,
        drop_ratio=0.75,
        min_drop=0.10,
        overflow_frac_threshold=5e-4,
        grad_peak_threshold=100.0,
        max_backoffs=4,
    )
    assert cooldown_decision.trigger is False

    maxed_out_decision = _evaluate_stability_guard(
        enabled=True,
        epoch_num=10,
        best_top1=0.40,
        current_top1=0.08,
        overflow_frac=0.002,
        grad_peak=120.0,
        backoff_count=4,
        cooldown_until_epoch=0,
        min_epoch=4,
        drop_ratio=0.75,
        min_drop=0.10,
        overflow_frac_threshold=5e-4,
        grad_peak_threshold=100.0,
        max_backoffs=4,
    )
    assert maxed_out_decision.trigger is False
