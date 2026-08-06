"""CPU smoke test for the extended robustness evaluation matrix."""


def test_extended_robustness_matrix_smoke():
    """The prototype script should complete its built-in CPU smoke path."""
    import experiments.prototypes.run_extended_robustness_matrix as erm

    results = erm.smoke_test()

    assert len(results) > 0, "No conditions were evaluated"
    assert "clean" in results, "Missing clean baseline condition"
    assert "noise_1.0px_joint_occlusion_20_view_dropout_30" in results, "Missing three-axis combo condition"

    for name, summary in results.items():
        assert "mpjpe" in summary, f"{name} missing mpjpe metric"
        assert "pa_mpjpe" in summary, f"{name} missing pa_mpjpe metric"
        assert "pck_auc" in summary, f"{name} missing PCK-AUC metric"
