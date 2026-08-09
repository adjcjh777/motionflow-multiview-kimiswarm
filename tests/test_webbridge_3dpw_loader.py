import pathlib
import pytest
import torch

from motionflow_mv.data.webbridge_mixed_dataset import WebBridgeCanonical17Dataset


@pytest.mark.parametrize("return_view_mask", [False, True])
def test_webbridge_3dpw_canonical17_dataset(return_view_mask: bool) -> None:
    """Verify 3DPW canonical .npz loads and maps to 17 joints / 14 padded views."""
    npz_path = pathlib.Path("data/webbridge/3dpw/converted/train/courtyard_arguing_00_pseudo.npz")
    if not npz_path.exists():
        pytest.skip("3DPW canonical .npz not available on this machine")

    dataset = WebBridgeCanonical17Dataset(
        str(npz_path),
        dataset_name="3dpw",
        clip_len=9,
        n_samples=2,
        return_view_mask=return_view_mask,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    if return_view_mask:
        x, y, K, R, t, dataset_id, view_mask = sample
        assert view_mask.shape == (14,)
        assert view_mask.sum() > 0
    else:
        x, y, K, R, t, dataset_id = sample

    assert x.shape == (9, 14, 17, 3)
    assert y.shape == (9, 17, 3)
    assert K.shape == (14, 3, 3)
    assert R.shape == (14, 3, 3)
    assert t.shape == (14, 3)
    assert dataset_id == 5
