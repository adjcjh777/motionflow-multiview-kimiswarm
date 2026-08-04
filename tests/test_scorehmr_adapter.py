"""Tests for ScoreHMR -> HumanMotionIR adapter."""

import numpy as np
import torch

from motionflow_mv.ir.human_motion_ir import HumanMotionIR
from motionflow_mv.ir.scorehmr_adapter import scorehmr_result_to_ir, ir_to_scorehmr_params


def test_scorehmr_result_to_ir():
    T = 8
    pred_smpl_params = {
        "global_orient": torch.randn(T, 1, 3, 3),
        "body_pose": torch.randn(T, 23, 3, 3),
        "betas": torch.randn(T, 10),
    }
    ir = scorehmr_result_to_ir(pred_smpl_params, sequence_id="cam_0", fps=30.0)
    assert isinstance(ir, HumanMotionIR)
    assert ir.sequence_id == "cam_0"
    assert ir.fps == 30.0
    assert ir.human_model == "smpl"
    assert "transl" in ir.pose
    assert ir.pose["global_orient"].shape == (T, 1, 3, 3)
    assert ir.pose["body_pose"].shape == (T, 23, 3, 3)
    assert ir.provenance["camera_relative"] is True


def test_ir_to_scorehmr_params():
    pred_smpl_params = {
        "global_orient": torch.randn(8, 1, 3, 3),
        "body_pose": torch.randn(8, 23, 3, 3),
        "betas": torch.randn(8, 10),
    }
    ir = scorehmr_result_to_ir(pred_smpl_params)
    params = ir_to_scorehmr_params(ir)
    assert "global_orient" in params
    assert "body_pose" in params
    assert "betas" in params
    assert "transl" in params


def test_scorehmr_single_frame():
    pred_smpl_params = {
        "global_orient": torch.randn(1, 3, 3),
        "body_pose": torch.randn(1, 23, 3, 3),
        "betas": torch.randn(10),
    }
    ir = scorehmr_result_to_ir(pred_smpl_params)
    assert ir.pose["global_orient"].shape == (1, 1, 3, 3)
    assert ir.pose["body_pose"].shape == (1, 23, 3, 3)
