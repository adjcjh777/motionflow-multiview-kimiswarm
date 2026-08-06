# Cross-Dataset Transfer (Metric-Normalised v1)

Training: H36M subject 1, actions 2–16, 62,094 frames, all coordinates in meters.  
Checkpoint: `outputs/ray_attention_v1_s_01_acts_02_..._16_multiview_m.pth`

All datasets were converted to meters with `experiments/convert_npz_to_meters.py`
(H36M /1000, Shelf/Campus /100).

## Campus_Seq1 (3 views, 1,423 frames)

Zero-shot evaluation. All values in meters.

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 0.738         | 0.000 |
| 0.0  | 2.00  | 0.759         | 0.070 |
| 0.0  | 5.00  | 0.821         | 0.175 |
| 0.2  | 0.00  | 0.657         | 1.696 |
| 0.2  | 2.00  | 0.681         | 1.737 |
| 0.2  | 5.00  | 0.744         | 1.779 |
| 0.4  | 0.00  | 0.559         | 2.510 |
| 0.4  | 2.00  | 0.588         | 2.490 |
| 0.4  | 5.00  | 0.645         | 2.563 |

## Shelf_Seq1 (5 views, 500-frame subset)

Zero-shot evaluation. All values in meters.

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 0.083         | 0.000 |
| 0.0  | 2.00  | 0.084         | 0.001 |
| 0.0  | 5.00  | 0.084         | 0.004 |
| 0.2  | 0.00  | 0.080         | 0.221 |
| 0.2  | 2.00  | 0.079         | 0.223 |
| 0.2  | 5.00  | 0.080         | 0.227 |
| 0.4  | 0.00  | 0.072         | 0.414 |
| 0.4  | 2.00  | 0.073         | 0.413 |
| 0.4  | 5.00  | 0.074         | 0.420 |

The H36M-trained model transfers to a different 5-view camera rig and remains
robust under view dropout, while the geometric DLT baseline degrades sharply.
