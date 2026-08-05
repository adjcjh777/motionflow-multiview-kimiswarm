# WebBridge Cross-Dataset Smoke Benchmark

| dataset | mpjpe_mm | pa_mpjpe_mm | pck_50 | pck_100 | pck_150 | pck_auc |
|---|---:|---:|---:|---:|---:|---:|
| mpi_s2_seq1_v14 | 14.7133 | 13.8615 | 0.9972 | 1.0000 | 1.0000 | 0.9019 |
| mpi_s3_seq1_v14 | 14.6957 | 11.4110 | 0.9984 | 1.0000 | 1.0000 | 0.9020 |
| mpi_s1_seq1_v4 | 27.9476 | 19.1000 | 0.8880 | 0.9810 | 1.0000 | 0.8137 |

- The v14 (14-view) sequences achieve ~14.7 mm MPJPE, consistent with the S2 validation.
- The v4 (4-view) sequence degrades to 27.95 mm, confirming the model trained with 14 views struggles when only 4 views are available without adaptation.
