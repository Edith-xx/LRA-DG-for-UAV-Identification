# Low-Rank-Aware Domain Generalization for Channel-Robust Multi-Granularity UAV Identification


> **Status:** This manuscript has been submitted to the *IEEE Internet of Things Journal* and is currently under review.
## Overview

This work studies channel-robust radio-frequency fingerprint identification for both UAV category recognition and individual-device identification. The problem is formulated as single-source domain generalization: only line-of-sight (LOS) data are available during training, while non-line-of-sight (NLOS) data are used only for testing.

The proposed framework combines sample-level low-rank reconstruction and feature-level domain-invariant learning. Under the LOS-to-NLOS setting, it achieves **94.01% category accuracy** and **90.85% individual accuracy**.

## Main Contributions

1. **Single-source domain generalization formulation.**  
   We formulate cross-channel multi-granularity UAV identification as an SDG problem and provide a theoretical analysis showing that device-dependent fingerprint variations approximately lie in a low-dimensional subspace.

2. **SVD-based low-rank spectrogram reconstruction.**  
   Truncated singular value decomposition retains the leading singular modes to preserve transmitter-intrinsic fingerprint structures while suppressing channel- and noise-sensitive components.

3. **Feature-level domain expansion and dual-consistency learning.**  
   Feature statistics are perturbed through Beta interpolation and adaptive instance normalization to generate label-preserving domains. Device-conditional covariance alignment and supervised contrastive learning are then used to learn representations that are both domain-invariant and device-discriminative.

## Datasets

The main LOS-to-NLOS experiments use the **DroneRFb-DIR** dataset, which contains six UAV categories and three devices per category, for a total of 18 UAV identities.

- Dataset page: https://www.scidb.cn/en/detail?dataSetId=84cf9101e739402784b1396783881202
- Dataset paper: https://doi.org/10.11999/JEIT240804
- **Citation:
  
J. Ren, N. Yu, C. Zhou, Z. Shi and J. Chen, ``DroneRFb-DIR: An RF Signal Dataset for Non-cooperative Drone Individual Identification,'' \emph{J. Electron. Inf. Technol.}, vol. 47, no. 3, pp. 573--581, 2025.

### Hovering UAVs RF Fingerprinting Dataset

The cross-distance experiments use signals collected at 6, 9, 12, and 15 ft. The model is trained on 6-ft data and evaluated on the unseen 9-, 12-, and 15-ft domains.

- Dataset page: https://genesys-lab.org/hovering-uavs

Please follow the licenses and usage policies specified by the original dataset providers.

## Citation

```bibtex
@unpublished{LRA_SDG,
  title  = {Low-Rank-Aware Domain Generalization for Channel-Robust Multi-Granularity UAV Identification},
  author = {Cai, Zhenxin and Wang, Yu and Li, Jingyuan and Sha, Jin},
  note   = {Manuscript submitted to IEEE Internet of Things Journal}
}
```

The citation information will be updated after the paper is officially accepted and published.
