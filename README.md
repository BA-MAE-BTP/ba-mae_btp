# BA-MAE
## Block-Aware Masked Autoencoder for Tail-Stratified Network Intrusion Detection

> A self-supervised representation learning framework for robust network intrusion detection through dependency-aware masking, tail-aware learning, and robust reconstruction.

---

## 1. Overview

Network Intrusion Detection Systems (NIDS) operate on high-dimensional network traffic in which many features are statistically dependent, highly skewed, and heavy-tailed. Conventional representation learning approaches may therefore face two important challenges:

1. **Feature dependency and reconstruction shortcuts:**  
   When individual features are randomly masked, a masked feature may have a strongly dependent feature remaining visible. The model can then reconstruct the missing value using this redundant information rather than learning a meaningful representation of the underlying traffic behavior.

2. **Extreme and tail traffic:**  
   Network traffic distributions are highly non-uniform. A relatively small proportion of flows can occupy the extreme tail of the flow-size distribution. These flows may represent important traffic patterns and can also produce disproportionately large reconstruction errors.

BA-MAE (**Block-Aware Masked Autoencoder**) investigates whether explicitly incorporating these properties into masked autoencoder pretraining can produce more robust network-traffic representations for downstream intrusion detection.

The framework combines:

- Train-only statistical preprocessing
- Mutual Information (MI)
- Spearman correlation
- Unified dependency modeling
- Data-driven alpha optimization
- Hierarchical dependency clustering
- Dependency-based feature blocks
- Whole-block masking
- Tail-aware masking
- Huber reconstruction loss
- Self-supervised MAE pretraining
- Latent representation learning
- Local Outlier Factor (LOF) based anomaly detection
- Overall and tail-stratified evaluation

---

# 2. Research Question

The central research question is:

> **Can dependency-aware and tail-adaptive masked autoencoder pretraining learn robust network-traffic representations and improve downstream network intrusion detection, particularly for extreme and tail traffic?**

The project therefore does not treat masking merely as a data-augmentation operation. Masking is considered a central component of the representation-learning objective.

---

# 3. Research Hypothesis

The working hypothesis is that:

> **Masking statistically dependent features together can reduce reconstruction shortcuts, while tail-adaptive masking and robust reconstruction can improve the treatment of extreme network traffic.**

The expected result is a latent representation that captures broader traffic characteristics rather than relying primarily on direct redundancy between individual features.

---

# 4. Dataset

The project uses **NF-UQ-NIDS-v2**, a large-scale NetFlow-based network intrusion detection dataset containing approximately 76 million network flows.

The original records contain network-flow features together with attack labels and dataset-source information.

### Dataset partition

The finalized dataset is divided into:

| Partition | Approx. Records | Proportion |
|---|---:|---:|
| Training | 53,192,767 | 70% |
| Validation | 11,398,001 | 15% |
| Testing | 11,397,208 | 15% |
| **Total** | **75,987,976** | **100%** |

The distributions of labels and attack categories were checked across the partitions to ensure that the resulting splits remained highly consistent.

---

# 5. Feature Selection

The original dataset contains 43 traffic-related features followed by:

- `Label`
- `Attack`
- `Dataset`

Four identity/address-related fields were excluded from model input:

- `IPV4_SRC_ADDR`
- `IPV4_DST_ADDR`
- `L4_SRC_PORT`
- `L4_DST_PORT`


The following three fields are retained strictly as metadata:

```text
Label
Attack
Dataset
