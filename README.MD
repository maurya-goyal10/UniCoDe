# UniCoDe: Unified Control for Inference-Time Guidance of Denoising Diffusion Models

[arXiv Link] (http://arxiv.org/abs/2512.12339) 

**Maurya Goyal¹, Anuj Singh¹², Hadi Jamali-Rad¹²**
\
¹Delft University of Technology, The Netherlands
²Shell Global Solutions International B.V., Amsterdam, The Netherlands

## Installation

This code is written in Python 3.8.5 and requires the packages listed in [cluster_requirements.txt](./cluster_requirements.txt).

To run the code, set up a virtual environment using conda:

```bash
# Navigate to the project directory
cd <path-to-cloned-directory>

# Create the conda environment from the requirements file
conda create --name <env> --file ./cluster_requirements.txt

# Activate the environment
conda activate <env>
```
## Running Experiments

This repository follows a **config-driven workflow** to streamline local and cluster-based experimentation.

---

### 1. Generate Configurations
Configure your experimental settings by modifying `_SCORES` and `_METHODS` in [create_config.py](./BoN/configs_split/create_config.py).

```bash
python ./BoN/configs_split/create_config.py
```
Result: YAML files are saved to `./configs_split/<method>/`
Purpose: Each file acts as a standalone specification for a single experiment

---

### 2. Execute Inference
Run the generation process locally or prepare scripts for a compute cluster.

**Local Run:**
```bash
python inference_split.py --config ../configs_split/<method>/<config_name>.yaml
```

**Cluster Run:**
Generate batch job scripts using [create_jobs.py](./BoN/jobs_split/create_jobs.py).

**Automated Workflow:**
* **Setup:** Loads parameters from the selected YAML configuration.
* **Generation:** Saves images to `outputs/<method>/<config_name>/`.
* **Scoring:** Writes computed reward values to `rewards.json`.

---

### 3. Evaluation Metrics
Post-inference, calculate aggregate performance using these specialized scripts:

| Metric | Script Path |
| :--- | :--- |
| **Aggregated Rewards, FID, & CMMD** | [compute_scores.py](./BoN/compute_scores.py) |
| **ClipScore** | [compute_scores.py](./clipscore/compute_scores.py) |

---

## Overview

**UniCoDe** is a universal inference-time algorithm designed to align diffusion model outputs with downstream objectives by unifying **sampling-based exploration** and **gradient-guided optimization**. While traditional methods often struggle with sampling inefficiency or poor prior adherence, UniCoDe integrates local gradient signals directly into the denoising steps to steer trajectories toward high-reward regions. This overcomes the exponential search costs associated with complex rewards, allowing for more efficient generation. Ultimately, UniCoDe enables a superior trade-off between reward alignment and adherence to the diffusion prior, remaining competitive with state-of-the-art baselines across Text-to-Image (T2I) and (T+I)2I tasks.

<div align="center">
    <img src="assets/teaser.png" alt="UniCoDe outputs" width="80%"/>
    <br>
    <em><strong>Visual Adaptability:</strong> UniCoDe effectively aligns T2I and (T+I)2I tasks with complex human-preference rewards like Pickscore and Aesthetic Scorer while maintaining high image fidelity.</em>
</div>

## The UniCoDe Framework
UniCoDe merges the exploratory nature of sampling with directed gradient steering to enable precise control over diverse guidance scenarios.

<div align="center">
    <img src="./assets/schematic_diagram.png" alt="UniCoDe framework schematic" width="80%"/>
    <br>
    <em><strong>Methodology Comparison:</strong> While <strong>BoN</strong> selects the best sample after full denoising and <strong>CoDe</strong> uses greedy blockwise selection, <strong>UniCoDe</strong> enhances efficiency by integrating blockwise gradient guidance, scheduled particle sizes, and multinomial sampling.</em>
</div>
<br>


**Core Mechanism:** The framework overcomes the sample inefficiency problem, where the gap between the reward distribution and diffusion prior grows exponentially, by adding a local gradient "push" to the denoising trajectory. This steers samples toward high-reward regions, significantly reducing the sample count needed for alignment. In this example can see that UniCoDe optimizes the balance between **reward alignment** (Pickscore) and **unconditional prior divergence** (CMMD).

<div align="center">
    <img src="./assets/rew_vs_div.png" alt="Reward vs Divergence plot" width="60%"/>
    <br>
    <em><strong>Pareto Frontier:</strong> Compared to standalone sampling or gradient-guided baselines, UniCoDe achieves higher rewards with significantly lower distribution shift (CMMD).</em>
</div>

## Parameter Guidelines
The trade-off between reward, fidelity, and compute is governed by three key variables:

* **$N$ (Exploration):** Enhances reward-alignment through broader search but increases computational cost and prior divergence.
* **$B_g$ (Gradient Intensity):** Lowering $B_g$ boosts rewards via a stronger gradient push, but excessively low values may lead to reward hacking.
* **$B_s$ (Sampling Frequency):** Lower $B_s$ values represent more aggressive sampling, improving alignment at the cost of higher divergence.

---


## References

#### Our approach
- UniCoDe

#### Baselines
##### Sampling-based
- CoDe [`Paper`](https://openreview.net/forum?id=DqPCWMiMU0) [`Code`](Universal-Guided-Diffusion/)

##### Guidance-based
- Universal Guidance [`Paper`](https://openreview.net/forum?id=pzpWBbnwiJ) [`Code`](Universal-Guided-Diffusion/)
- Diffusion Posterior Sampling [`Paper`](https://openreview.net/forum?id=OnD9zGAGT0k) [`Code`](BoN/)
- FreeDoM [`Paper`](https://openreview.net/forum?id=kzAMGYIoHu) [`Code`](FreeDoM/)
- MPGD [`Paper`](https://openreview.net/forum?id=o3BxOLoxm1) [`Code`](mpgd_pytorch/)

##### Learning-based

- AlignProp [`Paper`](https://openreview.net/forum?id=Vaf4sIrRUC) [`Code`](AlignProp/)

#### Evaluation Metrics

- Frechet Inception Distance (FID) [`Paper`](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html) [`Code`](pytorch-fid/)
- Clip-based Maximum Mean Discrepancy (CMMD) [`Paper`](https://arxiv.org/abs/2401.09603) [`Code`](cmmd-pytorch/)
- T-CLIP [`Code`](clipscore/)
- IGram [`CoDe`](BoN/src_sd/scorers/stylescorer.py)

#### Reward Models
- Pickscore [`Paper`](https://openreview.net/forum?id=G5RwHpBUv0) [`Code`](BoN/src_sd/scorers/pickscore_scorer.py)
- Aesthetic [`Github`](https://github.com/LAION-AI/aesthetic-predictor) [`Code`](BoN/src_sd/scorers/aesthetic_scorer.py)
- Compressibility [`Paper`]() [`Code`](BoN/src_sd/scorers/compressibilityscorer.py)
