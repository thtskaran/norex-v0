
# Project Norex: Latent-Space Diffusion on Frozen LLMs
## Experimental Investigation & Failure Analysis

**Author:** Karan Prasad (@thtskaran) | [Obvix Labs](https://obvix.io)  
**Status:** Research Prototype / Proof-of-Concept (Unstable)  
**Hardware:** Google Colab A100 GPU  
**Date:** December 2025

***

## Abstract

Project Norex explores latent-space diffusion as a mechanism for guiding text generation in frozen large language models (LLMs) without fine-tuning. The approach involves: (1) extracting hidden-state representations from Wikipedia text using a frozen 7B-parameter LLM (Mistral-7B-Instruct-v0.2), (2) compressing these representations into a structured latent space via a Variational Autoencoder (VAE), (3) learning a generative distribution over this latent space using flow matching, and (4) steering LLM generation by injecting sampled latent vectors as logit biases. While the end-to-end pipeline executes successfully, generation quality remains highly unstable and sensitive to hyperparameters—demonstrating both the feasibility and fragility of this frontier research direction.[1][2][3][4][5]

***

## Motivation & Context

Recent work on latent-space steering and flow-based generative models suggests that continuous latent representations can guide neural network behavior without modifying weights. This project investigates whether:[4][6][7][8][9][10]

- **Latent vectors** derived from VAE compression of LLM hidden states can capture semantic "abstractions" or "plans"
- **Flow matching models** trained on these latents can sample coherent, diverse latent codes[11][4]
- **Logit-space injection** of decoded latent vectors can steer frozen LLM generation toward desired outputs

This represents an attempt at **training-free, latent-guided generation**—an alternative to prompt engineering, fine-tuning, or reinforcement learning from human feedback (RLHF).

***

## Technical Pipeline

### 1. Data Collection & Hidden-State Extraction

```python
# Dataset: Wikipedia English (2023-11-01 snapshot)
wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
```

- **Source:** HuggingFace `wikimedia/wikipedia` (November 2023 English dump)
- **Preprocessing:** Filtered texts with length > 80 characters
- **Sample Size:** 300,000 Wikipedia articles
- **Tokenization:** Maximum sequence length of 256 tokens
- **Hidden-State Extraction:**
  - Model: `mistralai/Mistral-7B-Instruct-v0.2` (frozen, bfloat16 precision)
  - Hidden size: 4096 dimensions
  - Pooling: Mean-pooling over non-padding tokens from the final layer
  - Storage: Float16 precision on CPU to conserve memory
  - Batch size: 32 (encoding phase)

### 2. Variational Autoencoder (VAE) Architecture

The VAE compresses 4096-dimensional hidden states into 1024-dimensional latent vectors.[2][3]

**Architecture:**
```python
class LatentVAE(nn.Module):
    Encoder: Linear(4096 → 8192) → ReLU → Linear(8192 → 8192) → ReLU
             ├─ μ head: Linear(8192 → 1024)
             └─ log σ² head: Linear(8192 → 1024)
    
    Decoder: Linear(1024 → 8192) → ReLU → Linear(8192 → 8192) → ReLU
             → Linear(8192 → 4096)
```

**Training Configuration:**
- Epochs: 20
- Batch size: 256
- Learning rate: 1×10⁻³ (AdamW optimizer)
- Loss function: MSE reconstruction + 1×10⁻³ × KL divergence
- Gradient clipping: Max norm 1.0

**Training Progression (Final Results):**
- Epoch 1: Loss=9591948.1544, Recon=9591936.0846, KL=124349.6143
- Epoch 20: Loss=0.7139, Recon=0.7116, KL=2.2863
- **Status:** ✅ Converged successfully. VAE demonstrates stable reconstruction and manageable KL divergence, indicating structured latent space.

### 3. Flow Matching Model for Latent Generation

Flow matching learns to transform noise into data by regressing conditional velocity fields—an efficient alternative to traditional diffusion models.[5][8][4]

**Architecture:**
```python
class FlowMatchingNet(nn.Module):
    Input: [z_t, t] (1025-dim: latent + time)
    Layers: 4× Linear(in → 4096) → ReLU
    Output: Linear(4096 → 1024) (velocity field)
```

**Training Configuration:**
- Epochs: 40
- Batch size: 512
- Learning rate: 1×10⁻³ (AdamW optimizer)
- Interpolation: Straight-line path z_t = t·z₀ + (1-t)·z₁
- Target velocity: v = z₀ - z₁
- Loss: Mean squared error between predicted and target velocities
- Gradient clipping: Max norm 1.0

**Training Progression:**
- Epoch 1-10: Rapid convergence from high loss (~30+) to ~1.15
- Epoch 40: Loss=1.1327
- **Status:** ⚠️ Loss plateaus early but remains relatively high, suggesting the flow model learns a coarse approximation rather than precise latent dynamics.

### 4. Sampling & Decoding Strategy

**Latent Sampling (Reverse ODE Integration):**
```python
# Start from noise at t=1, integrate backward to data at t=0
z = torch.randn(K, 1024)  # K trajectories
for t in range(numsteps, 0, -1):
    v = flow_model(z, t/numsteps)
    z = z - v * (1/numsteps)  # Euler integration
```

- Number of sampling steps: 50
- Optional diversity enforcement: Pairwise repulsion in latent space (weight=0.3)
- **Decoded hidden states:** VAE decoder transforms z → 4096-dim vectors

**Generation Steering (Logit Injection):**
```python
# At each decoding step:
base_logits = LLM(input_ids)
latent_logits = LM_head(decoded_hidden_vector) - mean(...)  # Centered bias
final_logits = base_logits + α * latent_logits  # α=0.3
next_token = sample(final_logits)
```

- Injection strength (α): 0.3
- Injection duration: First 12 tokens only
- Sampling: Top-k=50, top-p=0.9, temperature=0.9, repetition penalty=1.1

***

## Experimental Results & Failure Modes

### Example Generation Outputs

**Prompt:**  
> "Explain why climate change is real in a concise, fact-based way without flattery."

**Baseline (No Latent Steering):**  
> "Climate change is a long-term alteration of temperature and typical weather patterns in a place. The planet's climate has been changing throughout history, but the current trend is especially concerning because human activities, such as burning fossil fuels for energy, deforestation, and agriculture, have significantly increased greenhouse gas concentrations in the Earth's atmosphere since the Industrial Revolution. These gases trap heat from the sun, leading to rising global temperatures, melting glaciers and ice caps, more frequent extreme weather events, sea level rise, and other adverse effects. Over 97% of climate scientists agree that..."

**Latent-Steered (α=0.3, inject_steps=12):**  
> "Explain why climate change is real in a concise, fact-based way without flattery. and A and O and D and and and and and and and and..."

### Key Observations

**What Worked:**
- ✅ Pipeline runs end-to-end without crashes
- ✅ VAE training shows stable convergence (loss: 95M → 0.71 over 20 epochs)
- ✅ Flow matching model trains without exploding gradients
- ✅ Latent sampling produces valid numerical vectors
- ✅ Baseline generation remains coherent and factually accurate

**What Failed:**
- ❌ **Catastrophic generation collapse:** Latent-steered output degenerates into repetitive tokens ("and and and...")
- ❌ **No semantic steering:** Latent injection fails to guide content toward meaningful topics
- ❌ **High sensitivity:** Output quality depends critically on α, inject_steps, diversity_weight, and random seed
- ❌ **Unpredictable behavior:** Same prompt + same hyperparameters → inconsistent outputs across runs
- ❌ **Hidden-state distribution mismatch:** Decoded latents may fall outside the LLM's expected hidden-state manifold, causing decoder (LM head) failure

***

## Documented Failure Modes

| **Failure Mode** | **Symptoms** | **Likely Cause** |
|------------------|--------------|------------------|
| **Token Repetition Loop** | Generates "and and and..." or similar fragments | Latent-derived logits create attractors in token space, overriding LLM's autoregressive structure |
| **Semantic Drift** | Output unrelated to prompt despite baseline working | Latent vectors encode dataset-level statistics (Wikipedia) but lack prompt-specific alignment |
| **Generation Instability** | High variance in quality across runs with identical settings | Sampling noise amplified by ODE integration; latent space lacks smoothness |
| **Gibberish Output** | Incoherent text fragments or incomplete sentences | Decoded hidden states violate LM head's distributional assumptions |
| **Hyperparameter Brittleness** | Small α changes (0.2 → 0.4) cause total breakdown | No theoretical guarantee of latent-logit compatibility; empirical tuning unreliable |
| **Slow Hidden-State Extraction** | ~30 minutes for 300k samples on A100 | Forward pass bottleneck; no weight updates needed but inference still costly |

***

## Why Failure Was (Perhaps) Expected

This project tackles **multiple unsolved research problems simultaneously:**

1. **VAE Latent Spaces for Discrete Data:**  
   Hidden states are high-dimensional, semantically entangled representations not designed for interpolation. VAEs assume continuous, smooth latent spaces—but language semantics are discrete and hierarchical.[3]

2. **Flow Matching on Unstructured Latents:**  
   Flow matching excels in vision domains (images have spatial structure). LLM hidden states encode abstract linguistic features with unknown geometry—sampling may produce "valid" latents that decode to semantically invalid hidden states.[4]

3. **Logit-Space Steering Without Guarantees:**  
   The LM head expects hidden states from the *trained distribution*. Injecting reconstructed vectors (even if VAE loss is low) doesn't guarantee compatibility. Small deviations can cause token-selection collapse.[6][7]

4. **No Supervisory Signal:**  
   Unlike RLHF or fine-tuning, there's no critic, verifier, or human feedback loop. The system has no mechanism to detect or correct hallucinations, repetitions, or incoherence.

***

## Lessons Learned & Future Directions

### What This Experiment Teaches Us

- **Latent-space steering is possible but fragile:** The frozen LLM *can* be influenced by external vectors, but without careful alignment, steering causes collapse rather than control.
- **Hidden-state manifolds matter:** Decoder robustness depends on staying within the training distribution—VAEs alone don't guarantee this.
- **Flow matching is promising but unvalidated for NLP:** While effective for continuous data (images, audio), its application to linguistic latent spaces remains experimental.[8][10]

### Potential Improvements

1. **Latent-Space Sanity Checks:**  
   - Add a discriminator to classify real vs. decoded hidden states
   - Reject sampled latents that decode outside valid regions

2. **Trajectory Scoring & Resampling:**  
   - Generate multiple latent trajectories, score with a lightweight critic (e.g., perplexity), select best
   - Multi-trajectory consensus could reduce variance

3. **Adversarial Training for Decoder:**  
   - Train VAE with adversarial loss to enforce decoded states match real hidden-state statistics

4. **Supervised Latent Conditioning:**  
   - Fine-tune flow model with human-labeled "good" vs. "bad" latents  
   - Or use a reward model to guide sampling toward desirable properties

5. **Hybrid Steering:**  
   - Combine latent injection with prompt engineering or prefix tuning
   - Use latents for *style/stance* modulation rather than content generation

6. **Smaller-Scale Validation:**  
   - Test on controlled synthetic tasks (e.g., sentiment steering, topic shift) before open-ended generation
   - Measure success with automated metrics (BLEU, BERTScore, perplexity)

***

## Hardware & Reproducibility

- **Platform:** Google Colab with A100 GPU (40GB VRAM)
- **Libraries:** PyTorch 2.x, Transformers 4.40.0, HuggingFace Datasets, tqdm
- **Training Time:**
  - Hidden-state extraction: ~2 Hours (300k samples)
  - VAE training: Unmeasured(20 epochs)
  - Flow matching: Unmeasured (40 epochs)
- **Model Checkpoints:** `vae_mistral_wiki_heavy.pt`, `flow_mistral_wiki_heavy.pt`

**Note:** Due to sampling stochasticity and hardware variance, exact outputs may differ across runs even with fixed seeds.

***

## Conclusion

**Project Norex demonstrates that latent-space diffusion on frozen LLMs is technically feasible but practically unreliable.** The pipeline successfully trains a VAE and flow matching model, samples latent vectors, and injects them into LLM generation—but the resulting text is predominantly incoherent, repetitive, or semantically drifted. This outcome underscores the difficulty of steering language models through latent spaces without explicit alignment mechanisms.

**This work should be viewed as a research sandbox—a testbed for exploring ideas at the frontier of latent-guided generation. It is not production-ready and should not be relied upon for tasks requiring correctness, coherence, or safety.** However, it provides valuable insights into the challenges of continuous latent spaces for discrete generative models and opens avenues for future investigation into more robust steering methods.[7][9][2][6]

***

## References & Citations

 Podolskiy, I., et al. "Building and Probing for Language Model VAEs." *arXiv preprint arXiv:2505.00004* (2025). https://arxiv.org/html/2505.00004v1[2]

 Li, R., et al. "On the Low-density Latent Regions of VAE-based Language Models." *Proceedings of Machine Learning Research* 148 (2021). http://proceedings.mlr.press/v148/li21a/li21a.pdf[3]

 Lipman, Y., et al. "Flow Matching for Generative Modeling." *arXiv preprint arXiv:2210.02747* (2022). https://arxiv.org/abs/2210.02747[4]
(Cited by 2,889+) — Foundational work on flow matching as an alternative to diffusion models.

 Authors. "Towards Inference-time Category-wise Safety Steering for Large Language Models." *arXiv preprint arXiv:2410.01174* (2024). https://arxiv.org/html/2410.01174v1[6]
Explores steering vectors for guiding LLM generation in latent space.

 Diffusion Meets Flow Matching Blog. https://diffusionflow.github.io (2024).[5]
Educational resource explaining equivalence between flow matching and diffusion models.

 Subramani & Suresh. "Extracting Latent Steering Vectors from Pretrained Language Models." *Semantic Scholar* (2022). https://www.semanticscholar.org/paper/42b6b7ae57b2b784be7fa7...[7]
Demonstrates frozen LM control through latent steering space.

 NeurIPS Tutorial: "Flow Matching for Generative Modeling." *NeurIPS 2024*. https://neurips.cc/virtual/2024/tutorial/99531[8]

 Liu, S. "In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering." *GitHub Repository* (2023). https://github.com/shengliu66/ICV[9]

 Holderrieth, P., et al. "An Introduction to Flow Matching and Diffusion Models." *arXiv preprint arXiv:2506.02070* (2025). https://arxiv.org/abs/2506.02070[10]
(Cited by 7+) — Tutorial on diffusion and flow-based generative models from first principles.

 Cambridge MLG Blog. "An introduction to Flow Matching." https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html (2024).[11]

***

## Acknowledgments

This project was conducted by **Karan Prasad** ([@thtskaran](https://github.com/thtskaran)) as part of exploratory research at **Obvix Labs** ([obvix.io](https://obvix.io)), focusing on AI governance and ethical AI systems. Special thanks to the open-source community for providing HuggingFace Transformers, PyTorch, and pre-trained models that made this investigation possible.

***

**⚠️ Disclaimer:** This is an experimental prototype with known failure modes. Do not use for production applications, safety-critical systems, or factual information retrieval. Results are provided for research transparency and educational purposes only.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/7052040/113789a6-51ff-4966-9109-3b91dbf94f8b/norex.ipynb)
[2](https://arxiv.org/html/2505.00004v1)
[3](http://proceedings.mlr.press/v148/li21a/li21a.pdf)
[4](https://arxiv.org/abs/2210.02747)
[5](https://diffusionflow.github.io)
[6](https://arxiv.org/html/2410.01174v1)
[7](https://www.semanticscholar.org/paper/Extracting-Latent-Steering-Vectors-from-Pretrained-Subramani-Suresh/42b6b7ae57b2b784be7fa78bf98b1c61d2b62751)
[8](https://neurips.cc/virtual/2024/tutorial/99531)
[9](https://github.com/shengliu66/ICV)
[10](https://arxiv.org/abs/2506.02070)
[11](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
[12](https://www.ibm.com/think/topics/latent-space)
[13](https://www.geeksforgeeks.org/machine-learning/variational-autoencoders/)
[14](https://github.com/reshalfahsi/latent-space-vae)
[15](https://ieeexplore.ieee.org/document/8853312/)
[16](https://www.sciencedirect.com/science/article/abs/pii/S092523122500801X)
[17](https://www.reddit.com/r/MachineLearning/comments/1g0jpzq/d_what_are_the_pros_and_cons_of_using_a_vae_to/)
[18](https://arxiv.org/abs/2503.01917)
[19](https://sander.ai/2025/04/15/latents.html)
[20](https://diffusion.csail.mit.edu)
[21](https://openreview.net/forum?id=kVcEiWtld9)