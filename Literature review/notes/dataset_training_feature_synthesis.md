# Literature synthesis: datasets, training, and translation-relevant features

Working notes across the collected papers, focused on three questions per paper:
what dataset(s) it trains/evaluates on, how the model is trained, and what
features/design choices matter for image-to-image translation quality. Compiled
to inform the CycleGAN root-cause fix and the eventual UNSB upgrade.

---

## Cluster: Image-to-Image Translation

### Dual-approx Bridge (Xiao et al., CVPR 2025) — `brownian_bridge_I2I_CVPR2025.pdf`
- **Dataset:** Cityscapes (256×256, label→photo, *paired*), Edges2Handbags (64×64,
  paired), DF2K for SR training, tested on Urban100/BSD100 (256×256 patches).
- **Training:** Paired dataset `D_{x,y}`. Two networks: a forward approximator
  (predicts x0 from noisy xt) and a reverse approximator (predicts the noise
  increment), trained independently via denoising-bridge SDE objectives — makes
  sampling deterministic instead of stochastic. PyTorch, 2× RTX A6000.
- **Feature/translation takeaway:** This is **paired** I2I — not directly
  applicable to our unpaired nadir↔perspective problem, but useful as an
  evaluation-metric reference (FID, LPIPS, PSNR, SSIM all reported together) and
  as a CycleGAN baseline data point: even on *clean, paired, same-content*
  benchmarks, CycleGAN is the weakest baseline (Cityscapes FID: CycleGAN 75.37 vs
  their model 48.70; Pix2pix 101.04, UNIT 82.74). CycleGAN structurally
  underperforms newer bridge/diffusion methods even under ideal data conditions.

### Unpaired Neural Schrödinger Bridge, UNSB (Kim et al., ICLR 2024) — `schrodinger_bridge_I2I.pdf`
**This is the exact model the methodology chapter targets as the Stage-2 upgrade
beyond the CycleGAN baseline — read closely.**
- **Datasets:** Horse2Zebra, Summer2Winter, Label2Cityscape, Map2Satellite. All
  resized to 256×256. **Critical observation: every one of these benchmark pairs
  shares scene content/structure between domains** — Horse2Zebra is the same
  quadruped-in-grassy-field composition with only texture changing; Map2Satellite
  and Label2Cityscape are literally the same geographic location rendered two
  ways, just used in an unpaired training regime. None of the standard unpaired
  I2I literature validates on domain pairs that differ in **underlying physical
  content** (terrain type, object density) the way our HiRISE-Oxia-Planum
  (smooth clay plain) ↔ Rover-Gale-Crater (boulder field) pairing does.
- **Training:** Schrödinger Bridge cast as a sequence of adversarial learning
  problems (N=5 discretized timesteps) rather than solved directly — direct SB
  solvers (Sinkhorn-Knopp, SB-FBSDE, DSB) suffer a "curse of dimensionality":
  in high-dimensional (image) space, finite unpaired samples fail to describe
  the true image manifold, so the optimal-transport map ends up pairing points
  that don't meaningfully correspond. Two components fix this:
  1. **A Markovian (patch-level) discriminator** instead of an instance-level
     one — ablation shows this alone takes Horse2Zebra FID from 230→66.3.
  2. **An explicit regularization term** `L_Reg` enforcing a similarity measure
     `R(x0, x1)` between the predicted target and the source (they use negative
     cosine similarity in feature space) — a *softer, more targeted* consistency
     constraint than CycleGAN's full round-trip cycle loss. Ablation: adding
     patch discriminator + regularization together takes FID from 66.3 → 35.7
     (full UNSB) on Horse2Zebra.
- **Explicit hallucination discussion (directly relevant to our failure):** the
  paper calls out that Neural Optimal Transport (NOT) "hallucinates structures
  not present in the source image" when the transport/correspondence problem is
  too hard, and diffusion baselines (SDEdit, P2P) "fail to fully reflect the
  target domain style." I.e., **when source/target content is mismatched or the
  problem is high-dimensional and hard, every unpaired method's known failure
  mode is either (a) hallucinating unsupported content or (b) ignoring the
  target domain and reproducing the source** — which is exactly the pattern we
  saw in the CycleGAN sample grid (near-identity reconstruction, no real
  target-domain content).
- **Quantitative superiority over CycleGAN:** UNSB beats CycleGAN by a large
  margin on every benchmark (Horse2Zebra FID 35.7 vs 77.2; Label2Cityscape 53.2
  vs 76.3), confirming CycleGAN is a weak baseline in general, not just in our
  broken-data case.
- **Translation-feature takeaway for us:** if/when we move to UNSB, (1) a
  patch-level discriminator is doing a lot of the work — worth checking our
  current CycleGAN's PatchGAN-70 is comparably effective, and (2) a *soft*
  similarity regularizer may transfer better to a content-mismatched domain pair
  than a strict cycle-consistency loss, since it doesn't force a full
  information-preserving round trip that a generator can game via steganography.

---

## Cluster: Hallucination / Reliability

### Understanding Hallucinations in Diffusion Models through Mode Interpolation (Aithal et al., NeurIPS 2024) — `hallucination_diffusion_neurips2024.pdf`
- **Dataset:** Synthetic (1D/2D Gaussian mixtures, Simple Shapes toy dataset),
  plus a real-world Hands dataset (5,000 images, 190 subjects) and MNIST for the
  recursive-training experiments.
- **Mechanism:** Diffusion models learn a *smooth* approximation of the true
  score function, but the true data distribution has sharp discontinuities
  between disjoint modes. The network can't represent that discontinuity, so it
  "mode interpolates" — generates samples that blend between two nearby training
  modes, landing in regions with ~zero real probability (hallucinations). Shown
  concretely on hands (extra/missing fingers) and shapes (duplicated shapes never
  seen together).
- **Explicit dependency on data quantity/diversity (directly relevant to us):**
  "The rate of mode interpolation depends primarily on three factors: (i) number
  of training data points, (ii) variance of (and distance between) the
  distributions, (iii) number of sampling timesteps." More training samples
  shrinks the interpolated/hallucinated region; sparser training data (our
  7-observation corpus) is exactly the regime that maximizes this effect.
- **Detection:** hallucinated samples show high variance in the predicted x0
  trajectory near the end of sampling — diffusion models "know" when they're
  hallucinating and this is detectable/filterable (>95% removal in their tests).
  This is a **DiffusionSR-stage-relevant technique** (our pipeline's Stage 1),
  not directly applicable to the GAN-based CycleGAN stage, but worth carrying
  into the methodology chapter's hallucination-rate metric design.

### From Hallucination to Reliability (Rathkopf, 2026) — `hallucination_to_reliability.pdf`
- Philosophical/methods paper, not empirical — no dataset of its own; case
  studies are AlphaFold3 (protein structure) and GenCast (weather).
- **Core argument:** hallucination is an *inevitable* consequence of any
  generative model producing high-dimensional structured output from
  comparatively sparse training data (information-theoretic and geometric
  arguments — output space vastly exceeds what any training set can cover, so
  regions with little/no training support will always exist and get sampled).
  Directly reinforces the mode-interpolation paper's finding, from a different
  angle: **hallucination isn't a bug to eliminate, it's a risk to bound.**
- **Two mitigation strategies observed in mature scientific generative
  systems**, both directly applicable to our methodology chapter:
  1. **Theory-guided training** — embedding known physical/domain constraints
     into the model architecture or loss (this is literally what our
     methodology's "physics-constrained UNSB" is attempting).
  2. **Confidence-based error screening** — post-hoc filtering/flagging of
     low-confidence outputs rather than trusting every generated sample
     (parallels the mode-interpolation paper's variance-based hallucination
     detector, and suggests our eventual hallucination-rate evaluation should
     include a per-sample confidence/uncertainty signal, not just an aggregate
     rate).
- **Reframe for our failure diagnosis:** our CycleGAN's problem isn't
  "hallucination" in this paper's narrow sense (it isn't confidently fabricating
  plausible-but-wrong content) — it's closer to the opposite failure, an
  under-constrained generator that fails to leave the source distribution at
  all. But the underlying cause is the same one this cluster identifies: too
  little, too narrow training data relative to the complexity of the target
  distribution.

---

## Cluster: Mars / Remote-Sensing Super-Resolution

### MARSGAN (Tao et al., MDPI Remote Sensing 2021) — `caSSIS_SISR.pdf`
- **Dataset:** Trained entirely on **HiRISE** images (self-supervised: real HiRISE
  = HR target, synthetically downsampled version = LR input — no separate sensor
  pairing needed), then *applied* (not trained) to CaSSIS (4-5m/px → ~3x
  enhancement). Evaluated across **8 deliberately terrain-diverse study sites**
  spanning bedrock layers, dune fields, gully networks, recurring slope lineae,
  scalloped depressions, and polar "spider" terrain — explicit design choice for
  generalization testing.
- **Training/architecture:** ESRGAN-derived backbone (AW-RRDB blocks with noise
  injection for stochastic local variation), relativistic discriminator,
  multi-scale reconstruction, PatchGAN-style discriminator.
- **Feature takeaway:** Confirms the standard practice in this exact sub-field
  (Mars orbital SR) is self-supervised paired training (downsample your own HR
  image), and that **terrain-type diversity in evaluation is treated as a
  first-class requirement**, not an afterthought — directly reinforcing the
  diversity gap we found in our corpus.

### RSTSRN (Wu et al., MDPI Applied Sciences 2024) — `RSTSRN_mars_SR.pdf`
- **Dataset:** 5,000 paired HiRISE/downsampled-HiRISE 512×512 crops, extracted
  from a **single** observation (ESP_066115_2055) for training; 1,500 test crops
  from two other observations. Also pretrained/validated on standard natural-
  image SR benchmarks (DIV2K, Set5/14, BSD100, Urban100, Manga109).
- **Explicit domain-difficulty note:** "The types of surface features on Mars
  are monotonous with no vegetation... equally monotonous color information...
  extremely challenging for SR... the monotonous image information is difficult
  for feature learning mechanisms to obtain the high-frequency information
  required." I.e., **Mars terrain is inherently low-texture/low-frequency
  compared to natural images** — this is a known, literature-documented
  challenge independent of our pipeline, and likely a genuine contributor (on
  top of the diversity-collapse bug) to why our "real HiRISE" samples look
  grain-like at Oxia Planum specifically (one of the smoothest Mars regions).
- **Important nuance vs. our CycleGAN task:** because SR here is a *paired,
  same-image* task (HR image downsampled to make its own LR pair), a single
  source image can legitimately generate thousands of valid, non-redundant
  training pairs — there's no cross-domain correspondence problem to solve. Our
  CycleGAN task has no such synthetic pairing available; it fundamentally
  depends on distributional diversity across many source images. Single/few-
  image training being fine for SR does **not** transfer as a justification for
  few-image training in our unpaired I2I setting.

### DiffusionSat (Khanna et al., ICLR 2024) — `DiffusionSat.pdf`
- **Dataset:** fMoW (Function Map of the World — global coverage, 62 land-use
  categories, GSD 0.3-1.5m), Satlas (NAIP + Sentinel-2), SpaceNet — a genuinely
  global-scale, high-diversity, high-category-count generative foundation model
  for satellite imagery, several orders of magnitude larger/more diverse than
  our 7-observation corpus.
  **Very** large diffusion model. Not directly transferable in scale.
- **Feature takeaway (transferable):** conditions generation on cheap numerical
  **metadata** (lat/long, GSD, cloud cover, timestamp) via sinusoidal embeddings
  instead of requiring text captions — and shows this measurably improves
  output quality/control over caption-only conditioning. For our pipeline, an
  equivalent would be conditioning the translation model on solar
  incidence/local time (available from HiRISE metadata) since lighting/shadow
  geometry is a known hard feature for Mars image realism.

### Taming a Diffusion Model to Revitalize RS Image SR / RSDiffSR (Zhu et al., MDPI RS 2025) — `taming_diffusion_RS_SR.pdf`
**Most strategically relevant SR paper — proposes a way around exactly our
data-scarcity problem.**
- **Core diagnosis (stated explicitly, mirrors our situation):** existing
  diffusion-based remote-sensing SR models underperform natural-image diffusion
  models for two reasons: fewer model parameters, and — critically —
  "the diversity, quantity, and quality of remote sensing image datasets are
  not comparable to those of natural image datasets," since RS diffusion models
  are trained on "tens of thousands" of images vs. billions for natural-image
  foundation models.
- **Their fix, not "collect more RS data" but transfer learning:** take a huge
  pretrained natural-image diffusion model (Stable Diffusion XL, pretrained on
  billions of images) as a **generative prior**, and adapt it to the RS domain
  via **LoRA (low-rank adaptation)** fine-tuning plus a lightweight
  content/edge-guidance module (Canny-edge + content encoder) to keep the
  output grounded in the actual input structure rather than the prior's natural-
  image bias.
- **Directly actionable for us:** this is a real alternative to "scale up the
  Oxia Planum corpus" — instead of trying to out-collect a fundamentally
  data-scarce domain, fine-tune a large pretrained I2I/diffusion backbone (e.g.
  a pretrained UNSB/diffusion checkpoint, or even a natural-image CycleGAN/I2I
  checkpoint) via LoRA on our small HiRISE/Rover corpus, rather than training a
  304-image CycleGAN from scratch. Worth weighing against the "collect a much
  bigger single-region corpus" plan discussed earlier — the two aren't mutually
  exclusive.

### Super-Resolution of Mars Thermal IR Images (Lu & Su, MDPI RS 2025) — `mars_thermal_IR_SR.pdf`
- **Problem framing directly parallels ours:** "There are no higher-resolution
  thermal infrared images on the Martian surface, so it is impossible to obtain
  real and credible references" for training/validation — the same
  no-ground-truth situation we face for Oxia Planum rover imagery (ExoMars
  hasn't landed).
- **Their solution:** cross-modal domain adaptation — use a *different,
  correlated* image modality (visible light, which Mars has in abundance) as a
  guidance signal, plus an explicit **physical-consistency constraint** (thermal
  radiation flux must match the original LR image after upsampling) so the
  network can't hallucinate radiometrically implausible detail.
- **Feature takeaway:** another concrete example (alongside UNSB's
  regularization and DiffusionSR's methodology chapter) of the field's general
  answer to "no ground truth in the target domain": don't try to force paired
  supervision, instead constrain the generator with a domain-appropriate
  physical/statistical invariant.

---

## Cluster: 3D Reconstruction / DTM Generation (physics-conditioned GANs)

### PFMGAN (Zou et al., MDPI Remote Sensing 2026) — `PFMGAN_DTM.pdf`
**Single most actionable paper for the "what features matter for translation"
question — read closely.**
- **Dataset (concrete scale benchmark for us):** 714 valid HiRISE image–DTM
  pairs → tiled into 250,407 candidate 256×256 patches → curated down to
  **18,986 final image-DTM training pairs** via a rigorous multi-stage filter:
  (1) SSIM between smoothed image and its hillshade + elevation-std to score
  texture/terrain-complexity, keep top 60%; (2) automatic removal of tiles with
  black-border/no-data pixels; (3) **manual removal of "nearly featureless or
  repetitive low-information areas"** (i.e. human-in-the-loop review beyond
  automatic entropy filtering — our pipeline has no equivalent manual QA step);
  (4) deliberately added extra dune-field scenes to boost aeolian-landform
  diversity. **Test regions are explicitly geographically independent of
  training regions.**
  **This is the concrete scale reference to cite: a published, working Mars
  generative-DTM model uses 714 source scenes; our CycleGAN corpus uses 7.**
- **Key feature-engineering insight:** they identify that grayscale brightness
  in a Mars image is a *coupled* function of intrinsic surface albedo and
  slope-induced shading, and that models which learn this relationship
  implicitly (as our CycleGAN does) suffer a "bump-and-hollow" sign ambiguity.
  Their fix: feed **solar azimuth and elevation angle as explicit conditioning
  vectors** (equal footing to the image itself) through an Albedo-Aware
  Attention module, explicitly decoupling reflectance from topography — >50%
  reconstruction-error reduction vs. baselines that omit this.
  **Directly transferable to our nadir→perspective translation:** HiRISE
  images carry solar-incidence metadata for free; conditioning the
  generator on it could help it distinguish "this is dark because of shadow"
  from "this is dark because of material," which matters for a
  low-relief, shading-dependent site like Oxia Planum.
- **Also uses a downsampled version of the target itself as an auxiliary
  low-frequency prior** ("to substitute for the lack of high-resolution global
  base maps") — another instance of the general pattern: constrain the
  generator with an auxiliary physically grounded signal instead of hoping
  raw pixel-to-pixel adversarial training alone will generalize from little
  data.

### MADNet 2.0 (Tao et al., MDPI Remote Sensing 2021) — `MADNet2_topography.pdf`
- **Directly on our target site:** demonstrated specifically on HiRISE imagery
  over **Oxia Planum**, the ExoMars Rosalind Franklin landing site.
- **Global HiRISE scarcity context (useful methodology-chapter framing):**
  HiRISE stereo coverage is only 3.4% of the Martian surface, and PDS HiRISE
  DTMs cover a mere 0.03% — so working with a geographically narrow, low-
  coverage dataset is an intrinsic constraint of HiRISE-based Mars ML in
  general, not unique to our project. The field's response isn't "wait for
  more data" but architectural: use co-registration against a lower-resolution
  *global* reference (CTX/MOLA/HRSC DTM) as a coarse structural prior, then let
  the network add fine detail — again, auxiliary-prior-constrained generation
  rather than pure data scaling.
- **Training:** relativistic GAN, multi-scale U-Net generator, trained on
  PDS HiRISE DTMs + iMars CTX DTMs (multiple observations, not single-scene).

---

## Cluster: NeRF / VR / Planetary Rendering

### Martian World Model / M3arsSynth + MarsGen (Li et al., 2025) — `martian_world_model.pdf`
**Most sophisticated data-curation pipeline seen across this whole review — directly
worth adapting for our HiRISE/Rover preprocessing.**
- **Dataset:** Curiosity/Perseverance rover stereo navcam imagery from NASA PDS,
  reconstructed into 3D via a pretrained geometric foundation model (VGGT) into
  10K+ physically accurate 3D Martian surface models, used to train a
  video-generation model (MarsGen).
- **Explicit problem framing (mirrors ours exactly):** "the scarcity of
  high-quality Martian data and the significant domain gap between Martian and
  terrestrial imagery" is *the* named challenge; raw PDS rover imagery is
  "sparse and photometrically inconsistent."
- **Automated data-filtering pipeline (directly transferable technique for our
  HiRISE/Rover preprocessing):**
  1. Discard thumbnails/grayscale images (low RGB channel variance).
  2. **De-duplicate near-identical frames via perceptual hashing + Hamming
     distance** — this is exactly the problem our corpus has (30k patches from
     7 images, heavily redundant) and exactly the kind of automated check our
     pipeline lacks.
  3. Sharpness filter via Laplacian variance (reject blurry/out-of-focus).
  4. Reject frames with anomalous color-intensity histograms.
  5. **Semi-automated refinement:** Grounded-SAM segmentation to identify and
     mask out non-terrain objects (rover hardware, shadows of the arm, etc.)
     before use.
- **Explicit acknowledgement that Mars terrain reconstruction is intrinsically
  harder than Earth scenes** "stemming from the planet's often texture-poor
  terrain and the inherent scarcity of observational data" — same texture-
  poverty point RSTSRN made for SR.

### MaRF (Giusti et al., 2022) — `MaRF_mars_nerf.pdf`
- **Dataset:** Curiosity, Perseverance, and Ingenuity imagery from PDS, used to
  train per-scene NeRFs compressed into small MLPs ("neural graphics
  primitives") for mixed-reality/VR use.
- **Relevance:** direct precedent for our Stage 3 (Gaussian Splatting for VR) —
  same rover-imagery source, same end goal (immersive scientist "presence" on
  Mars), same limitation (works only where rover imagery already exists — it
  cannot synthesize a site no rover has visited, which is exactly the gap our
  nadir→perspective translation stage is meant to fill).

### High-fidelity 3D reconstruction for planetary exploration (Martínez-Petersen et al., 2026) — `high_fidelity_3D_planetary.pdf`
- **Method, not dataset-driven:** a NeRF/Gaussian-Splatting (Nerfstudio +
  COLMAP + Splatfacto-W) pipeline for rover-collected rosbag data.
- **Named failure conditions for correspondence-based reconstruction on Mars**
  (relevant corroboration): "low-texture or homogeneous terrains (e.g., flat
  sand or uniform rock surfaces)," "large illumination changes," "low overlap
  between consecutive frames," "repetitive or ambiguous geometric patterns."
  This is the third independent paper in this review to flag flat/low-texture
  Mars terrain (exactly Oxia Planum's character) as a recognized hard case for
  vision pipelines generally, not an artifact specific to our project.

*(NeRF_space_review.pdf and neural_3D_descent_imagery.pdf were scanned at
title/abstract level only — general NeRF survey and descent-phase imagery
reconstruction respectively; lower direct relevance to the translation-model
data question and not read in full given time budget.)*

---

## Cluster: HiRISE Data Characteristics

### Predicting HiRISE-equivalent Rock Density on Mars Using CTX Image Features (Serrano et al., JPL/AIAA) — `HiRISE_rock_density_JPL.pdf`
**Surfaces a confound worth checking in our own corpus.**
- **Dataset:** HiRISE + CTX image pairs of the Mars Phoenix landing site
  (Vastitas Borealis / Scandia region, northern plains), with 7 geomorphic
  units mapped (highlands, blocks/mesas, knobs, lowlands bright/dark, crater
  interior, crater ejecta) — rock density varies by an order of magnitude
  across these units.
- **Key mechanism (a real confound for our diagnosis):** rocks in HiRISE
  imagery are detected via **shadow analysis** — an automatic rock-detection
  algorithm extracts cast shadows (Maximum Entropy Thresholding) and fits a
  shadow ellipse using the **known solar incidence angle** to back out rock
  height/diameter. This means **rock visibility in a nadir HiRISE image is not
  just a function of true rock abundance — it depends heavily on solar
  elevation at capture time.** A HiRISE observation taken near local noon
  (high sun, minimal shadows) will look flatter/more textureless than one
  taken at a low sun angle, *independent of the underlying terrain*.
  **Action item:** worth checking whether the local time / solar incidence
  angle of our 7 Oxia Planum observations happened to be high-sun (thus
  suppressing shadow-driven texture) — this would compound, and partially
  explain independently of, the terrain-content and diversity-collapse causes
  already identified.
- Confirms rock density is literally the primary landing-site-safety metric,
  reinforcing that Oxia Planum was selected (like Phoenix's site) specifically
  to minimize rock hazard — i.e., genuinely low rock content by mission-design
  intent, not just perception.

*(Note: `HiRISE_DTMs_USGS.pdf` in the collection is mislabeled/misfiled — its
actual content is a USGS "System Characterization of Earth Observation
Sensors" report about Landsat 8, unrelated to Mars or HiRISE. The intended
paper per the reading list, "Revealing active Mars with HiRISE digital terrain
models" (USGS), was not actually downloaded. Worth re-fetching if that specific
paper's content is needed.)*

---

## Cluster: Web search — papers not in the local collection

### Satellite-to-street-view / cross-view synthesis (a whole subfield we were missing)
**This is the single biggest gap in the existing collection.** The existing
papers frame our problem via generic unpaired I2I benchmarks (Horse2Zebra,
Cityscapes) or Mars-specific SR/DTM work, but there is an entire established
Earth-remote-sensing subfield doing **exactly our translation direction** —
nadir/orbital → ground-level/perspective — that wasn't in the original ~50-paper
list: **Sat2Den, Sat2Density++, Sat2Scene, Sat2Vid, Sat3DGen, SatDreamer360,
CrossViewDiff, Geometry-Guided Street-View Panorama Synthesis (Shi et al.,
TPAMI 2022)**. [Sat2Den (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/af28c8429dbaab9d7156f5612c044b28-Paper-Conference.pdf),
[Sat2Scene](https://arxiv.org/pdf/2401.10786),
[Sat2Vid](https://arxiv.org/pdf/2012.06628),
[Geometry-Guided Street-View Panorama Synthesis](https://arxiv.org/pdf/2103.01623),
[CrossViewDiff](https://arxiv.org/abs/2408.14765).
- **Consistent methodological pattern across this whole subfield, and the
  single most important architectural takeaway from this entire literature
  review:** none of these do pure end-to-end pixel-space translation the way
  CycleGAN/UNSB do. They all insert an **explicit geometric intermediate**
  between the nadir input and perspective output:
  - Sat2Den learns a 3D density/height field from the satellite image as
    geometric guidance, then projects/renders the street-level panorama from
    it (a satellite-to-street-view projection module using the deterministic
    geometric relationship between a 3D point's height and its projected
    position in each view).
  - CrossViewDiff explicitly separates **"satellite scene structure
    estimation"** (structural control) from **"cross-view texture mapping"**
    (textural control), feeding both into a diffusion denoising process — i.e.
    solve the geometry problem first, then the appearance problem, rather than
    asking one network to solve both simultaneously from pixels alone.
  - Several of these use building/terrain **height as an explicit prior**
    input, not something the network has to infer purely from adversarial
    pressure.
- **Why this matters for us specifically:** our own pipeline *already* has a
  natural source for exactly this kind of geometric intermediate — the DTM/
  height-estimation techniques from the 3D-reconstruction cluster above
  (MADNet 2.0, PFMGAN, both of which already target single-HiRISE-image
  height retrieval, one of them *specifically at Oxia Planum*). Rather than
  asking CycleGAN to learn HiRISE-pixels → Rover-pixels directly (which is
  what's failing), the field's established solution for this exact
  nadir→perspective problem is: **HiRISE image → estimated height/DTM (via
  MADNet2/PFMGAN-style network) → geometric re-projection to a ground-level
  camera pose → texture/appearance synthesis in that projected view (learned,
  ideally still unpaired, from Rover imagery).** This decomposes an
  under-constrained, content-mismatched pixel-translation problem into a
  well-posed geometry step (which doesn't need Rover data at all) plus a
  much easier local-texture-synthesis step. Worth raising as a candidate
  architecture change, not just a data fix.

### Other relevant hits
- **GP-UNIT** (Generative Prior for Versatile Unsupervised I2I Translation,
  [arXiv:2306.04636](https://arxiv.org/pdf/2306.04636)) — designed to bridge
  unpaired translation tasks with *large* content/appearance gaps by
  distilling a generative prior from a large, diverse pretrained model, then
  adapting it to the specific domain pair. Directly relevant to our
  content-mismatch problem, and pairs naturally with the RSDiffSR "don't
  collect more RS data, transfer from a huge pretrained prior instead"
  strategy already noted in the SR cluster. Worth a closer read before the
  next model-selection decision.
- **Masked Discriminators for Content-Consistent Unpaired I2I**
  ([arXiv:2309.13188](https://arxiv.org/pdf/2309.13188)) — targets the
  specific failure mode where a generator changes content it shouldn't;
  relevant to preventing the near-identity/steganographic shortcut diagnosed
  in our CycleGAN run.
- **Exploring reference-guided unpaired I2I translation under limited data**
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125018799))
  — directly on our exact constraint (small target-domain dataset); worth
  reading before finalizing the corpus-scaling plan.
- **LoGAN** (local attentive GAN for Martian 3D reconstruction from HiRISE +
  DTMs) — already in the original ~50-paper list but never downloaded as a
  PDF; confirmed to exist on [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0924271625001492).
  Worth fetching given how relevant the DTM-conditioning cluster turned out to
  be.
- General confirmation from search results (not a single paper, a recurring
  finding across this literature): **GAN translation quality degrades
  markedly, and can fail on content variations absent from training, as
  training-sample count drops** — independent corroboration, from outside our
  specific paper set, of the same "too little, too narrow data" root cause
  already identified from the local collection.

---

## Cross-cutting synthesis: answers to the three original questions

**1. What datasets does this literature typically use?**
Every successful Mars/remote-sensing generative model in this review trains on
substantially more source scenes and more geographic diversity than our
current CycleGAN corpus: MARSGAN uses full HiRISE + 8 deliberately diverse
test sites; RSTSRN's *paired SR* task only needs 1 source image (not
comparable — see caveat below); PFMGAN curates 714 scenes down to ~19,000
tiles with explicit diversity augmentation; MADNet2 trains across many
HiRISE+CTX observations; the Martian World Model paper de-duplicates and
QA-filters 10,000+ 3D reconstructions; the VR inpainting paper uses ~1,150
global HiRISE heightmaps. **Our corpus — 7 observations, 1 region — is roughly
two orders of magnitude below the scale of comparable published Mars deep
learning work**, and unlike the SR papers (where 1 source image can validly
generate thousands of non-redundant *paired* training examples via
self-supervised downsampling), our CycleGAN task is unpaired cross-domain
translation, which has no such shortcut and genuinely needs source diversity.

**2. How are these models trained?**
Three recurring, transferable training patterns, none of which our current
CycleGAN run uses:
- **Physical/metadata conditioning** instead of pure pixel-to-pixel
  adversarial learning: solar angle (PFMGAN), geolocation/GSD/time
  (DiffusionSat), height/geometry (the whole cross-view synthesis subfield).
- **Auxiliary low-fidelity global priors** to compensate for sparse
  high-resolution data: a downsampled DTM (PFMGAN), a coarse CTX/MOLA
  reference (MADNet2), a correlated second modality (visible-light guiding
  thermal-IR SR).
- **Transfer learning from a large pretrained generative prior** rather than
  training from scratch on a small domain-specific set: RSDiffSR (LoRA-tunes
  SDXL), GP-UNIT (distills a generic prior for large-gap translation) — a
  direct, currently-unexplored alternative to "collect a much bigger Mars
  corpus" for our own next step.

**3. What features matter for translation quality?**
- **Patch-level (Markovian) discriminators**, not instance-level ones — UNSB's
  ablation shows this alone is responsible for most of its gain over plain
  CycleGAN-style translation; worth double-checking our current PatchGAN-70
  is actually configured effectively.
- **A soft similarity regularizer instead of (or alongside) strict
  cycle-consistency** — UNSB's regularization term is less exploitable via
  the steganographic near-identity shortcut we diagnosed in the CycleGAN run.
- **Illumination/solar geometry as an explicit input**, not something left for
  the network to infer — directly relevant given Oxia Planum's low relief
  makes it especially shadow-dependent for perceptible texture, and given
  solar angle also directly affects how much real detail is even present in a
  given HiRISE frame (rock-density cluster finding).
- **An explicit geometric (height/DTM) intermediate representation** between
  the nadir input and the perspective output, rather than one network
  learning appearance and viewpoint change simultaneously — the single
  strongest architectural signal from the whole review, and the one most
  clearly actionable given our pipeline already produces single-image DTM
  estimates in the 3D-reconstruction stage.

**Full paper-by-paper notes are in the sections above. Not yet read in full
(time-budget cutoff, noted where relevant): NeRF_space_review.pdf,
neural_3D_descent_imagery.pdf, generative_3D_mars_ISPRS.pdf, MCTED_dataset.pdf.**
