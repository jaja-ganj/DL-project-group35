# Urban Wildlife Monitoring with Bias-Aware YOLO11

**Project category:** AI in Urban Tech

**Project stage:** Existing deep learning research prototype; proposed urban monitoring extension

**Repository:** [github.com/jaja-ganj/DL-project-group35](https://github.com/jaja-ganj/DL-project-group35)

**Keywords:** Urban Biodiversity; Wildlife Monitoring; Computer Vision; YOLO11 Object Detection; Domain Adaptation

## 1. Project Overview & Problem Statement

This project proposes an AI-assisted system for reviewing wildlife images from urban parks, green corridors, and habitats at the edge of cities. It builds on an existing YOLO11 research prototype that detects animals in camera-trap images and compares performance at familiar and unfamiliar camera locations.

Useful wildlife monitoring requires consistent observations across sites and conditions. Camera traps produce image collections that must be reviewed for animal presence, identity, and activity. Manual review takes time, while a model trained at one camera location may become less reliable when used at another.

The central technical problem is **location and background bias**. Fixed cameras repeatedly capture similar backgrounds; lighting, vegetation, camera style, and species frequencies can become prediction shortcuts. Night and infrared imagery, small animals, and partial occlusion create additional difficulties. A detector that works at familiar cameras may therefore miss animals or mislabel observations at a new park.

The existing project investigates transfer using the [Caltech Camera Traps ECCV 2018 benchmark (CCT-20)](https://lila.science/datasets/caltech-camera-traps). Its familiar-location and held-out-location splits permit explicit evaluation of this problem. The proposed extension would turn the research into a practical review tool, followed by validation on local urban species and camera conditions.

**Fit with AI in Urban Tech:** reviewed wildlife observations can support management of urban green infrastructure. City-specific validation, municipal partnerships, and local deployment are proposed work; they have not yet been demonstrated.

## 2. Proposed Solution

The solution combines a wildlife detector, a transparent evaluation workflow, and a planned image-review interface.

### Existing technical foundation

The repository contains five notebooks implementing or exploring:

- COCO-style annotation conversion into YOLO labels, with separate train, cis/trans validation, and cis/trans test preparation.
- A stock YOLO11s architecture baseline with class-weighted detection loss and appearance augmentation.
- Early Instance Normalization and CBAM attention, intended to reduce sensitivity to visual style and improve feature selection.
- Explicit mapping of compatible pretrained weights into the modified architecture.
- Separate cis/trans metrics, validation history, checkpoint export code, and qualitative predictions.
- Experimental DANN-style adaptation, Fourier style transfer, background suppression, and night/IR simulation.

These experimental components are candidates for evaluation, rather than established improvements. The baseline remains the reference and can become the initial demonstration model if modified methods do not improve validation performance.

### Proposed urban workflow

1. A permitted camera-trap image batch is selected for review.
2. The detector proposes boxes, class labels, and confidence scores.
3. A reviewer checks uncertain or unusual observations and corrects or rejects predictions.
4. Approved observations are exported with image IDs, site IDs, timestamps where available, and review status.
5. Repeated frames are grouped into capture events before generating activity summaries by site and time.

The initial demonstration will process stored images. Continuous camera streaming, automatic emergency alerts, and edge-device deployment are outside the initial deliverables. Unrecognized species will require manual review; confidence scores will not be presented as calibrated probabilities of correctness.

## 3. Objectives & Expected Outcomes

| Objective | Planned output and assessment |
| --- | --- |
| Establish a reproducible reference | Baseline configuration, dependency record, checkpoint provenance, and cis/trans metrics. |
| Assess cross-location robustness | Controlled comparisons using trans mAP50–95, per-class AP, precision, recall, and cis/trans gaps. |
| Examine night and difficult images | Day/night and small/occluded-animal analysis where reliable labels are available. |
| Support practical review | Annotated predictions, reviewer decisions, and CSV export in a demonstration interface. |
| Evaluate urban applicability | Permitted local pilot or clearly identified public urban sample, with supported-species and failure-case reporting. |

The research target is to improve held-out-location mAP50–95 over a reproduced baseline, while preserving useful rare-class performance and controlling false positives. This is a target, not a claimed result. If improvements are not demonstrated, the strongest validated baseline will be retained and the comparison published.

The expected practical benefit is faster access to relevant wildlife observations and consistent review records. A small reviewer study will compare assisted review time and correction rate with manual review on the same sample. No numerical workload reduction, ecological benefit, or accuracy guarantee is claimed before measurement.

## 4. Target Users

- **Park managers and municipal environmental teams:** review observations across managed green spaces.
- **Conservation researchers and ecologists:** examine activity patterns and detection quality across locations.
- **Wildlife organizations and field staff:** prioritize image batches and inspect uncertain observations.
- **Students and AI researchers:** reproduce experiments and investigate camera-location generalization.

The initial tool serves reviewers and researchers. Urban planners would use reviewed summaries rather than act directly on unverified predictions. Activity counts do not directly estimate population size or identify individual animals.

## 5. Key Features / Deliverables

### Already present

Five research notebooks provide dataset preparation, training, evaluation, and experimental robustness code. The repository contains saved final test tables for three notebooks, prediction examples, and a night/IR augmentation diagnostic. Code generates YAML configurations, validation logs, exported checkpoints, and test metrics.

Dataset images and trained checkpoints are not distributed in the repository. Some experimental branches require fresh execution and validation before their results can be reported independently.

### Proposed development deliverables

1. **Reproduction package:** working environment, dataset instructions, baseline configuration, and run record.
2. **Robustness report:** controlled comparisons, per-class and cis/trans metrics, and disclosed data-access protocols.
3. **Validated model artifact:** checkpoint and inference instructions, distributed where applicable terms permit.
4. **Review demonstration:** batch image input, annotated detections, human review, and CSV export.
5. **Urban pilot report:** supported species, transfer limitations, failure cases, and reviewer effort on permitted imagery.
6. **Public documentation:** README, proposal, evidence notes, example predictions, and clear limitations on GitHub.

## 6. Implementation Plan & Timeline

The existing prototype is the starting point. Evidence packaging supports the **15 October 2026 registration deadline supplied by the applicant**. The following is an **indicative eight-week plan after selection or project commencement**, subject to the championship's official schedule and data availability.

| Period | Planned work | Milestone |
| --- | --- | --- |
| Week 1 | Audit dependencies, labels, category mapping, empty-frame policy, split integrity, and actual training arguments. | Verified setup and data audit. |
| Week 2 | Reproduce stock YOLO11s; archive checkpoint source, configuration, validation history, and test results. | Reference model and run record. |
| Weeks 3–4 | Compare attention/normalization and night/appearance augmentation independently; check transform execution and repeat seeds where feasible. | Controlled robustness comparison. |
| Week 5 | Verify DANN trainer integration and gradient flow; separate target adaptation from validation. | Independent adaptation result or documented omission. |
| Week 6 | Build batch inference, review, and CSV export; profile processing time on available hardware. | Working demonstration. |
| Week 7 | Evaluate permitted urban imagery, select thresholds using validation data, and measure review time and corrections. | Pilot findings. |
| Week 8 | Finalize model choice, results, documentation, examples, and demonstration walkthrough. | Development submission package. |

If field access is delayed, benchmark evaluation and a public-data demonstration can proceed. The fallback will be labeled and will not be described as a completed local deployment.

## 7. Resources or Requirements

- **Compute:** a CUDA-capable training GPU, such as a Kaggle Tesla T4 or comparable resource. Existing saved runs used a T4; inference requirements will be profiled.
- **Data:** CCT-20 images and official splits, followed by permitted urban images with suitable labels and site/time metadata. Local species may require new classes and annotation.
- **Software:** Python, Jupyter/Kaggle, PyTorch, Ultralytics, Albumentations, OpenCV, NumPy, pandas, PyYAML, pycocotools, tqdm, and Matplotlib. A lightweight interface framework will be chosen during development.
- **Storage/access:** space for images, labels, checkpoints, and artifacts; internet for initial downloads if not already attached locally.
- **Expert input:** an experienced reviewer or ecologist for a small validation sample and interpretation. No partner commitment is assumed.
- **Effort:** approximately eight weeks of part-time development, adjusted after baseline reproduction. A monetary budget is not fixed at registration.

## 8. Potential Challenges or Risks

| Challenge | Planned response |
| --- | --- |
| US benchmark species differ from local urban wildlife | Check taxonomy, annotate additional species where needed, and validate locally before operational claims. |
| Background bias persists | Preserve the baseline, compare changes independently, and select the strongest verified model. |
| Rare, small, or occluded animals are missed | Report per-class AP/support, inspect failures, and add labels where feasible. |
| A frame without detections may still contain an animal | Measure animal-presence recall and empty-frame false positives separately; avoid discarding unreviewed pilot images. |
| Synthetic night images do not match real cameras | Inspect previews and evaluate on actual labeled night imagery. |
| Dependency changes or inactive code paths undermine results | Pin an environment and verify transforms, trainer calls, feature dimensions, and gradient flow. |
| Adaptation contaminates validation | Separate adaptation, validation, and final test data and disclose protocol differences. |
| Repeated frames inflate counts | Group capture events and distinguish observation frequency from population estimates. |
| Incidental people or sensitive wildlife locations appear | Use permitted data, restrict raw pilot access, and redact personal imagery and sensitive precise locations from public outputs. |
| GPU time or field access is limited | Prioritize the baseline and one promising experiment; provide a clearly labeled public-data demonstration if needed. |

## 9. Additional Details: Evidence and Contribution

The following metrics are saved in the notebooks' final test outputs; they have not been rerun for this application.

| Notebook | cis_test mAP50–95 | trans_test mAP50–95 |
| --- | ---: | ---: |
| BASELINE.ipynb | 0.514975 | 0.321051 |
| arch_changed.ipynb | 0.514301 | 0.312920 |
| p2_arch.ipynb | 0.486633 | 0.304591 |

These results show a substantial transfer gap and a concrete research starting point. The baseline has the highest saved overall trans score here; a claim that the modified methods already outperform it would be unsupported.

The DANN notebook's saved training/test outputs repeat the architecture notebook's outputs and do not establish a separate DANN result. The IR notebook's training stream ends during epoch 28 of 30 and its final test cell has no saved output. Fresh runs are planned to resolve these gaps. See the [experiment evidence notes](https://github.com/jaja-ganj/DL-project-group35/blob/main/docs/EXPERIMENT_RESULTS.md).

The contribution is an implemented wildlife-detection workflow with explicit evaluation of location transfer, class imbalance, and night imagery, followed by a proposed urban review application. YOLO11, CBAM, DANN, and Fourier adaptation are established methods; the project does not claim to have invented them.

### Supporting materials and references

- [Project repository and notebooks](https://github.com/jaja-ganj/DL-project-group35).
- [Saved held-out-location predictions](https://github.com/jaja-ganj/DL-project-group35/blob/main/docs/assets/trans-test-predictions.png), extracted unchanged from `arch_changed.ipynb`.
- Beery, Van Horn, and Perona, [Recognition in Terra Incognita](https://arxiv.org/abs/1807.04975), ECCV 2018; [CCT access and terms](https://lila.science/datasets/caltech-camera-traps).
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
- Woo et al., [CBAM](https://arxiv.org/abs/1807.06521), ECCV 2018.
- Ganin et al., [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818), JMLR 2016.
- Yang and Soatto, [Fourier Domain Adaptation](https://arxiv.org/abs/2004.05498), CVPR 2020.

Dataset and third-party software terms apply to their materials. No project license file is currently included; future model/data distribution will account for applicable terms.
