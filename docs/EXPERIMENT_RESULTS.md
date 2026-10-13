# Experiment Results and Evidence Notes

These metrics were read from notebooks at source revision `30dbeea` of [DL-project-group35](https://github.com/jaja-ganj/DL-project-group35). They summarize outputs already saved in Git; training and evaluation were not rerun during documentation preparation.

## Recorded final test metrics

Values use the 0–1 scale and describe detection, rather than classification accuracy.

| Notebook | Split | mAP50 | mAP50–95 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| [BASELINE.ipynb](../BASELINE.ipynb) | cis_test | 0.778203 | 0.514975 | 0.751980 | 0.742706 |
| BASELINE.ipynb | trans_test | 0.486425 | 0.321051 | 0.517390 | 0.500686 |
| [arch_changed.ipynb](../arch_changed.ipynb) | cis_test | 0.783622 | 0.514301 | 0.734642 | 0.761439 |
| arch_changed.ipynb | trans_test | 0.484752 | 0.312920 | 0.510542 | 0.493144 |
| [p2_arch.ipynb](../p2_arch.ipynb) | cis_test | 0.769448 | 0.486633 | 0.735096 | 0.741782 |
| p2_arch.ipynb | trans_test | 0.484081 | 0.304591 | 0.573965 | 0.480153 |

**Evidence locations:** final `eval_df` output in cell 25 of `BASELINE.ipynb`, cell 34 of `arch_changed.ipynb`, and cell 36 of `p2_arch.ipynb`, counting all cells from 1. Edits can change these numbers. The code writes `test_metrics.csv` under each artifact directory; those CSVs and checkpoints are not committed in the reviewed revision.

## Interpretation

The baseline has the highest saved overall `trans_test` mAP50–95 here. The custom architecture is lower by 0.008131, approximately **0.81 percentage points**. The `p2_arch` result is lower by 0.016460, approximately **1.65 percentage points**.

| Notebook | cis minus trans mAP50–95 |
| --- | ---: |
| BASELINE.ipynb | 0.193924 |
| arch_changed.ipynb | 0.201381 |
| p2_arch.ipynb | 0.182042 |

The smaller `p2_arch` gap accompanies lower scores on both splits, so does not establish improved transfer. A future report should include per-class AP and support, animal-only summaries, and empty-frame false positives.

These runs support a measurable generalization problem and an implemented benchmark workflow. They do not establish superiority, statistical significance, or validated urban deployment. Notebook outputs can survive subsequent source edits, so provenance and fresh execution are needed for controlled comparisons.

## Experimental evidence gaps

### arch_dann.ipynb

Source defines gradient reversal, a domain discriminator, and unlabeled target-image loading from `trans_val`. Its saved training stream and final table are identical to the corresponding outputs in `arch_changed.ipynb`; these are excluded as independent DANN evidence.

A new run needs verification that the trainer calls the adversarial step, discriminator dimensions match the backbone scale, and domain gradients reach the backbone. Separate target adaptation from checkpoint-selection validation and final test data.

### IR_best_simpl.ipynb

Saved output contains transform previews, pretrained mapping, and training logs. Training ends during epoch 28 of 30; final test evaluation has no saved output. No final IR test score is reported, and the filename does not establish a best-model ranking.

### p2_arch.ipynb

Source includes Fourier style transfer, night/IR simulation, and a `WildlifeBiasTrainerV2` suppression path. Fresh training selects `WildlifeBiasTrainer`, so optional box-aware suppression is not active on that path. Saved metrics cannot be attributed to every current augmentation without a new execution record.

## Reproduction plan

1. Record dataset files, class mapping, package versions, revision, seed, and checkpoint provenance.
2. Audit labels, empty-frame policy, splits, and transform previews; resolve ignored arguments.
3. Reproduce the baseline with a unique run name and actual output directory.
4. Compare changes independently with matched compute/initialization; report adaptation under its own data-access protocol.
5. Run multiple seeds where feasible; export metrics, per-class results, checkpoints, and configurations.
6. Select using validation data, evaluate reserved test sets, and validate separately on permitted urban imagery.

## Image provenance

[`assets/trans-test-predictions.png`](assets/trans-test-predictions.png) is an unchanged PNG from the saved prediction output of cell 36 in `arch_changed.ipynb`. Underlying images come from [CCT](https://beerys.github.io/CaltechCameraTraps/). It illustrates model output, not an urban pilot or a substitute for full-test metrics.
