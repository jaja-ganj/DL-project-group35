# DL-project-group35

Wildlife monitoring notebooks for training and evaluating YOLO11-based object detection models on the Caltech Camera Traps dataset. The experiments focus on converting COCO-style camera trap annotations into YOLO format, reducing location and background bias, and comparing cis/trans split performance.

## Notebooks

| Notebook | Description |
| --- | --- |
| `BASELINE.ipynb` | Stock YOLO11s baseline for the Caltech Camera Traps dataset. It keeps the same data conversion, split setup, augmentations, weighted classification loss, and evaluation workflow as the modified runs so the architecture comparison stays fair. |
| `arch_changed.ipynb` | Bias-reduced YOLO11 pipeline with a custom architecture using early Instance Normalization and CBAM attention blocks. It prepares all CCT splits, trains with `trans_val` as the main validation split, and evaluates on both cis and trans test sets. |
| `arch_dann.ipynb` | DANN-style domain adaptation experiment for YOLO11. It adds a gradient reversal layer and domain discriminator so the model learns features that are less tied to camera-location-specific backgrounds. |
| `p2_arch.ipynb` | Second architecture-focused bias-reduction notebook. It uses custom YOLO11 modules, class weighting, and stronger location-decoupling augmentations such as photometric transforms, safe cropping, cutout-style occlusion, mosaic, and cutmix. |
| `IR_best_simpl.ipynb` | Simplified best-run notebook with an IR/night-condition augmentation recipe. It keeps the full CCT conversion, custom YOLO11 setup, weighted loss, warm-up freezing, cis/trans evaluation, and qualitative prediction visualization. |

## Notes

- The notebooks are written for a Kaggle-style environment and reference Kaggle dataset paths.
- Training is based on Ultralytics YOLO11.
- Most experiments compare performance across `cis_*` and `trans_*` splits to measure generalization across camera locations.
