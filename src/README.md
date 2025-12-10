# Visual Autolabel TODO List

These are code-cleanup goals for the Visual Autolabel Hybrid CNN project.

1. Clean up `scripts/optuna_hybridcnn`: put code for types and abstract stuff in
   `src/` library; keep training code and configuration code in the script.
2. Review the 3D-to-2D notebook (make sure it's ready for others to use it
   potentially).
3. Edit `_hybrid.py` and `_data3D.py` (in `images/`) to make them more like
   `_data2D.py`.
   * The `image/_hybrid.py` file should only contain dataset-independent code;
     anything that interacts directly with the HCP should go to the new module
     in `visual_autolabel/hybrid/`.
   * For example: there is an explicit list of HCP subjects in `_hybrid.py`, but
     the `visual_autolabel/images` directory is supposed to be
     dataset-independent (and `_data2D.py` is dataset independent). We should
     rewrite this to put dataset configuration details in a new module:
     `visual_autolabel.hybrid` (we will probably rename this later when we are
     ready to submit/publish).
