# `visual_autolabel.apply` package

This package of the `visual_autolabel` library is used to apply the
model of V1, V2, and V3 visual areas (in terms of anatomical structures
derived from a T1-weighted image) to a FreeSurfer subject.

To use this package, there are two interfaces:
1. Command line: `python -m visual_autolabel.apply <path-to-subject>`  
   In this case, the command saves files to the subjects `<path-to-subject>/surf/`
   directory. The saved files are `lh.visual_autolabel.mgz` and `rh.visual_autolabel.mgz`,
   and they contain labels for V1, V2, and V3 of each hemisphere.  
   (Initially low priority--we want to write the python interface first.)
2. Python:  
   ```python
   from visual_autolabel.apply import apply_model
   results = apply_model('<path-to-subject>')
   ```
   The `results` should be a tuple `(lh_labels, rh_labels)` for each
   hemisphere.  
   How to make this function:
   * First, look at `visual_autolabel.benson2024.nyu._datasets`: there is an
     `NYUImageCache` and an `NYUDataset` class. We can use these as templates
     for a new pair of classes that don't specifically load the NYU dataset
     and instead load FreeSurfer subjects.
   * We'll need to write these two classes: `FreeSurferImageCache` and
     `FreeSurferDataset` so that they work for a simple freesurfer subject.
     Ignore retinotopic mapping data for now--this will be just for the anatomical
     CNNs.  
     Start by just copying the NYU classes and rewrite/remove extra code.
   * Once we have ImageCache and Dataset classes that are similar to the
     NYU classes, then we can make a dataset with 1 subject. Once we have a dataset
     object `ds` with that one subject, we should be able to load a model and run
     something like this:
     ```python
     mdl = unet(
         'anat', 'area', 'model',
         model_cache_path=model_cache_path)
     # This code comes from line 199 of visual_autolabel/benson2024/nyu/_core.py
     labels = ds.predlabels({'subject':sid}, mdl, view=view, labelsets=labelsets)
     ```
     
