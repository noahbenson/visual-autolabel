# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/cmd/_core.py
# Command-line interface implementation.

# Dependencies #################################################################

import sys
from pathlib import Path

import numpy as np
import torch
import neuropythy as ny
import nibabel as nib

from ._apply import apply_benson2025


# Configuration ################################################################

syntax_message = """Usage: python -m visual_autolabel <subject-path>

The visual_autolabel supports a command-line interface that applies the
convolutional neural network model that uses T1-weighted image data only from
Benson, Song, et al. (2025) to the FreeSurfer subject located at the given
`<subject-path>`. The path must point to a FreeSurfer subject directory, and the
results are saved to the files `<subject-path>/surf/lh.benson2025.mgz` and
`<subject-path>/surf/rh.benson2025.mgz`.

The following options are also accepted:
 * -h | --help
   Print this message and exit.
 * -v | --verbose
   Use verbose output.
 * -t <tag> | --tag=<tag>
   Instead of exporting files named, e.g., `lh.benson2025.mgz`, use
   file names such as `lh.<tag>.mgz`
 * -o <dir> | --output-dir=<dir>
   Instead of writing the files into the appropriate FreeSurfer directories,
   all output files are written to the given directory <dir>.
 * -r | --rings
   Instead of predicting visual area boundaries for V1, V2, and V3, predict the
   iso-eccentric rings between 0-0.5, 0.5-1, 1-2, 2-4, and 4-7 degrees of
   eccentricity.
 * -b | --both
   In addition to predicting visual area boundaries for V1, V2, and V3, also 
   predict the iso-eccentric rings between 0-0.5, 0.5-1, 1-2, 2-4, and 4-7
   degrees of eccentricity.
 * -a | --annot
   Export the results to a pair of FreeSurfer annotation files named
   `lh.benson2025.annot` and `rh.benson2025.annot`. The colors for V1, V2, and
   V3 in this case are red, green, and blue.
 * -V <template> | --volume=<template>
   Export the results to a volume file that uses the given `<template>`, which
   must be an MRImage (MGZ or Nifti format). The volume file is saved in
   `<subject-path>/mri/benson2025.mgz` or `<subject-path>/mri/benson2025.nii.gz`
   depending on the image type of the template image.
 * -x | --no-surface
   Do not export the surface MGZ files that are exported by default.
"""


# Code #########################################################################

def main(argv=None):
    """Implementation of the visual-autolabel command-line tools.

    `main()` executes the command-line tool using the arguments passed in the
    `sys.argv` list.

    `main(argv)` uses the list of arguments `argv` instead.

    The `main` function returns the return value that should be used as an exit
    status; however, it never uses `sys.exit` to end the process as that is
    considered the job of the calling frame.
    """
    # If argv is None, we use the sys.argv
    if argv is None:
        argv = sys.argv
    # Process the arguments; start with the defaults:
    args = dict(
        verbose=False, tag='benson2025', outputs='area',
        surface=True, annot=False, volume=None)
    targets = []
    # We always ignore argv[0], which is the command typically.
    arglist = argv[1:]
    while len(arglist) > 0:
        arg = arglist[0]
        arglist = arglist[1:]
        if arg == '-h' or arg == '--help':
            print(syntax_message)
            return 0
        elif arg == '-v' or arg == '--verbose':
            args['verbose'] = True
        elif arg == '-r' or arg == '--rings':
            args['outputs'] = 'ring'
        elif arg == '-b' or arg == '--both':
            args['outputs'] = 'both'
        elif arg == '-a' or arg == '--annot':
            args['annot'] = True
        elif arg == '-x' or arg == '--no-surface':
            args['surface'] = False
        elif arg in ('-t', '--tag', '-V', '--volume', '-o', '--output-dir'):
            if len(arglist) == 0:
                raise RuntimeError(f"option {arg} requires an argument")
            elif len(arglist[0]) == 0:
                raise RuntimeError(f"option {arg} cannot be empty")
            if arg == '-t' or arg == '--tag':
                args['tag'] = arglist[0]
            elif arg == '-V' or arg == '--volume':
                args['volume'] = arglist[0]
            else:
                args['outputdir'] = arglist[0]
            arglist = arglist[1:]
        elif arg.startswith('-t'):
            args['tag'] = arg[2:]
        elif arg.startswith('--tag='):
            args['tag'] = args[6:]
        elif arg.startswith('-V'):
            args['volume'] = arg[2:]
        elif arg.startswith('--volume='):
            args['volume'] = arg[9:]
        elif arg.startswith('-o'):
            args['outputdir'] = arg[2:]
        elif arg.startswith('--output-dir='):
            args['outputdir'] = arg[13:]
        else:
            # This is a freesurfer directory!
            targets.append(arg)
    # Make sure everything is fine:
    vol = args['volume']
    if vol is not None:
        vol = Path(vol)
        if not vol.is_file():
            raise RuntimeError(
                f"given volume template is not a file: {str(vol)}")
        elif str(vol).endswith('.nii') or str(vol).endswith('.nii.gz'):
            args['volume_format'] = 'nii.gz'
        elif any(str(vol).endswith(k) for k in ('.mgz', '.mgh', '.mgh.gz')):
            args['volume_format'] = 'mgz'
        else:
            raise RuntimeError(
                f"volume template must be a nifti or mgh/mgz file")
        vol = ny.load(args['volume'])
        args['volume'] = ny.image_copy(
            vol,
            dataobj=np.zeros(vol.shape, dtype=np.int32))
    if not args['surface'] and not args['volume'] and not args['annot']:
        raise RuntimeError(
            f"no output files requested: --no-surface must be used with either"
            f" --volume or --annot")
    if len(targets) == 0:
        raise RuntimeError("no subject path targets were given")
    # Print some details if verbose output is requested:
    if args['verbose']:
        print("Running Benson2025 model on the following subjects:")
        for k in targets:
            print("  *", k)
    # Okay, the arguments seem fine; let's apply the model.
    fsa = ny.freesurfer_subject('fsaverage')
    if args['verbose']:
        print("\nApplying model:")
    for targ in targets:
        # This is a reasonable check, but it's commented out because we want
        # the user to be able to specify a local output directory for a subject
        # in the cloud, like:
        # s3://openneuro.org/ds003787/derivatives/freesurfer/sub-wlsubj001
        #if not targ.is_dir():
        #    raise RuntimeError(
        #        f"FreeSurfer subject must be a directory: {str(targ)}")
        if args['verbose']:
            print(f"  * {str(targ)}")
        # Make sure we have a subject or subject path!
        try:
            sub = ny.freesurfer_subject(targ)
        except Exception:
            try:
                sub = ny.hcp_subject(targ)
            except Exception:
                try:
                    sub = ny.freesurfer_subject(targ, check_path=False)
                except Exception as e:
                    raise e.with_traceback(None)
        targpath = Path(sub.path)
        # Apply the model:
        outputs = args['outputs']
        outputs = ('area', 'ring') if outputs == 'both' else (outputs,)
        for output in outputs:
            (lh_labels, rh_labels) = apply_benson2025(sub, output)
            tag = args['tag'].format(subject=sub.name)
            # Write the outputs:
            if args['surface']:
                opath = Path(args.get('outputdir', targpath / 'surf'))
                filepath = str(opath / f'lh.{tag}_v{output}.mgz')
                if args['verbose']:
                    print(f"    Saving file: {filepath}")
                ny.save(filepath, lh_labels)
                filepath = str(opath / f'rh.{tag}_v{output}.mgz')
                if args['verbose']:
                    print(f"    Saving file: {filepath}")
                ny.save(filepath, rh_labels)
            if args['annot']:
                ctab = np.array(
                    [[0,0,0,0], [255,0,0,255], [0,255,0,255], [0,0,255,255]])
                names = ['none', 'V1', 'V2', 'V3']
                opath = Path(args.get('outputdir', targpath / 'label'))
                filepath = opath / f'lh.{tag}_v{output}.annot'
                if args['verbose']:
                    print(f"    Saving file: {filepath}")
                nib.freesurfer.io.write_annot(
                    filepath,
                    labels=lh_labels,
                    ctab=ctab,
                    names=names)
                filepath = opath / f'rh.{tag}_v{output}.annot'
                if args['verbose']:
                    print(f"    Saving file: {filepath}")
                nib.freesurfer.io.write_annot(
                    filepath,
                    labels=rh_labels,
                    ctab=ctab,
                    names=names)
            if args['volume'] is not None:
                # Convert to volume first:
                if args['verbose']:
                    print("    Converting surface labels into volume labels...")
                vol = sub.cortex_to_image(
                    (lh_labels, rh_labels),
                    args['volume'],
                    method='nearest')
                opath = Path(args.get('outputdir', targpath / 'mri'))
                filepath = opath / f'{tag}_v{output}.{args["volume_format"]}'
                filepath = str(filepath)
                if args['verbose']:
                    print(f"    Saving file: {filepath}")
                ny.save(filepath, vol)
    # That is all we do!
    if args['verbose']:
        print("visual-autolabel complete!")
    return 0
