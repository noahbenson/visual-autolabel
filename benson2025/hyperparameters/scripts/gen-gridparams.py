#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates options files for use with hyperparameter training.

This script should be run with either 1 or 2 arguments. The first argument must
always be the `grid-options.json` file (see the `grid-search` directory), which
specifies what parameters are stable versus hyperparameters. The script
generates the option (JSON) files that are used in the hyperparameter search. If
an integer is provided as the second argument, then the script produces a single
pair of files in the current directory: `opts.json` and `plan.json`, for use
with the `train.py` script. If provided only a single argument, then all
possible files are generated with names `opts_<>.json` and `plan_<>.json` where
the `<>` is replaced with an integer number.
"""


#===============================================================================
# Dependencies

import os, sys, json


#===============================================================================
# Parse the arguments and load the parameters.

args = sys.argv[1:]
if len(args) < 1:
    raise RuntimeError(f"grid-params.json argument missing")
elif len(args) == 1:
    justel = None
elif len(args) == 2:
    justel = int(args[1])
else:
    raise RuntimeError("only 1 or 2 parameters accepted")
paramfl = args[0]
with open(paramfl, 'rt') as fl:
    opts = json.load(fl)
if not (len(opts) == 2 and 'hyper' in opts and 'static' in opts):
    raise RuntimeError("grid-params file must contain 'hyper' and 'static'")
hyper = opts['hyper']
static = opts['static']


#===============================================================================
# Functions to convert the hyper- and static-params into opts and plan data.

def bce_weight_into_plan(bce_weight, plan):
    plan[0]['bce_weight'] = bce_weight
    plan[1]['bce_weight'] = bce_weight / 2
    plan[2]['bce_weight'] = 0
def lr_into_plan(lr, plan):
    plan[0]['lr'] = lr
    plan[1]['lr'] = lr * 2/3
    plan[2]['lr'] = lr * 1/3
def genjsons(hyper, static, n):
    """Generates and returns json-ready data structures for a grid search cell.

    The return value is `(opts_data, plan_data)`.
    """
    plan = [{}, {}, {}]
    opts = {}
    # We're going to want to add a tag to the model; go ahead and calculate it.
    opts["model_key"] = "grid%05d" % (n,)
    # Okay, now we process the hyper-options; first, figure out which of each
    # of the options we are going to use; this basically becomes a static option
    # for this run.
    static = dict(static)
    for (k,v) in hyper.items():
        # Figure out which of these parameters we need and add it to static.
        ii = n % len(v)
        static[k] = v[ii]
        n //= len(v)
    if n != 0:
        raise IndexError(n)
    # Now that we've processed the hyper options into static options, just go
    # through the static options and put them in the opts/plan data structures.
    for (k,v) in static.items():
        if k == 'bce_weight':
            bce_weight_into_plan(v, plan)
        elif k == 'lr':
            lr_into_plan(v, plan)
        else:
            # Literally everything else goes into the options.
            opts[k] = v
    return (opts, plan)
def genfiles(hyper, static, n, suffix):
    """Generates and saves JSON files for the opts and plan for a grid cell.

    The `hyper` and `static` parameters must be from the `grid-params.json`
    file; the `n` must be the cell number to generate, and the `suffix` must be
    either `False` (no filename suffix) or `True` (the `model_key` is used as
    the filename suffix).
    """
    # First generate the opts and plan.
    try:
        (opts, plan) = genjsons(hyper, static, n)
    except IndexError:
        return False
    # Then the filenames.
    if suffix:
        optsfn = f"opts_{opts['model_key']}.json"
        planfn = f"plan_{opts['model_key']}.json"
    else:
        optsfn = "opts.json"
        planfn = "plan.json"
    # Now write out the files.
    with open(optsfn, 'wt') as fl:
        json.dump(opts, fl)
    with open(planfn, 'wt') as fl:
        json.dump(plan, fl)
    # That's it!
    return True


#===============================================================================
# Generate the actual files!

if justel is None:
    # We are generating alllll of the files; we generate them into the current
    # directory.
    n = 0
    while genfiles(hyper, static, n, True):
        n += 1
else:
    # We generate just the one file.
    if not genfiles(hyper, static, justel, False):
        raise RuntimeError(f"requested index {n} is invalid")

# That's all that this script does.
sys.exit(0)
