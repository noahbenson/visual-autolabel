# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/hcp/train/_core.py


#===============================================================================
# Dependencies

import os, sys, json

from .._datasets import (
    make_dataloaders,
    HCPDataset
)
from .._core import (
    features as features,
    input_properties,
    output_properties)
from ....train import (
    train_until,
    load_training,
)
from ....util import (
    autolog)
from ....image import (
    UNet)

import argparse


# required, action,defatul
# subparsers = parser.add_subparsers(dest="command")


def main(args, /, exit_on_error=False):
    """Executes the training used by Benson et al. (2024).

    This function allows you to run the main function for the command line:
    ```bash
    $ python -m visual_autolabel.benson2025.hcp.train \
                <model_key> <options.json> <plan.json>
    ```
    The function should be called using the following syntax:
    `main([model_key, options_json_filename, plan_json_filename],**options)`.
    """

    p = argparse.ArgumentParser(description="Training CLI for hcp",exit_on_error=False)

    # positional
    p.add_argument("model_key",type=str,help="A string that should be appended, as a sub-directory name, to the `model_cache_path`; this argument allows one to save model training to a specific sub-directory of the `model_cache_path` directory.")

    p.add_argument("opts_filename",type=str,help="Location of opts json file")
    p.add_argument("plan_filename",type=str,help="Location of plan json file. The training-plan to pass to the `visual_autolabel.run_modelplan()`")

    p.add_argument("--multiproc", action="store_false",default=True, help="Whether to use multiple cpu threads")
    p.add_argument("--nthreads",  type=int, help="Whether to use multiple cpu threads")
    p.add_argument("--nice",      type=int, help="Whether to use multiple cpu threads")

    p.add_argument("--until",     type=int, default=None,help="Only `until` groups of trainings are performed, then the result is returned. If `None`, then the training continues until a `KeyboardInterrupt` is caught. The default is `None`.")
    p.add_argument("--num_epochs",type=int, default=10,help="Number of training epochs")
    p.add_argument("--batch-size",type=int, default=4, help="Batch training size")

    p.add_argument("--base_model",type=str,choices=["resnet18","resnet34"],help="what model to use")
    p.add_argument("--inputs",                type=str,choices=input_properties.keys(),default="anat",help="What inputs")
    p.add_argument("--outputs","--prediction",type=str,choices=output_properties.keys(),help="What to predict")
    p.add_argument("--raters",type=str,nargs='*',choices=output_properties.keys(),help="What to predict")

    p.add_argument("--partition",type=str,default="default",choices=["default", "trn", "val"], help="Data partation to use")

    p.add_argument("--mkdirs",              action="store_false",default=True,help="Whether to make directories if they dn't exist")
    p.add_argument("--mkdir_mode",          type=int, default=0o775,help="What mode to make directoreies")
    p.add_argument("--dataset_cache_path",  type=str, help="Where data is cached")
    p.add_argument("--model_cache_path",    type=str, help="Where model is cached")

    p.add_argument("--resume_type",type=str,default=None,choices=['new','continue','continue_if_exists','exit_if_exists'],help="continue training from previous run(s) if possible")

    p.add_argument("--hyper_load_if_exists",action="store_false",default=True,help="load optuna hyperparamter study if exists")
    p.add_argument("--hyper_n_trials",      type=int, default=10,help="Number of trials in optuna hyperparamter study")
    p.add_argument("--hyper_storage",       type=str, default=None,help="Storage destination for optuna hyperparameter study")
    p.add_argument("--hyper_sampler",       type=str, default=None,help="Sampler algorithm to use for optuna hyperparamter study")
    p.add_argument("--hyper_pruner",        type=str, default=None,help="Pruner algorithm to use for optuna hyperparamter study")
    p.add_argument("--hyper_study_name",    type=str, default=None,help="Name of study for optuna hyperparamter study")

    p.add_argument("--debug",    action="store_true", default=False,help="Print Debugging information")
    p.add_argument("--error_on_interrupt",    action="store_true", default=False,help="Print Debugging information")

    if exit_on_error:
        parsed_args = vars(p.parse_args(args))
    else:
        parsed_args = vars(p.parse_args(args, namespace=argparse.Namespace()))

    run(exit_on_error,parsed_args)

def run(exit_on_error,arg_opts):
    debug=arg_opts.pop('debug')
    resume_type=arg_opts.pop('resume_type')
    if resume_type is None:
        resume_type='new'
    error_on_interrupt=arg_opts.pop('error_on_interrupt')

    # Plan
    plan_filename = arg_opts.pop('plan_filename')
    plan_filename = os.path.expanduser(plan_filename)
    plan=HCPDataset.load_plan(plan_filename,exit_on_error=exit_on_error)


    # File opts
    opts_filename = os.path.expanduser(arg_opts.pop('opts_filename'))
    fil_opts=HCPDataset.load_opts(opts_filename,exit_on_error=exit_on_error,parse=False)

    # merge opts, arg_opts takes priority
    arg_opts={k:v for k,v in arg_opts.items() if v is not None}
    fil_opts={k:v for k,v in fil_opts.items() if v is not None}
    opts={**fil_opts,**arg_opts}

    # key
    model_key=opts.pop('model_key')

    # parse
    (inputs,outputs,opts)=HCPDataset.parse_opts(opts);

    # Make an auto-logger with a log-file.
    mcp = opts.get('model_cache_path')
    if mcp is None:
        log=None
    else:
        log_path = os.path.join(mcp, model_key, 'training.log')
        log = autolog(log_path, clear=resume_type.startswith('continue'))
    # If we have multiple copies of the model_key, ensure that they match.
    if 'model_key' in opts:
        if model_key == opts['model_key']:
            del opts['model_key']

    # rename hyperopts
    opts_copy=opts.copy()
    opts={}
    for k,v in opts_copy.items():
        if k.startswith("hyper_"):
            opts[k.split("_",1)[1]]=v
        else:
            opts[k]=v

    if debug:
        print()
        print('CLI ARGS')
        for k,v in arg_opts.items():
            print({k:v})
        print()
        print('FILE ARGS')
        for k,v in fil_opts.items():
            print({k:v})
        print()
        print('OPTS')
        for k,v in opts.items():
            print({k:v})
        print()
        print('PLAN')
        for i,d in enumerate(plan):
            print(i)
            for k,v in d.items():
                print({k:v})


    # Train the model.
    train_until(
       inputs, outputs, plan,
        model_key=model_key,
        model=UNet,
        dataloaders=make_dataloaders,
        features=features,
        logger=log,
        resume_type=resume_type,
        error_on_interrupt=error_on_interrupt,
        _debug=debug,
        **opts)
