# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/train/_core.py
# Implementation of utilities for training the CNN models of the
# visual_autolabel library.

#===============================================================================
# Constants / Globals

import sys
import os
import time
import copy
import inspect
import json
from collections.abc import Mapping
from numbers import Number

import torch
import neuropythy as ny

from ..util import (
    is_partition,
    partition as make_partition,
    partition_id,
    trndata,
    valdata,
    loss as calc_loss,
    autolog,
    filter_options)
from ..image import (
    UNet)

#-------------------------------------------------------------------------------
# Logging
# These are strings that encapsulate the printing of training information to the
# terminal or to a log-file.

log_header_format = "%-5s  %-7s  %-5s   %-8s  %-8s  %-8s   %-8s  %-8s  %-8s"
log_format = " | ".join(["%2d/%2d  %7.5f  %5.1f", 
                         "%8.3f  %8.3f  %8.3f",
                         "%8.3f  %8.3f  %8.3f %s"])
log_header = log_header_format % (
    "epoch", "  lr   ", "dt[s]",
    "trn bce ", "trn dice", "trn loss",
    "val bce ", "val dice", "val loss")
log_header_hline = log_header_format % (
    "="*5, "="*7, "="*5, "="*8, "="*8, "="*8, "="*8, "="*8, "="*8)
log_header_hline = log_header_hline.replace(' ', '-')
log_header_hline = "%s+%s+%s" % (
    log_header_hline[:22], log_header_hline[23:53], log_header_hline[54:])


#===============================================================================
# Training Functions

#-------------------------------------------------------------------------------
# Logging

def log_epoch(metrics, epochno=None, epochmax=None, lr=None, dt=None,
              logger=print, endl=""):
    """Logs a single epoch of model-training.

    `log_epoch(metrics, epochno, lr, dt)` logs an epoch's metrics, which must be
    a dict-like object containing the keys `'trn'` and `'val'`, the values for
    each of which must also be a dict-like object whose keys are `'dice'`,
    `'bce'`, and `'loss'`.  If metrics is None, then a header is printed. If
    metrics is Ellipsis, then a separator line is printed.

    Parameters
    ----------
    metrics : dict-like or None or Ellipse
        A dict-like object of the metrics to report.
    epochno : int or None, optional
        The epoch number.
    epochmax : int or None, optional
        The maximum number of epochs that are being run.
    lr : float or None, optional
        The learning-rate of the current epoch.
    dt : float or None, optional
        The amount of time passed during the current epoch.
    logger : function or None, optional
        The logging function to use. If `None`, then nothing is logged. The
        default is `print`.
    endl : str or None, optional
        The end-of-line string to use when printing. For logging functions like
        `print`, which automatically append a newline, this can be `None` or
        just `""`.
    
    Returns
    -------
    None
    """
    if logger is None: return None
    if epochno is None: epochno = 0
    if epochmax is None: epochmax = 0
    if lr is None: lr = 0
    if dt is None: dt = 0
    if endl is None: endl = ""
    if metrics is None:
        logger(log_header + endl)
        logger(log_header_hline + endl)
    elif metrics is Ellipsis:
        logger('-' * len(log_header) + endl)
    else:
        t = metrics['trn']
        v = metrics['val']
        tup = ((epochno+1, epochmax, lr, dt) +
               tuple([t[q] for q in ['bce','dice','loss']]) +
               tuple([v[q] for q in ['bce','dice','loss']]) +
               (endl,))
        logger(log_format % tup)
    return None

#-------------------------------------------------------------------------------
# Training

def train_model(model, optimizer, scheduler, dataloaders,
                num_epochs=10,
                cache_path=None,
                device=None,
                hlines=False,
                logger=print,
                endl='',
                logits=None,
                bce_weight=0.5,
                reweight=True,
                smoothing=1):
    """Trains and returns a model based on the various optional arguments.
    
    `train_model(model, optimizer, scheduler, dataloaders)` runs training on the
    given model and returns the newly trained model. This represents a single
    set of epochs with a single `StepLR` decay.

    Parameters
    ----------
    model : PyTorch module
        The PyTorch model to be trained.
    optimizer : PyTorch Optimizer
        The PyTorch optimizer to use for training.
    scheduler : PyTorch Scheduler
        The PyTorch scheduler to be used (for example, PyTorches `StepLR`
        scheduler).
    dataloaders : mapping of DataLoader objects
        A mapping whose keys are `'trn'` and `'val'` and whose values are the
        PyTorch `DataLoader` objects associated with the training and validation
        datasets.
    num_epochs : int, optional
        The number of training epochs to run (default: 10).
    cache_path : str or None, optional
        If provided, the model and optimizer state will be pickled to files in
        this directory at the end of every epoch.
    device : str or None, optional
        The device to use for PyTorch training. If `None` (the default), then
        uses CUDA if CUDA is available and otherwise uses CPU.
    hlines : boolean, optional
        Whether to print horizontal lines in the logger for clarity (default:
        False).
    logger : function or None, optional
        The logging function to use. If `None`, then nothing is logged. The
        default is `print`.
    endl : str or None, optional
        The end-of-line string to use when printing. For logging functions like
        `print`, which automatically append a newline, this can be `None` or
        just `""`.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    bce_weight : float, optional
        The weight to give the BCE-based loss; the weight for the 
        dice-coefficient loss is always `1 - bce_weight`. The default is `0.5`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    smoothing : number, optional
        The smoothing coefficient `s` to use with the dice-coefficient liss.
        The default is `1`.

    Returns
    -------
    tuple
        A 3-tuple of `(trained_model, best_loss, best_dice_loss)`.
    """
    # Parse some arguments.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if endl is None:
        endl = ""
    # We need to keep track of the best model weights: start with the current
    # model weights.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_dice = 1e10
    # Log the header.
    log_epoch(None, logger=logger, endl=endl)
    # Now, for each epoch...
    for epoch in range(num_epochs):
        since = time.time()
        allmetrics = {}
        savestr = ""
        lr0 = optimizer.param_groups[0]['lr']
        # Each epoch has a training and validation phase
        for phase in ['trn', 'val']:
            if phase == 'trn':
                model.train()  # Set model to training mode.
            else:
                model.eval()   # Set model to evaluate mode.
            metrics = {}
            epoch_samples = 0
            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients.
                optimizer.zero_grad()
                # Calculate the forward model.
                with torch.set_grad_enabled(phase == 'trn'):
                    outputs = model(inputs.float())
                    loss = calc_loss(outputs, labels,
                                     logits=logits,
                                     bce_weight=bce_weight,
                                     smoothing=smoothing,
                                     reweight=reweight,
                                     metrics=metrics)
                    # backward + optimize only if in training phase
                    if phase == 'trn':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                # statistics
                epoch_samples += inputs.size(0)
            for k in metrics.keys():
                metrics[k] /= epoch_samples
            epoch_loss = metrics['loss']
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    savestr = "*"
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if metrics['dice'] < best_dice:
                    best_dice = metrics['dice']
            allmetrics[phase] = metrics
        time_elapsed = time.time() - since
        log_epoch(allmetrics, epoch, num_epochs, lr0, time_elapsed,
                  endl=savestr, logger=logger)
        if hlines: log_epoch(Ellipsis, logger=logger, endl=endl)
        if cache_path is not None:
            torch.save(model.state_dict(), 
                       os.path.join(cache_path, "model%06d.pt" % epoch))
            torch.save(optimizer.state_dict(), 
                       os.path.join(cache_path, "optim%06d.pt" % epoch))
    if logger is not None:
        logger('Best val loss: {:4f}'.format(best_loss) + endl)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return (model, best_loss, best_dice)

def _make_dataloaders_and_model(
        # We require that the dataloaders argument be provided.
        dataloaders,
        # We also require the model argument.
        model,
        # The rest is just options that we will filter to the dataloader and
        # model functions--if they are functions and not preconstructed models
        # or dataloaders.
        **kwargs):
    # First, make the dataloaders.
    if not is_partition(dataloaders):
        # Dataloaders must be a function if it's not a partition.
        if not callable(dataloaders):
            raise TypeError(
                'dataloaders must be a partition of DataLoader objects or a'
                ' callable that constructs such partitions')
        dataloader_opts = filter_options(dataloaders, **kwargs)
        dataloaders = dataloaders(**dataloader_opts)
    dl_trn = trndata(dataloaders)
    dl_val = valdata(dataloaders)
    # Next, we make the starting model.
    if isinstance(model, torch.nn.Module):
        start_model = model
    else:
        if model is Ellipsis:
            model = UNet
        model_opts = filter_options(model, **kwargs)
        start_model = model(
            dl_trn.dataset.feature_count,
            dl_trn.dataset.segment_count,
            **model_opts)
    # See if we need to initialize the weights.
    if init_weights is not None:
        from pathlib import Path
        if isinstance(init_weights, (str, Path)):
            weights = torch.load(init_weights)
        else:
            # otherwise, assume the init_weights are the weights dictionary
            weights = init_weights
        start_model.load_state_dict(weights)
    return (dataloaders, start_model)
def build_model(
        # Required options for building the initial model and the dataloaders.
        dataloaders, model,
        # Step 3: Optimizer and Scheduler.
        lr=0.004,
        step_size=None,
        gamma=0.90,
        # Step 4: Training.
        nthreads=Ellipsis,
        nice=10,
        num_epochs=10,
        model_cache_path=None,
        device=None,
        hlines=False,
        logger=print,
        endl='',
        bce_weight=0.5,
        reweight=True,
        smoothing=1,
        # All remaining options are filtered into dataloaders and model.
        **kwargs):
    """Creates, trains, and returns a PyTorch model.

    `build_model()` encapsulates the creation of the PyTrch model and the model
    `DataLoader` objects as well as the `train_model()` function. Given a set of
    parameters for all of these functions, runs a single batch of epochs and
    returns the trained model.

    The first step in this function is construction of the PyTorch `DataLoader`
    objects for the model training. The `dataloaders` option (default: `None`)
    can either be a partition (see `is_partition`) of training and validation
    dataloader objects or `None`, in which case the `make_dataloaders()`
    function is used, and the options `features`, `partition`,
    `data_cache_path`, `image_size`, `datasets`, `shuffle`, and `batch_size` are
    all passed to it. If `make_dataloaders` is not used, then these arguments
    are ignored.

    Next is the construction of the model itself. If a PyTorch model is given
    for the `model` parameter (below), then this step is skipped. Otherwise, the
    `model` parameter must be a callable or `None`, in which case `UNet` is
    used. The callable is called with the arguments `feature_count`,
    `class_count`, `pretrained_resnet`, `middle_branches`, and `apply_sigmoid`,
    and the return value must be a PyTorch `Module` to use as the model.

    Next, the optimizer and scheduler are created. The optimizer is always the
    PyTorch `Adam` optimizer, and the scheduler is always the `StepLR` type.
    The learning rate parameter (`lr`) is passed to the optimizer, and the
    `step_size` and `gamma` parameters are passed to the optimizer.
    Additionally, the number of PyTorch threads is set to `nthreads`, and the
    OS's nice value is set to `nice`.

    Finally, the model is trained using the `train_model()` function. The
    optional parameters num_epochs`, `model_cache_path`, `device`, `hlines`,
    `logger`, `endl`, `bce_weight`, `reweight`, and `smoothing` are passed to
    this function.

    Parameters
    ----------
    dataloaders : partition of DataLoaders or function
        The PyTorch `DataLoader` objects to use. If given a function that
        constructs the dataloaders, then they are created by providing this
        function with all keyword arguments passed to `build_model` that the
        function accepts. Alternatively, `dataloaders` may be a valid partition
        of a training and validation dataloader (e.g., a dict with keys `'trn'`
        and `'val'` or a tuple of `(trn, val)`).
    model : PyTorch Module or function
        The PyTorch `Module` (model) object to train, or a function that
        generates such a model. If a function is provided, then it must accept
        as its first two arguments the number of input features that are being
        used in training and the number of segments being predicted; beyond
        these positional arguments, all options passed to `build_model` that the
        function accepts are forwarded to it.
    lr : float, optional
        The initial learning rate for the optimizer. The default is 0.004.
    step_size : int or None, optional
        The step size for the scheduler. If `None`, then the size of the
        training dataset is used (i.e., one step per epoch).
    gamma : float or None, optional
        The rate of exponential decay in the learning rate; the default is
        0.9.
    nthreads : int or None, optional
        The number of PyTorch threads to use during training. If `None`, then
        the number of threads is not changed. If a negative number is given,
        the the number of CPUs plus this number is used. If `Ellipsis` is given
        (the default), then all CPUs are used.
    nice : int or None, optional
        If not `None`, then sets the OS's nice value for the process to this
        number before training the model. The default is 10.
    num_epochs : int, optional
        The number of training epochs to run (default: 10).
    model_cache_path : str or None, optional
        If provided, the model and optimizer state will be pickled to files in
        this directory at the end of every epoch.
    device : str or None, optional
        The device to use for PyTorch training. If `None` (the default), then
        uses CUDA if CUDA is available and otherwise uses CPU.
    hlines : boolean, optional
        Whether to print horizontal lines in the logger for clarity (default:
        False).
    logger : function or None, optional
        The logging function to use. If `None`, then nothing is logged. The
        default is `print`.
    endl : str or None, optional
        The end-of-line string to use when printing. For logging functions like
        `print`, which automatically append a newline, this can be `None` or
        just `""`.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    bce_weight : float, optional
        The weight to give the BCE-based loss; the weight for the 
        dice-coefficient loss is always `1 - bce_weight`. The default is `0.5`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    smoothing : number, optional
        The smoothing coefficient `s` to use with the dice-coefficient loss.
        The default is `1`.
    **kwargs
        All additional arguments are passed along to either the `dataloaders`
        or the `model` function. If either of these arguments is a function that
        creates the required object, then options accepted by that function are
        automatically passed to it.

    Returns
    -------
    tuple
        A 3-tuple of `(trained_model, best_loss, best_dice_loss)`.

    """
    allopts = dict(
        kwargs,
        lr=lr, step_size=step_size, gamma=gamma,
        nthreads=nthreads, nice=nice, num_epochs=num_epochs,
        model_cache_path=model_cache_path, device=device,
        hlines=hlines, logger=logger, endl=endl,
        bce_weight=bce_weight, reweight=reweight,
        smoothing=smoothing)
    # First, create the dataloaders and starting model.
    (dataloaders, start_model) = _make_dataloaders_and_model(
        dataloaders, model,
        **allopts)
    # Next, create the optimizer.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, start_model.parameters()),
        lr=lr)
    # Next, create the scheduler.
    if step_size is None: step_size = len(dataloaders['trn'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma)
    # Prepare for optimization.
    if nthreads is not None:
        import os
        if nthreads is Ellipsis:
            nthreads = os.cpu_count()
        elif nthreads < 0:
            nthreads = os.cpu_count() + nthreads
        torch.set_num_threads(nthreads)
    if nice is not None:
        os.nice(nice)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_model = start_model.to(device)
    return train_model(
        start_model, optimizer, scheduler, dataloaders,
        smoothing=smoothing,
        num_epochs=num_epochs,
        cache_path=model_cache_path,
        logger=logger,
        bce_weight=bce_weight,
        reweight=reweight, 
        device=device,
        hlines=hlines)

def run_modelplan(modelplan, dataloaders, model, **kwargs):
    """Executes a model-plan, which builds and trains a model.

    The `run_modelplan` is intended as a way to build and train a model using a
    number of rounds of training, each consisting of a single call to the
    `train_model` function. The first argument, `modelplan` must be an iterable
    of dict-like objects. Each dictionary contains specific parameters for a
    single call to `build_model`.

    The initial model is built using the optional parameters passed to this
    function, along with the initial dataloaders. These are then passed from
    training round to training round. If a parameter for the `make_dataloaders`
    function appears in an entry of the `modelplan`, then the dataloaders are
    replaced with new dataloaders using these parameters (which overwrite this
    function's parameters).

    Additionally, `run_modelplan` catches `KeyboardInterrupt` exceptions and
    returns the current best model.
    """
    # First, create the dataloaders and the model.
    make_opts = filter_options(_make_dataloaders_and_model, **kwargs)
    # We want to eliminate these options--in the future we will just pass the
    # model and dataloaders.
    for k in make_opts.keys():
        del kwargs[k]
    # Now make the dataloaders:
    make_opts = dict(
        {'dataloaders':dataloaders, 'model':model},
        **make_opts)
    (dataloaders, model) = _make_dataloaders_and_model(**make_opts)
    # Save the original arguments (with dataloader/model arguments fixed).
    kwargs = dict(
        {'dataloaders':dataloaders, 'model':model},
        **kwargs)
    # Extract a few options we need...
    mkdirs = kwargs.get('mkdirs', True)
    mkdir_mode = kwargs.get('mkdir_mode', 0o775)
    # We also use the model cache path and logger if they are provided.
    model_cache_path = kwargs0.get('model_cache_path', None)
    # Prepare for the rounds of training.
    best_dice = 1e10
    best_loss = 1e10
    best_mdl = model # We track best model by dice, not combined loss.
    best_mdl_wts = copy.deepcopy(model.state_dict())
    for (ii,step_opts) in enumerate(modelplan):
        # The new instructions are a the passed options, overwritten by the
        # specific model plan step options.
        opts = dict(kwargs)
        opts.update(step_opts)
        logger = opts.get('logger', print)
        # Note that we do not support regeneration of dataloaders or the model
        # when their parameters are passed in as step options; provide a warning
        # in such a case.
        step_make_opts = filter_options(_make_dataloaders_and_model, step_opts)
        if len(step_make_opts) > 0:
            from warnings import warn
            warn(
                f"step options contain dataloader/model options, which are"
                f" ignored after initialization:"
                f" {sorted(step_make_opts.keys())}")
        # Print a blank line between rounds.
        if ii > 0 and logger is not None:
            logger("")
        # Update the cache-path if we have one cache path.
        if model_cache_path is not None:
            cpath = os.path.join(model_cache_path, f'round{ii+1:02d}')
            if mkdirs and not os.path.isdir(cpath):
                os.makedirs(cpath, mode=mkdir_mode, exist_ok=True)
            kw['model_cache_path'] = cpath
        # Run the build-model function.
        (model, loss, dice) = build_model(**opts)
        if dice < best_dice:
            best_dice = dice
            best_mdl = model
            best_mdl_wts = copy.deepcopy(model.state_dict())
            best_loss = loss
    best_mdl.load_state_dict(best_mdl_wts)
    return (best_mdl,best_loss,best_dice)

def train_until(in_features, out_features, training_plan,
                model_key=None,
                until=None,
                mkdirs=True,
                mkdir_mode=0o775,
                **kwargs):
    """Continuously runs the given training plan for models until an interrupt.
        
    Runs training on `'anat'`, `'func'`, and `'both'` models, sequentially,
    using random partitions until a keyboard interrupt is caught, at which
    point a `pandas` dataframe of the results is returned. The partition is
    generated only once per group of model trainings (i.e., per training of an
    anatomical, functional, and combined model).
    
    Parameters
    ----------
    training_plan : list of dicts
        The training-plan to pass to the `visual_autolabel.run_modelplan()`
        function.
    model_key : str or None, optional
        A string that should be appended, as a sub-directory name, to the
        `model_cache_path`; this argument allows one to save model training
        to a specific sub-directory of the `model_cache_path` directory.
    until : int or None, optional
        If an integer is provided, then only `until` groups of trainings are
        performed, then the result is returned. If `None`, then the training
        continues until a `KeyboardInterrupt` is caught. The default is `None`.
    mkdirs : boolean, optional
        Whether to create cache directories that do not exist (default `True`).
    mkdir_mode : int, optional
        What mode to use when creating directories (default: `0o755`).
    logger : function or None, optional
        The logging function to use. If `None`, then nothing is logged. The
        default is `print`.
    **kwargs
        Additional named parameters are passed along to the `run_modelplan`
        function; these must at a minimum provide for a dataset partition and
        a model.
    """
    logger = kwargs.get('logger', print)
    if model_key is not None:
        if model_cache_path is not None:
            model_cache_path = os.path.join(model_cache_path, model_key)
            if mkdirs and not os.path.isdir(model_cache_path):
                os.makedirs(model_cache_path, mkdir_mode)
    if isinstance(in_features, str):
        in_features = (in_features,)
    if isinstance(in_features, (tuple, list, set)):
        in_features = {'inputs': tuple(in_features)}
    if isinstance(in_features, Mapping):
        for (k,feats) in in_features.items():
            if isinstance(feats, str):
                feats = (feats,)
            if (not isinstance(feats, (tuple, list, set)) or
                not all(isinstance(f, str) for f in feats)):
                raise ValueError(f"invalid features: {feats}")
            in_features[k] = tuple(feats)
    else:
        raise ValueError("in_features must be a list, set, dict, str, or tuple")
    if isinstance(out_features, str):
        out_features = (out_features,)
    elif isinstance(out_features, (tuple, list, set)):
        out_features = tuple(out_features)
    else:
        raise ValueError("out_features must be a list, set, str, or tuple")
    partition = kwargs.get('partition', Ellipsis)
    if partition is Ellipsis:
        from ..config import default_partition
        partition = default_partition
    # Options we need for below.
    lr = kwargs.get('lr', 0.004)
    gamma = kwargs.get('gamma', 0.9)
    batch_size = kwargs.get('batch_size')
    num_epochs = kwargs.get('num_epochs', 10)
    if model_cache_path is not None:
        if not os.path.isdir(model_cache_path) and mkdirs:
            os.makedirs(model_cache_path, mkdir_mode)
        # Go ahead and save the plan out to a json.
        with open(os.path.join(model_cache_path, "plan.json"), "wt") as fl:
            json.dump(training_plan, fl)
        # And the options dictionary.
        with open(os.path.join(model_cache_path, "options.json"), "wt") as fl:
            opts = dict(
                until=until, model_key=model_key, base_model=base_model,
                partition=partition, lr=lr, batch_size=batch_size,
                pretrained=pretrained, num_epochs=num_epochs,
                feature_names=(list(features.keys()) if features else None))
            json.dump(opts, fl)
    if dataset_cache_path is not None:
        if not os.path.isdir(dataset_cache_path) and mkdirs:
            os.makedirs(dataset_cache_path, mkdir_mode, exist_ok=True)
    training_history = []
    try:
        if logger:
            logger('')
        iterno = 0
        while True:
            if until is not None and iterno >= until:
                break
            iterno += 1
            # Make one partition for all three minimization types.
            if partition is None:
                # We regenerate the partition each round; sids is a required
                # option in this case.
                from ..config import default_partition
                part = make_partition(kwargs['sids'], how=default_partition)
            elif is_partition(partition):
                (trn,val) = (trndata(partition), valdata(partition))
                if isinstance(trn, Number) and isinstance(val, Number):
                    part = make_partition(kwargs['sids'], how=partition)
            else:
                part = make_partition(kwargs['sids'], how=partition)
            pid = partition_id(part)
            if logger:
                els = ('Iteration %d' % iterno, 'Partition ID: %s' % pid)
                logger('%-15s%70s' % els)
                logger('=' * 85)
            for (dnm,infeats) in in_features.items():
                if logger:
                    logger('')
                    logger(dnm + ' ' + '-'*(85 - len(dnm) - 1))
                    logger('')
                t0 = time.time()
                runopts = filter_options(run_modelplan, **kwargs)
                (model, loss, dice) = run_modelplan(training_plan, **runopts)
                t1 = time.time()
                row = dict(
                    input=dnm, loss=loss, dice=dice, training_time=(t1-t0))
                # See if this one is good enough that it needs to be saved.
                if model_cache_path is not None:
                    hh = [d for d in training_history if d['input'] == dnm]
                    hh = sorted(hh, key=lambda d:d['dice'])
                    if len(hh) == 0 or hh[0]['dice'] > dice:
                        # We need to save out this model!
                        savepath = os.path.join(
                            model_cache_path, f"best_{dnm}.pt")
                        torch.save(model.state_dict(), savepath)
                training_history.append(row)
                if logger: logger('')
    except KeyboardInterrupt:
        if logger:
            logger('')
            logger('KeyboardInterrupt caught; ending training.')
    training_history = ny.to_dataframe(training_history)
    if model_cache_path is not None:
        ny.save(
            os.path.join(model_cache_path, "training.tsv"),
            training_history)
    return training_history

def lookup_sids(dataset):
    """Returns a list of subject IDs for the given dataset name.

    The only argument, `dataset`, should be either `'hcp'` or `'nyu'`.
    """
    if dataset == 'hcp':
        from ..benson2024.config import hcp_sids
        return hcp_sids
    elif dataset == 'nyu':
        from ..benson2024.config import nyu_sids
        return nyu_sids
    else:
        raise ValueError("unrecognized dataset: {dataset}")
def load_training(model_key,
                  model_cache_path=None,
                  partition_log_marker='Partition ID:',
                  base_model='resnet18',
                  sids='hcp'):
    """Loads data from a directory written to during `train_until`.
    
    `load_training(model_key)` can be used to load data saved by a call to
    `train_until` with the given `model_key`.
    
    Parameters
    ----------
    model_key : str
        The model key used to save the `train_until` results.
    model_cache_path : str, optional
        The cache path in which the models are saved. By default uses
        `model_cache_path` (the gloabl value).
        
    Returns
    -------
    dict
        A dictionary whose keys include `'history'`, `'partition'`, `'models'`,
        `'plan'`, and `'options'`. The history is a summary of the training in
        the form of a PANDAS dataframe).  The `'models'` entry is a dictionary
        of the best models for each input type. The partition is represented as
        a tuple of `(trn_sids, val_sids)`. The plan and options are the
        parameters given to the `train_until` function.
    """
    if model_cache_path is None:
        path = model_key
    else:
        path = os.path.join(model_cache_path, model_key)
    hist_path = os.path.join(path, 'training.tsv')
    hist = ny.load(hist_path) if os.path.isfile(hist_path) else None
    opts_path = os.path.join(path, 'options.json')
    if os.path.isfile(opts_path):
        with open(opts_path, 'rt') as fl:
            opts = json.load(fl)
    else:
        opts = None
    plan_path = os.path.join(path, 'plan.json')
    if os.path.isfile(plan_path):
        with open(plan_path, 'rt') as fl:
            plan = json.load(fl)
    else:
        plan = None
    mdls = {}
    for fl in os.listdir(path):
        if fl.startswith('best_') and fl.endswith('.pt'):
            flnm = os.path.join(path, fl)
            state = torch.load(flnm)
            #nfeat = state['base_model.conv1.weight'].shape[1]
            nfeat = state['layer0.0.weight'].shape[1]
            nsegm = state['conv_last.weight'].shape[0]
            mdl = UNet(nfeat, nsegm, base_model=base_model)
            mdl.load_state_dict(state)
            mdls[fl[5:-3]] = mdl
    # Load in the partition as well.
    log_path = os.path.join(path, 'training.log')
    sids = lookup_sids(sids)
    try:
        with open(log_path, 'rt') as fl:
            for ln in fl:
                if partition_log_marker in ln:
                    part = ln.split(partition_log_marker)[-1].strip()
                    part = make_partition(sids, part)
    except Exception:
        part = None
    return dict(
        history=hist,
        models=mdls,
        partition=part,
        options=opts,
        plan=plan)
