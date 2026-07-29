Neuro-currentRF
===============

Neuro Current Response Functions (NCRFs) are cortical temporal response
functions (TRFs) directly estimated from continuous M/EEG data.
The NCRF framework combines the temporal response model with a
distributed forward model and estimates the source-space filters with a
Bayesian optimization algorithm :cite:`das2020neuro`.

The simplest entry point is :func:`ncrf.fit_ncrf`, which accepts Eelbrain
:class:`~eelbrain.NDVar` objects, prepares the regression design, selects a
regularization value when needed, and returns a structured fit report::

    from ncrf import fit_ncrf

    result = fit_ncrf(
        meg,
        stim,
        lead_field,
        noise,
        mu="auto",
        tstop=1.0,
        n_splits=3,
    )
    trf = result.model.h
    training_scores = result.scores

The fitted :class:`~ncrf.NCRF` in ``result.model`` is independent of the
optimizer that produced it. It can predict or evaluate another compatible
:class:`~ncrf.RegressionData` dataset. Optimization diagnostics remain in
``result.solver_fit``, and cross-validation results can be inspected with
:meth:`~ncrf.NCRFResult.cv_info`.

Architecture
------------

The package separates the fitting pipeline into components with distinct
responsibilities:

* :class:`~ncrf.RegressionData` prepares and stores the sensor data, lagged
  basis-projected covariates, and the metadata needed to reconstruct TRFs.
* :class:`~ncrf.NCRFEstimator` owns the forward model and whitening transform,
  selects a solver candidate through cross-validation, and runs the final fit.
* :class:`~ncrf.Solver` implementations, such as :class:`~ncrf.ChampLasso`,
  define optimization and candidate-selection behavior.
* :class:`~ncrf.NCRFResult` separates the reusable :class:`~ncrf.NCRF` model
  from the selected solver, solver-specific fit state, scores, and diagnostics.

See :doc:`architecture` for the data flow, the lower-level fitting API, and how
to evaluate a fitted model on new data.

.. toctree::
   :caption: User guide
   :maxdepth: 1

   installing
   guide
   architecture
   changes
   development
   references

.. toctree::
   :caption: Examples and API
   :maxdepth: 2

   auto_examples/index
   api/index

Project
-------

The NCRF package is maintained by Proloy Das at National Brain Research Centre,
Gurgaon and Christian Brodbeck at McMaster University. Current funding: NIH
1R01MH132660-01A1 (2024-). Past funding: NSF 1552946; NSF 1734892; DARPA
N6600118240224; NIH R01-DC-014085 (2016-2020).

This repository is free software covered by the MIT License. When using it for
a publication or talk, please cite :cite:`das2020neuro`.
