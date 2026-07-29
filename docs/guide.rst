Guide
=====

Normalization
-------------

The covariates that an NCRF is fit against can be centered and scaled.
:class:`~ncrf.RegressionData` handles this normalization: it computes the
values, applies them to the covariates, and records them on its ``design``.

:meth:`~ncrf.RegressionData.from_data` and :func:`~ncrf.fit_ncrf` take one
``scale`` argument. Each predictor's mean is subtracted, and each covariate
channel is divided by one factor:

- ``'spectral'`` (the default): the average spectral norm of the channel's
  covariates. This equalizes covariate scales across predictor variables, so that
  one regularization parameter is appropriate for all of them.
- ``'l1'``/``'l2'``: the predictor's mean absolute deviation or standard
  deviation.
- ``None``: leave the covariates on their raw scale, without centering.

Centering and scaling are a single choice because they only make sense together.
M/EEG data is high-pass filtered, so an uncentered predictor's mean would
contribute a constant ``mean * sum(h)`` to the prediction, which the fit has to
reconcile with zero-mean data -- in effect constraining the response function to
sum to zero. And the scale factors measure variation around the mean rather than
raw magnitude, since a predictor's offset would otherwise dominate its scale and
leave its variation over-regularized.

:attr:`~ncrf.NCRF.h` is in units of the normalized covariates.
:attr:`~ncrf.NCRF.h_scaled` restores the original stimulus scale,
and is the appropriate representation for comparing response
functions across predictors or across models.

Applying a model to new data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fitted model can only be applied to covariates on the scale it was fit on --
multiplying coefficients learned from ``(stim - baseline) / scaling`` with a raw
stimulus would silently produce wrong predictions and scores. New data therefore
has to carry the same normalization as the training data. Prepare it
unnormalized and apply the model's own normalization with
:meth:`~ncrf.RegressionData.normalize`::

    test_data = RegressionData.from_data(
        [test_meg],
        [[test_stim]],
        tstart=0.0,
        tstop=1.0,
        stim_is_single=True,
        scale=None,
    )
    test_data = test_data.normalize(result.model.design)
    scores = result.model.evaluate(test_data)

Note that this uses the training data's normalization values, which is what
makes the scores comparable; re-deriving them from the test data would put the
covariates on a different scale.

:meth:`~ncrf.NCRF.predict`, :meth:`~ncrf.NCRF.evaluate` and
:meth:`~ncrf.NCRF.voxelwise_explained_variance` raise an error when ``data``
carries a different normalization, or when its design describes different
predictors, TRF timings, or a different basis. Normalizing data that is already
normalized is a no-op when the values agree, and an error otherwise, so a dataset
can safely be passed to :meth:`~ncrf.RegressionData.normalize` more than once.
For large datasets, ``inplace=True`` avoids copying the covariates.
