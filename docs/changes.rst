Changes
=======

Normalization
-------------

Normalization is now owned by :class:`~ncrf.RegressionData` and checked when a
fitted model is applied to new data (see :doc:`guide`).

- :func:`~ncrf.fit_ncrf` and :meth:`~ncrf.RegressionData.from_data` take a single
  ``scale`` argument instead of ``normalize`` and
  ``do_post_normalization``/``post_normalize``, which were not independent
  (post-normalization cancelled out ``'l1'``/``'l2'`` scaling):

  ====================  ==========================  ==================================
  ``normalize``         ``do_post_normalization``    now
  ====================  ==========================  ==================================
  ``False`` (default)   ``True`` (default)          ``scale='spectral'`` (default)
  ``'l1'``/``'l2'``     ``True``                    ``scale='spectral'``
  ``'l1'``/``'l2'``     ``False``                   ``scale='l1'``/``'l2'``
  ``False``             ``False``                   ``scale=None``
  ====================  ==========================  ==================================

- Every ``scale`` also subtracts each predictor's mean. Previously the mean was only
  subtracted with ``normalize``, so the default fit scaled the covariates without
  centering the predictors; it now centers them.
- **Bug fix:** :attr:`ncrf.NCRF.h_scaled` undid the covariate scaling in the wrong
  direction. The covariates are divided by the scale, so restoring stimulus units
  requires dividing ``h`` by it as well, but ``h_scaled`` multiplied instead. Values
  obtained with ``normalize='l1'`` or ``normalize='l2'`` were therefore off by the
  square of the scale factor.
- :attr:`ncrf.NCRF.h_scaled` now also undoes spectral scaling; previously ``h``
  from a post-normalized fit was not in stimulus units.
- Spectral scaling is now also applied to data with a single covariate channel,
  which changes the scale of ``h`` (but not of ``h_scaled``) for such fits. Since
  ``mu`` regularizes coefficients on the covariate scale, values tuned before this
  change are not comparable for those fits.
