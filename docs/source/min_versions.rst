===================================
Minimum Version of Python and NumPy
===================================


This project will only support ``python >= 3.6`` and will generally support the most up-to-date versions of ``numpy``, ``scipy``, etc.

The minimum supported version of Python will be set to
``python_requires`` in ``setup``. All supported minor versions of
Python will be in the test matrix and have binary artifacts built
for releases.

The project should adjust upward the minimum Python and NumPy
version support on every minor and major release, but never on a
patch release.

This is consistent with NumPy `NEP 29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.
