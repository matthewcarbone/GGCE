===============
The GGCE logger
===============

The GGCE logger is an MPI-aware logging framework designed to expose the most important information to the user at all times. We use the wonderful logging package `Loguru <https://loguru.readthedocs.io/en/stable/>`__ for all of our logging needs, and utilize six logging levels:


Debug
-----

This stream used for debugging purposes only. Generally, you will not need to use the debugging feature of GGCE. However, if you want to turn it on, it can be controlled dynamically the ``debug`` context manager:

.. code-block:: python
    
    from ggce.logger import debug
    with debug():
        # do stuff

Note that all ranks pipe to their own debug stream during MPI jobs. This can get quite verbose if not properly used.

Info and Success
----------------

A general information ...

