===============
The GGCE logger
===============

The GGCE logger is an MPI-aware logging framework designed to expose the most important information to the user at all times. We use the wonderful logging package `Loguru <https://loguru.readthedocs.io/en/stable/>`__ for all of our logging needs, and utilize six logging levels. In general, we follow the conventions for logging levels as outlined `here <https://docs.python.org/3/howto/logging.html#when-to-use-logging>`__.

Logging levels
==============

Debug
-----

This stream used for debugging purposes only. Generally, you will not need to use the debugging feature of GGCE.

.. warning::

    Note that all ranks pipe to their own debug stream during MPI jobs. This can get quite verbose if not properly used.

Info and Success
----------------

A general information pipeline. Used for generic messages. Both the info and success logging levels are only printed for the main MPI rank.


Warning
-------

Generally used to indicate that something happened the user should be aware of. However, warnings are _not_ something to be concerned about. Warnings can always be safely ignored, but they should be noted as more important than information piped to the info and success streams.

Error
-----

Errors indicate a serious issue, but not one that will necessarily terminate the program. Errors can sometimes be ignored, but should never be disregarded.

Critical
--------

If GGCE throws a critical error the program will almost certainly terminate. Critical errors also throw a ``sys.exit(1)`` exception or ``COMM.Abort()`` depending on if GGCE is running in MPI mode or not.


The GGCE logger context manager
===============================

By default, the GGCE logger will print info and above. However, one can easily control the logging level dynamically using our context managers in ``ggce/logger.py``.





