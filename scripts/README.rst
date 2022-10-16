Helper scripts
==============
This directory contains a variety of scripts which are used not only by the continuous integration (CI) pipeline, but can also be used by contributors during the development process.


Installation
------------
``install.sh`` is a helper script for parsing the ``pyproject.toml`` file (see `here <https://github.com/pypa/pip/issues/8049>`_ for the inspiration for these scripts) and installing the required packages as specified there. As also explained in the `README <../README.rst>`_, we can install all or part of GGCE's dependencies via the following scripts:

.. code-block:: bash
    
    bash scripts/install.sh       # Install GGCE's core dependencies
    bash scripts/install.sh test  # Install the test requirements only
    bash scripts/install.sh doc   # Install requirements for building the docs

Building
--------
We provide two self-contained scripts for building the GGCE source and docs. These scripts are `build_project.sh` and `build_docs.sh`, respectively.

Versioning
----------
The version of GGCE is controlled in a completely automatic fashion during CI execution. It works using the script `update_version.sh`.

License
=======
The code in this directory is derived/modified from software in the ``scripts`` directory (`README permalink <https://github.com/AI-multimodal/Lightshow/blob/f7d2d6458bf7532994d4f2fe2ffdfe6d2627bdd7/scripts/README.rst>`__) in `AI-multimodal/Lightshow/scripts <https://github.com/AI-multimodal/Lightshow/tree/master/scripts>`__ under a BSD 3-clause License, which allows for redistribution and modification with attribution:

.. code-block::

    BSD 3-Clause License

    Copyright (c) 2022, Brookhaven Science Associates, LLC, Brookhaven National Laboratory
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
