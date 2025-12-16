NFACT documentation
======================================

**NFACT** (Non-negative matrix Factorisation of Tractography data) is a set of modules (as well as an end to end pipeline) that decomposes 
tractography data using NMF/ICA.

It consists of three "main" decomposition modules:
    
    - nfact_pp (Pre-process data for decomposition)
    
    - nfact_decomp (Decomposes a single or average group matrix using NMF or ICA)
    
    - nfact_dr (Dual regression on group matrix)

as well as three axillary "modules":
    
    - nfact_config (creates config files for the pipeline and changing any hyperparameters)
    
    - nfact_Qc (Creates hitmaps to check for bias in decomposition)

    - nfact_stats

and a pipeline wrapper
    
    - nfact 
    (runs nfact_pp, nfact_decomp, nfact_Qc and nfact_dr. nfact_pp, nfact_Qc and nfact_dr can all individually be skipped)

.. note::

   This project is currently under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   usage
   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`