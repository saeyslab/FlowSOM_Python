# API

Import FlowSOM using the alias `fs` for convenience. Functions and classes can then be accessed as `fs.FlowSOM` or `fs.io.read_FCS`:

```
import flowsom as fs
```

The functionality is organised in subpackages:

-   `io` for reading and writing FCS files
-   `pp` for preprocessing
-   `models` for models
-   `tl` for tools
-   `pl` for plotting

The central class is `FlowSOM`, which can be used to read in data and directly perform the FlowSOM algorithm {cite:p}`van_gassen_flowsom_2015`. There is also a convenience function `flowsom_clustering` which performs the same steps but directly on an existing AnnData object. For more low-level control, the `FlowSOMEstimator` class with a scikit-learn Estimator API can be used.

For more background information, see the paper for this software package {cite:p}`couckuyt_efficient_2024` and the Nature Protocols paper {cite:p}`quintelier_analyzing_2021` on applying FlowSOM in practice.

```{eval-rst}
.. module:: flowsom
.. autosummary::
    :toctree: generated

    FlowSOM
    flowsom_clustering
```

## Reading

```{eval-rst}
.. module:: flowsom.io
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    io.read_FCS
    io.read_csv
```

## Preprocessing

```{eval-rst}
.. module:: flowsom.pp
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    pp.aggregate_flowframes
```

## Models

```{eval-rst}
.. module:: flowsom.models
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    models.FlowSOMEstimator
    models.SOMEstimator
    models.ConsensusCluster
    models.BaseClusterEstimator
    models.BaseFlowSOMEstimator
    models.SOM
    models.map_data_to_codes
```

## Tools

```{eval-rst}
.. module:: flowsom.tl
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    tl.get_channels
    tl.get_cluster_percentages_positive
    tl.get_counts
    tl.get_features
    tl.get_markers
    tl.get_metacluster_percentages_positive
    tl.get_percentages

```

## Plotting

```{eval-rst}
.. module:: flowsom.pl
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    pl.FlowSOMmary
    pl.plot_2D_scatters
    pl.plot_labels
    pl.plot_numbers
    pl.plot_variable
    pl.plot_marker
    pl.plot_stars
    pl.plot_pies
```
