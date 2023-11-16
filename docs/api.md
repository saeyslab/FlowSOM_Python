# API

Import FlowSOM as:

```
import flowsom as fs
```

The functionality is organised in subpackages:
- `io` for reading and writing FCS files
- `pp` for preprocessing
- `tl` for tools
- `pl` for plotting

The central class is `FlowSOM`.

## Reading

```{eval-rst}
.. module:: flowsom.io
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    io.read_FCS
```

## Preprocessing

```{eval-rst}
.. module:: flowsom.pp
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    pp.aggregate_flowframes
    pp.normalize_estimate_logicle
```

## Tools

```{eval-rst}
.. module:: flowsom.tl
.. currentmodule:: flowsom

.. autosummary::
    :toctree: generated

    tl.ConsensusCluster
    tl.SOM
    tl.map_data_to_codes
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
