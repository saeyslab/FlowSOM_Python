# FlowSOM

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/artuurC/FlowSOM/test.yaml?branch=main
[link-tests]: https://github.com/artuurC/FlowSOM/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/FlowSOM

The complete FlowSOM package known from R, now available in Python

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install FlowSOM:

<!--
1) Install the latest release of `FlowSOM` from `PyPI <https://pypi.org/project/FlowSOM/>`_:

```bash
pip install FlowSOM
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/artuurC/FlowSOM.git@main
```

## Usage
Starting from an FCS file that is properly transformed, compensated and checked for quality, the following code can be used to run the FlowSOM algorithm:

```python
# Import the FlowSOM package
import FlowSOM as fs

# Load the FCS file
ff = fs.main.read_FCS("./tests/data/ff.fcs")

# Run the FlowSOM algorithm
fsom = fs.main.FlowSOM(ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], xdim=10, ydim=10, n_clus=10)

# Plot the FlowSOM results
p = fs.pl.plot_stars(fsom, background_values=fsom.get_cluster_data().obs.metaclustering
p.show()
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation
If you use `FlowSOM` in your work, please cite the following paper:

> Sofie Van Gassen, Britt Callebaut, Mary J. Van Helden, Bart N. Lambrecht, Piet Demeester, Tom Dhaene and Yvan Saeys. FlowSOM: Using self-organizing maps for visualization and interpretation of cytometry data. Cytometry A 2015, volume 87.7 (p. 636-645)

> DOI: 10.1002/cyto.a.22625

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/artuurC/FlowSOM/issues
[changelog]: https://FlowSOM.readthedocs.io/latest/changelog.html
[link-docs]: https://FlowSOM.readthedocs.io
[link-api]: https://FlowSOM.readthedocs.io/latest/api.html
