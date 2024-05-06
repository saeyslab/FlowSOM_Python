# FlowSOM

<!-- [![Tests][badge-tests]][link-tests] -->

[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saeyslab/FlowSOM_Python/test.yaml?branch=main
[link-tests]: https://github.com/saeyslab/FlowSOM_Python/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/flowsom

The complete FlowSOM package known from R, now available in Python!

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [Example FlowSOM notebook][link-docs-example]
-   [API documentation][link-api]

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
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
pip install git+https://github.com/saeyslab/FlowSOM_Python
```

## Usage

Starting from an FCS file that is properly transformed, compensated and checked for quality, the following code can be used to run the FlowSOM algorithm:

```python
# Import the FlowSOM package
import flowsom as fs

# Load the FCS file
ff = fs.io.read_FCS("./tests/data/ff.fcs")

# Run the FlowSOM algorithm
fsom = fs.FlowSOM(
    ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], xdim=10, ydim=10, n_clus=10
)

# Plot the FlowSOM results
p = fs.pl.plot_stars(fsom, background_values=fsom.get_cluster_data().obs.metaclustering)
p.show()
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests or if you found a bug, please use the [issue tracker][issue-tracker].

## Citation

If you use `FlowSOM` in your work, please cite the following papers:

> A. Couckuyt, B. Rombaut, Y. Saeys, and S. Van Gassen, “Efficient cytometry analysis with FlowSOM in Python boosts interoperability with other single-cell tools,” Bioinformatics, vol. 40, no. 4, p. btae179, Apr. 2024, doi: [10.1093/bioinformatics/btae179](https://doi.org/10.1093/bioinformatics/btae179).

> S. Van Gassen et al., “FlowSOM: Using self-organizing maps for visualization and interpretation of cytometry data,” Cytometry Part A, vol. 87, no. 7, pp. 636–645, 2015, doi: [10.1002/cyto.a.22625](https://doi.org/10.1002/cyto.a.22625).

[issue-tracker]: https://github.com/saeyslab/FlowSOM_Python/issues
[changelog]: https://flowsom.readthedocs.io/en/latest/changelog.html
[link-docs]: https://flowsom.readthedocs.io
[link-docs-example]: https://flowsom.readthedocs.io/en/latest/notebooks/example.html
[link-api]: https://flowsom.readthedocs.io/en/latest/api.html
