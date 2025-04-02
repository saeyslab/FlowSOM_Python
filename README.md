# FlowSOM

[![PyPI version][badge-pypi]][link-pypi]
[![Conda version][badge-conda]][link-conda]
[![Documentation][badge-docs]][link-docs]
[![Tests][badge-tests]][link-tests]
[![codecov][badge-codecov]][link-codecov]
[![DOI][badge-doi]][link-doi]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saeyslab/FlowSOM_Python/test.yaml?branch=main
[link-tests]: https://github.com/saeyslab/FlowSOM_Python/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/flowsom
[link-docs]: https://flowsom.readthedocs.io/en/latest/
[badge-pypi]: https://img.shields.io/pypi/v/flowsom
[link-pypi]: https://pypi.org/project/flowsom/
[badge-conda]: https://img.shields.io/conda/vn/conda-forge/flowsom
[link-conda]: https://anaconda.org/conda-forge/flowsom
[badge-doi]: https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtae179-blue
[link-doi]: https://doi.org/10.1093/bioinformatics/btae179
[badge-codecov]: https://codecov.io/gh/saeyslab/FlowSOM_Python/branch/main/graph/badge.svg
[link-codecov]: https://codecov.io/gh/saeyslab/FlowSOM_Python

The complete FlowSOM package known from R and [Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/FlowSOM.html), now available in Python with [scverse](https://scverse.org/) integration!

FlowSOM is a clustering and visualization algorithm originally based on a self-organizing map (SOM). FlowSOM is used to distinguish cell populations from cytometry data in an unsupervised way and
can help to gain deeper insights in fields such as immunology and oncology.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the following resources are available:

- [Example FlowSOM notebook][link-docs-example]
- [API documentation][link-api]
- [FlowSOM Python Cheatsheet][cheatsheet]

## Installation

You need to have Python 3.9 or newer installed on your system. There are several options to install FlowSOM:

**Recommended installation**: install the latest release of FlowSOM from [PyPI](https://pypi.org/project/FlowSOM/):

```bash
pip install flowsom
```

Alternative 1: install the development version from the GitHub repository:

```bash
pip install git+https://github.com/saeyslab/FlowSOM_Python
```

Alternative 2: install the [FlowSOM Conda package](https://anaconda.org/conda-forge/flowsom) via the Conda package manager:

```bash
conda install -c conda-forge flowsom
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
    ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], xdim=10, ydim=10, n_clusters=10, seed=42
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
[cheatsheet]: https://flowsom.readthedocs.io/en/latest/_static/FlowSOM_CheatSheet_Python.pdf
