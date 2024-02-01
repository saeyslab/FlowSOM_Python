import csv
from pathlib import Path

import pytest

import flowsom as fs


def read_fcs_file(filename):
    return fs.io.read_FCS(filename)


@pytest.fixture(scope="session")
def ff_path():
    return Path("./tests/data/ff.fcs")


@pytest.fixture(scope="session")
def fcs_path():
    return Path("./tests/data/fcs.csv")


@pytest.fixture(scope="session")
def gating_path():
    return Path("./tests/data/gating_result.csv")


@pytest.fixture(scope="session")
def unprocessed_path():
    return Path("./tests/data/not_processed.fcs")


@pytest.fixture(scope="session")
def fcs(ff_path):
    file = ff_path
    return read_fcs_file(file)


@pytest.fixture(scope="session")
def FlowSOM_res(fcs):
    fsom = fs.FlowSOM(fcs, cols_to_use=[8, 11, 13, 14, 15, 16, 17])
    return fsom


@pytest.fixture(scope="session")
def gating_results(gating_path):
    with open(gating_path) as file:
        data = list(csv.reader(file))
    return [i[0] for i in data]
