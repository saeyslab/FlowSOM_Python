import pytest


from FlowSOM.main import read_FCS, FlowSOM
from pathlib import Path


def read_fcs_file(filename):
    return read_FCS(filename)


@pytest.fixture(scope="session")
def fcs():
    file = Path("./tests/data/ff.fcs")
    return read_fcs_file(file)


@pytest.fixture(scope="session")
def FlowSOM_res(fcs):
    fsom = FlowSOM(fcs, cols_to_use=[8, 11, 13, 14, 15, 16, 17])
    return fsom