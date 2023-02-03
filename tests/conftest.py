import pytest

# from FlowSOM import example_dataset
from FlowSOM.main import read_FCS, FlowSOM


def read_fcs_file(filename):
    return read_FCS(filename)


@pytest.fixture(scope="session")
def fcs():
    # file = example_dataset.fetch("flow/FlowRepository_FR-FCM-ZZPH/Levine_13dim.fcs")
    file = "./data/ff.fcs"
    return read_fcs_file(file)


@pytest.fixture(scope="session")
def FlowSOM_res():
    # file = example_dataset.fetch("flow/FlowRepository_FR-FCM-ZZPH/Levine_13dim.fcs")
    file = "./data/ff.fcs"
    fcs = read_fcs_file(file)
    fsom = FlowSOM(fcs, cols_to_use=[8, 11, 13, 14, 15, 16, 17])
    return fsom
