import pandas as pd
import pytest
from pytest import FixtureRequest, Parser

import wandb

wandb_project = "nyc_airbnb"
wandb_group = "development"
wandb_job_type = "data_tests"


def pytest_addoption(parser: Parser):
    parser.addoption("--csv")
    parser.addoption("--ref")
    parser.addoption("--kl_threshold")
    parser.addoption("--min_price")
    parser.addoption("--max_price")


@pytest.fixture(scope="session")
def data(request: FixtureRequest) -> pd.DataFrame:
    run = wandb.init(
        project=wandb_project,
        group=wandb_group,
        job_type=wandb_job_type,
        resume=True,
    )
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line.")

    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def ref_data(request: FixtureRequest) -> pd.DataFrame:
    run = wandb.init(
        project=wandb_project, group=wandb_group, job_type=wandb_job_type, resume=True
    )
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def kl_threshold(request: FixtureRequest) -> float:
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope="session")
def min_price(request) -> float:
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope="session")
def max_price(request) -> float:
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must prive max_price")

    return float(max_price)
