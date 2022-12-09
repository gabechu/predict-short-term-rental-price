import os
import tempfile

import fire
import pandas as pd

import wandb


# TODO: make it a python package to reuse across projects
def log_artifact(
    artifact_name, artifact_type, artifact_description, filename, wandb_run
):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()


def drop_outliers(df: pd.DataFrame, min_price: int, max_price: int) -> pd.DataFrame:
    is_valid = df["price"].between(min_price, max_price)
    return df[is_valid]


def basic_cleaning(df: pd.DataFrame, min_price: int, max_price: int) -> pd.DataFrame:
    return drop_outliers(df, min_price=min_price, max_price=max_price)


def run(
    input_artifact: str,
    output_artifact: str,
    output_type: str,
    output_description: str,
    min_price: int,
    max_price: int,
):
    args = locals()

    # TODO: extract project and group to config
    wandb_project = "nyc_airbnb"
    wandb_group = "development"
    wandb_job_type = "basic_cleaning"

    run = wandb.init(project=wandb_project, group=wandb_group, job_type=wandb_job_type)
    run.config.update(args)
    downloaded_file = wandb.use_artifact(input_artifact).file()

    df = pd.read_csv(downloaded_file)
    clean_df = basic_cleaning(df=df, min_price=min_price, max_price=max_price)

    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, "clean_sample.csv")
        clean_df.to_csv(save_path, index=False)

        log_artifact(
            artifact_name=output_artifact,
            artifact_type=output_type,
            artifact_description=output_description,
            filename=save_path,
            wandb_run=run,
        )


if __name__ == "__main__":
    fire.Fire(run)
