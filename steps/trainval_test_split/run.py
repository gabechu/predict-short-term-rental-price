import logging
import tempfile

import fire
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# TODO: make it a python package to reuse across projects
def log_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_description: str,
    local_artifact_path: str,
    wandb_run,
):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(local_artifact_path)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()


def run(wandb_file: str, test_size: float, random_seed: int, stratify_by: str):
    args = locals()

    wandb_project = "nyc_airbnb"
    wandb_group = "development"
    wandb_job_type = "train_val_test_split"

    run = wandb.init(project=wandb_project, group=wandb_group, job_type=wandb_job_type)
    run.config.update(args)

    logger.info(f"Fetching artifact {wandb_file}")
    artifact_local_path = run.use_artifact(wandb_file).file()

    df = pd.read_csv(artifact_local_path)
    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[stratify_by] if stratify_by != "none" else None,
    )

    for df, file_key in zip([trainval, test], ["trainval", "test"]):
        logger.info(f"Uploading {file_key}_data.csv dataset")
        with tempfile.NamedTemporaryFile() as fp:
            df.to_csv(fp.name, index=False)
            log_artifact(
                artifact_name=f"{file_key}_data.csv",
                artifact_type=f"{file_key}_data",
                artifact_description=f"{file_key} split of data",
                local_artifact_path=fp.name,
                wandb_run=run,
            )


if __name__ == "__main__":
    fire.Fire(run)
