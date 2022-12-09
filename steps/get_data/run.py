import logging
import os

import fire
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# TODO: define wandb environment variables in a config file and read from config
os.environ["WANDB_PROJECT"] = "nyc_airbnb"
os.environ["WANDB_RUN_GROUP"] = "development"


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


def run(sample: str, artifact_name: str, artifact_type: str, artifact_description: str):
    # this is a bit hacky and must be running at the beginning of the function
    # other solutions check
    # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
    args = locals()

    # TODO: add arguments project
    run = wandb.init(job_type="download_file")
    run.config.update(args)

    logger.info(f"Returning sample {sample}")
    logger.info(f"Uploading {artifact_name} to Weights & Biases")
    log_artifact(
        artifact_name=artifact_name,
        artifact_type=artifact_type,
        artifact_description=artifact_description,
        filename=os.path.join("data", sample),
        wandb_run=run,
    )


if __name__ == "__main__":
    fire.Fire(run)
