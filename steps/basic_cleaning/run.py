import fire
import wandb
import pandas as pd


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    min_price = 10
    max_price = 350

    is_valid = df["price"].between(min_price, max_price)
    return df[is_valid]


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    return drop_outliers(df)


def run():
    wandb_project = "nyc_airbnb"
    wandb_job_type = "basic_cleaning"
    artifact_name = "sample.csv"
    artifact_alias = "latest"

    run = wandb.init(project=wandb_project, job_type=wandb_job_type)
    # run.config.update()
    downloaded_file = wandb.use_artifact(f"{artifact_name}:{artifact_alias}").file()

    df = pd.read_csv(downloaded_file)
    clean_df = basic_cleaning(df)

    # TODO: log artifact


if __name__ == "__main__":
    fire.Fire(run)
