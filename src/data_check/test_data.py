import pandas as pd
import scipy.stats


def test_column_names(data: pd.DataFrame):
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    actual_columns = list(data.columns.values)
    assert actual_columns == expected_columns


def test_neighbourhood_names(data: pd.DataFrame):
    expected_neighbourhood = set(
        ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    )
    actual_neighbourhood = set(data["neighbourhood_group"].unique())

    assert expected_neighbourhood == actual_neighbourhood


def test_proper_boundaries(data: pd.DataFrame):
    "Test proper longitude and latitude boundaries for properties in and around NYC"
    assert (
        data["longitude"].between(-74.25, -73.50) & data["latitude"].between(40.5, 41.2)
    ).all()


def test_similar_neighbourhood_distribution(
    data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float
):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data["neighbourhood_group"].value_counts().sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data: pd.DataFrame):
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data: pd.DataFrame, min_price: int, max_price: int):
    assert data["price"].between(min_price, max_price).all()
