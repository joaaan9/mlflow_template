from build import create_features
from utils.utils import load_data, save_data


def workflow():
    """
    Load data extracted from Snowflake (from local or s3 bucket), create features and save it in data/interim
    """
    df = load_data("raw")
    df = create_features(df)
    for i in df:
        save_data(df[i], "interim", name=f"{i}_")


if __name__ == '__main__':
    workflow()
