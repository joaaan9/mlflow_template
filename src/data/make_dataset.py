from utils.utils import load_from_snowflake, save_data


def workflow():
    """
    Load data from snowflake and save it as a pickle file in data/raw directory or s3 bucket
    """
    df = load_from_snowflake()
    save_data(df, "raw")


if __name__ == '__main__':
    workflow()
