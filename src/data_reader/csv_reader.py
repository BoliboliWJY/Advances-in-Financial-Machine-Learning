import polars as pl


def read_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def read_csv_with_schema(path: str, schema: dict) -> pl.DataFrame:
    return pl.read_csv(path, schema=schema)


