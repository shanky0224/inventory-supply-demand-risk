import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Clean unit price
    df["unit_price"] = (
        df["unit_price"]
        .astype(str)
        .str.replace("$", "", regex=False)
    )
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)

    # Date handling
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["last_order_date"] = pd.to_datetime(df["last_order_date"], errors="coerce")

    today = pd.Timestamp.today()

    df["inventory_age_days"] = (today - df["date_received"]).dt.days.fillna(0)
    df["days_since_last_order"] = (today - df["last_order_date"]).dt.days.fillna(0)

    # Business logic
    df["reorder_pressure"] = df["reorder_level"] - df["stock_quantity"]

    return df
