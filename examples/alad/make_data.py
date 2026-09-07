"""Data of the alad example: anomaly_data.parquet, the table the ALAD objectives train on."""

from pathlib import Path

from kalfa.synthetic import write_anomaly

HERE = Path(__file__).parent


def main():
    write_anomaly(HERE / "anomaly_data.parquet")
    print("wrote anomaly_data.parquet (2000 rows, about nine percent anomalies)")


if __name__ == "__main__":
    main()
