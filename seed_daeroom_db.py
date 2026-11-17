"""
Seeder & Verifikasi Database Daeroom Cloud (db_bikesharingdemand)
Membuat tabel bike_sharing_demand dan prediction_logs, lalu mengimpor seluruh 10.886 baris bike_sharing.csv.
"""
from pathlib import Path
import pandas as pd
from daeroom_client import query_db


def create_tables() -> None:
    print("1. Membuat skema tabel di PostgreSQL Daeroom Cloud...")
    query_db(
        """
        CREATE TABLE IF NOT EXISTS bike_sharing_demand (
            id SERIAL PRIMARY KEY,
            datetime TIMESTAMP NOT NULL UNIQUE,
            season SMALLINT NOT NULL,
            holiday SMALLINT NOT NULL,
            workingday SMALLINT NOT NULL,
            weather SMALLINT NOT NULL,
            temp DOUBLE PRECISION NOT NULL,
            atemp DOUBLE PRECISION NOT NULL,
            humidity SMALLINT NOT NULL,
            windspeed DOUBLE PRECISION NOT NULL,
            casual INTEGER NOT NULL,
            registered INTEGER NOT NULL,
            count INTEGER NOT NULL
        );
        """
    )
    query_db(
        """
        CREATE INDEX IF NOT EXISTS idx_bike_sharing_datetime
        ON bike_sharing_demand (datetime);
        """
    )
    query_db(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            jam SMALLINT NOT NULL,
            suhu DOUBLE PRECISION NOT NULL,
            tipe_hari VARCHAR(64) NOT NULL,
            cuaca VARCHAR(64) NOT NULL,
            pred_total INTEGER NOT NULL,
            pred_komuter INTEGER NOT NULL,
            pred_rekreasi INTEGER NOT NULL,
            truk_dibutuhkan INTEGER NOT NULL,
            catatan TEXT
        );
        """
    )
    print("   Tabel bike_sharing_demand dan prediction_logs siap.")


def seed_dataset() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "bike_sharing.csv"
    df = pd.read_csv(csv_path)
    total_csv = len(df)
    print(f"2. Membaca CSV lokal: {csv_path.name} ({total_csv} baris).")

    existing = query_db("SELECT COUNT(*) AS total FROM bike_sharing_demand;", readonly=True)
    current_count = int(existing["rows"][0]["total"])
    if current_count == total_csv:
        print(f"   Database sudah berisi lengkap {current_count} baris. Tidak perlu insert ulang.")
        return

    batch_size = 500
    inserted_batches = 0
    for start in range(0, total_csv, batch_size):
        chunk = df.iloc[start : start + batch_size]
        placeholders = []
        params = []
        p_idx = 1
        for row in chunk.itertuples(index=False):
            placeholders.append(
                f"(${p_idx}, ${p_idx+1}, ${p_idx+2}, ${p_idx+3}, ${p_idx+4}, "
                f"${p_idx+5}, ${p_idx+6}, ${p_idx+7}, ${p_idx+8}, ${p_idx+9}, ${p_idx+10}, ${p_idx+11})"
            )
            params.extend([
                str(row.datetime),
                int(row.season),
                int(row.holiday),
                int(row.workingday),
                int(row.weather),
                float(row.temp),
                float(row.atemp),
                int(row.humidity),
                float(row.windspeed),
                int(row.casual),
                int(row.registered),
                int(row.count),
            ])
            p_idx += 12

        sql = (
            "INSERT INTO bike_sharing_demand "
            "(datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count) "
            "VALUES " + ", ".join(placeholders) + " "
            "ON CONFLICT (datetime) DO NOTHING;"
        )
        query_db(sql, params=params)
        inserted_batches += 1
        done = min(start + batch_size, total_csv)
        print(f"   Batch {inserted_batches}: {done}/{total_csv} baris diproses...")


def verify_database() -> None:
    print("3. Verifikasi data di PostgreSQL Daeroom Cloud...")
    stats = query_db(
        """
        SELECT
            COUNT(*) AS total_rows,
            MIN(datetime) AS min_dt,
            MAX(datetime) AS max_dt,
            ROUND(AVG(count)::numeric, 2) AS avg_count,
            SUM(count) AS total_rentals
        FROM bike_sharing_demand;
        """,
        readonly=True,
    )
    print("   Statistik tabel bike_sharing_demand:", stats["rows"][0])

    sample = query_db(
        "SELECT id, datetime, season, temp, humidity, count FROM bike_sharing_demand ORDER BY datetime LIMIT 3;",
        readonly=True,
    )
    print("   3 baris awal:", sample["rows"])


if __name__ == "__main__":
    create_tables()
    seed_dataset()
    verify_database()
