"""Migrate the project's SQLite tables to the local XAMPP MySQL database."""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymysql


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "db_tembakau_ambat"
BATCH_SIZE = 500


@dataclass
class SourceTable:
    connection: sqlite3.Connection
    database_path: Path
    table_name: str
    columns: list[tuple[Any, ...]]
    indexes: list[tuple[str, bool, list[str]]]
    row_count: int
    create_sql: str


def quote_mysql_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def inspect_source_table(path: Path, table_name: str, required: bool) -> SourceTable | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Database SQLite sumber tidak ditemukan: {path}")
        print(f"[LEWATI] Database opsional tidak ditemukan: {path}")
        return None

    connection = open_sqlite_readonly(path)
    connection.execute("BEGIN")
    table = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if table is None:
        connection.close()
        if required:
            raise ValueError(f"Tabel wajib {table_name!r} tidak ditemukan di {path}")
        print(f"[LEWATI] Tabel opsional {table_name!r} tidak ditemukan di {path}")
        return None

    columns = connection.execute(
        f"PRAGMA table_info({quote_sqlite_identifier(table_name)})"
    ).fetchall()
    index_rows = connection.execute(
        f"PRAGMA index_list({quote_sqlite_identifier(table_name)})"
    ).fetchall()

    indexes: list[tuple[str, bool, list[str]]] = []
    for index_row in index_rows:
        index_name = index_row[1]
        is_unique = bool(index_row[2])
        origin = index_row[3]
        if origin == "pk":
            continue
        index_columns = connection.execute(
            f"PRAGMA index_info({quote_sqlite_identifier(index_name)})"
        ).fetchall()
        names = [item[2] for item in index_columns]
        if not names or any(name is None for name in names):
            connection.close()
            raise ValueError(
                f"Indeks ekspresi {index_name!r} pada tabel {table_name!r} "
                "tidak dapat dimigrasikan dengan aman."
            )
        indexes.append((index_name, is_unique, names))

    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {quote_sqlite_identifier(table_name)}"
    ).fetchone()[0]
    return SourceTable(
        connection=connection,
        database_path=path,
        table_name=table_name,
        columns=columns,
        indexes=indexes,
        row_count=row_count,
        create_sql=table[0] or "",
    )


def quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def indexed_text_lengths(source: SourceTable) -> dict[str, int]:
    indexed_columns = {
        column
        for _, _, index_columns in source.indexes
        for column in index_columns
    }
    lengths: dict[str, int] = {}
    for column in source.columns:
        name = column[1]
        declared_type = (column[2] or "").upper()
        if name not in indexed_columns or not any(
            marker in declared_type for marker in ("CHAR", "CLOB", "TEXT")
        ):
            continue
        maximum = source.connection.execute(
            f"SELECT MAX(LENGTH({quote_sqlite_identifier(name)})) "
            f"FROM {quote_sqlite_identifier(source.table_name)}"
        ).fetchone()[0]
        length = max(1, maximum or 0)
        if length > 768:
            raise ValueError(
                f"Kolom terindeks {name!r} pada {source.table_name!r} "
                f"memiliki nilai sepanjang {length} karakter; batas aman indeks "
                "utf8mb4 MySQL adalah 768 karakter."
            )
        lengths[name] = length
    return lengths


def mysql_column_type(
    declared_type: str,
    indexed_length: int | None,
) -> str:
    normalized = declared_type.upper().strip()
    if "INT" in normalized:
        return "BIGINT"
    if any(marker in normalized for marker in ("REAL", "FLOA", "DOUB")):
        return "DOUBLE"
    if "TIMESTAMP" in normalized or "DATETIME" in normalized:
        return "DATETIME"
    if "BLOB" in normalized:
        return "LONGBLOB"
    if any(marker in normalized for marker in ("CHAR", "CLOB", "TEXT")):
        if indexed_length is not None:
            return f"VARCHAR({indexed_length}) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
        return "LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
    raise ValueError(f"Tipe SQLite tidak didukung untuk migrasi: {declared_type!r}")


def mysql_default(default_value: str | None) -> str:
    if default_value is None:
        return ""
    value = default_value.strip()
    if value.upper() == "CURRENT_TIMESTAMP":
        return " DEFAULT CURRENT_TIMESTAMP"
    if value.upper() == "NULL":
        return " DEFAULT NULL"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return f" DEFAULT {value}"
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        literal = value[1:-1].replace(value[0] * 2, value[0])
        return " DEFAULT '" + literal.replace("'", "''") + "'"
    raise ValueError(f"Default SQLite tidak didukung tanpa konversi: {default_value!r}")


def create_mysql_table(cursor: Any, source: SourceTable) -> None:
    indexed_lengths = indexed_text_lengths(source)
    auto_increment = "AUTOINCREMENT" in source.create_sql.upper()
    definitions: list[str] = []
    primary_keys: list[str] = []

    for column in source.columns:
        _, name, declared_type, not_null, default_value, primary_key = column
        definition = (
            f"{quote_mysql_identifier(name)} "
            f"{mysql_column_type(declared_type or "", indexed_lengths.get(name))}"
        )
        if not_null or primary_key:
            definition += " NOT NULL"
        if primary_key:
            primary_keys.append(name)
            if auto_increment:
                definition += " AUTO_INCREMENT"
        definition += mysql_default(default_value)
        definitions.append(definition)

    if primary_keys:
        definitions.append(
            "PRIMARY KEY ("
            + ", ".join(quote_mysql_identifier(name) for name in primary_keys)
            + ")"
        )

    table_name = quote_mysql_identifier(source.table_name)
    cursor.execute(
        f"CREATE TABLE {table_name} ({', '.join(definitions)}) "
        "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin"
    )
    for index_name, is_unique, index_columns in source.indexes:
        unique_keyword = "UNIQUE " if is_unique else ""
        columns_sql = ", ".join(
            quote_mysql_identifier(name) for name in index_columns
        )
        cursor.execute(
            f"CREATE {unique_keyword}INDEX {quote_mysql_identifier(index_name)} "
            f"ON {table_name} ({columns_sql})"
        )


def copy_rows(cursor: Any, source: SourceTable) -> None:
    column_names = [column[1] for column in source.columns]
    columns_sql = ", ".join(quote_mysql_identifier(name) for name in column_names)
    placeholders = ", ".join(["%s"] * len(column_names))
    statement = (
        f"INSERT INTO {quote_mysql_identifier(source.table_name)} "
        f"({columns_sql}) VALUES ({placeholders})"
    )
    select_sql = (
        f"SELECT {', '.join(quote_sqlite_identifier(name) for name in column_names)} "
        f"FROM {quote_sqlite_identifier(source.table_name)}"
    )
    sqlite_cursor = source.connection.execute(select_sql)
    while rows := sqlite_cursor.fetchmany(BATCH_SIZE):
        cursor.executemany(statement, rows)


def migrate() -> int:
    feature_source = inspect_source_table(
        DATABASE_DIR / "tobacco_features.db",
        "fitur_data_train",
        required=True,
    )
    history_source = inspect_source_table(
        DATABASE_DIR / "tobacco_history.db",
        "inspection_history",
        required=False,
    )
    sources = [source for source in (feature_source, history_source) if source]
    if not sources:
        raise ValueError("Tidak ada tabel sumber yang dapat dimigrasikan.")

    try:
        # Validate schema conversions before making any changes in MySQL.
        for source in sources:
            indexed_text_lengths(source)

        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4",
            autocommit=False,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS {quote_mysql_identifier(MYSQL_DATABASE)} "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
                )
            connection.select_db(MYSQL_DATABASE)

            table_names = [source.table_name for source in sources]
            placeholders = ", ".join(["%s"] * len(table_names))
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT TABLE_NAME FROM information_schema.TABLES "
                    f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({placeholders})",
                    [MYSQL_DATABASE, *table_names],
                )
                existing = [row[0] for row in cursor.fetchall()]
                if existing:
                    raise ValueError(
                        "Migrasi dibatalkan karena tabel tujuan sudah ada: "
                        + ", ".join(existing)
                        + ". Skrip tidak menimpa atau menggabungkan data. "
                        "Periksa data di phpMyAdmin; hapus/ubah nama tabel tujuan "
                        "secara manual hanya jika memang ingin menjalankan migrasi ulang."
                    )

                for source in sources:
                    create_mysql_table(cursor, source)

                for source in sources:
                    copy_rows(cursor, source)

                for source in sources:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {quote_mysql_identifier(source.table_name)}"
                    )
                    target_count = cursor.fetchone()[0]
                    if target_count != source.row_count:
                        raise RuntimeError(
                            f"Verifikasi gagal pada {source.table_name}: "
                            f"SQLite={source.row_count}, MySQL={target_count}."
                        )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
    finally:
        for source in sources:
            source.connection.close()

    print(f"Migrasi selesai ke database MySQL {MYSQL_DATABASE!r}.")
    for source in sources:
        print(
            f"  {source.table_name}: {source.row_count} baris "
            f"(jumlah baris SQLite dan MySQL cocok)"
        )
    return 0


def main() -> int:
    try:
        return migrate()
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error, pymysql.MySQLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
