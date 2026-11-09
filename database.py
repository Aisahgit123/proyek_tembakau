import pymysql


MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "db_tembakau_ambat"
MYSQL_CHARSET = "utf8mb4"

pymysql.install_as_MySQLdb()


def get_database_connection():
    connection = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset=MYSQL_CHARSET,
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                f"CHARACTER SET {MYSQL_CHARSET} COLLATE {MYSQL_CHARSET}_unicode_ci"
            )
        connection.select_db(MYSQL_DATABASE)
        return connection
    except Exception:
        connection.close()
        raise


def print_mysql_startup_warning(error):
    print(
        "-> [PERINGATAN] Tidak dapat tersambung ke MySQL XAMPP "
        f"({MYSQL_HOST}:{MYSQL_PORT}): {error}"
    )
    print(
        "-> [PETUNJUK] Jalankan modul MySQL melalui XAMPP Control Panel, "
        "pastikan port 3306 tersedia, lalu coba kembali."
    )
