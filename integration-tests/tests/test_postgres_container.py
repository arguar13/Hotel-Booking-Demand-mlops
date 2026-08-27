"""Proves the Postgres connectivity pattern MLflow's backend store depends on
works, against a real (ephemeral) Postgres server - not a mock.
"""

import psycopg2
from testcontainers.community.postgres import PostgresContainer


def test_postgres_container_accepts_connections_and_queries() -> None:
    with PostgresContainer("postgres:18.3") as postgres:
        conn = psycopg2.connect(
            host=postgres.get_container_host_ip(),
            port=postgres.get_exposed_port(5432),
            user=postgres.username,
            password=postgres.password,
            dbname=postgres.dbname,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                assert cur.fetchone() == (1,)
        finally:
            conn.close()
