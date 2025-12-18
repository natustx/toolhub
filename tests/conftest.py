"""Pytest fixtures for toolhub tests.

Provides Docker-based Postgres (pgvector) + MinIO fixtures for integration tests.
Each test worker gets isolated databases and S3 prefixes to enable parallelism.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from typing import TYPE_CHECKING

import boto3
import psycopg
import pytest
from botocore.client import Config as BotoConfig

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# Test container configuration (matches docker-compose.test.yml)
TEST_POSTGRES_HOST = os.getenv("TEST_POSTGRES_HOST", "localhost")
TEST_POSTGRES_PORT = os.getenv("TEST_POSTGRES_PORT", "5433")
TEST_POSTGRES_USER = os.getenv("TEST_POSTGRES_USER", "toolhub")
TEST_POSTGRES_PASSWORD = os.getenv("TEST_POSTGRES_PASSWORD", "toolhub")
TEST_POSTGRES_DB = os.getenv("TEST_POSTGRES_DB", "toolhub_test")

TEST_MINIO_ENDPOINT = os.getenv("TEST_MINIO_ENDPOINT", "http://localhost:9010")
TEST_MINIO_ACCESS_KEY = os.getenv("TEST_MINIO_ACCESS_KEY", "minioadmin")
TEST_MINIO_SECRET_KEY = os.getenv("TEST_MINIO_SECRET_KEY", "minioadmin")
TEST_MINIO_BUCKET = os.getenv("TEST_MINIO_BUCKET", "toolhub-test")


def get_worker_id() -> str:
    """Get pytest-xdist worker ID or 'main' for single-process runs."""
    return os.getenv("PYTEST_XDIST_WORKER", "main")


def get_test_database_url(worker_id: str) -> str:
    """Generate database URL for a specific worker."""
    db_name = f"{TEST_POSTGRES_DB}_{worker_id}"
    return f"postgresql://{TEST_POSTGRES_USER}:{TEST_POSTGRES_PASSWORD}@{TEST_POSTGRES_HOST}:{TEST_POSTGRES_PORT}/{db_name}"


@pytest.fixture(scope="session")
def postgres_admin_conn() -> Generator[psycopg.Connection, None, None]:
    """Session-scoped connection to create/drop worker databases."""
    conn_str = f"postgresql://{TEST_POSTGRES_USER}:{TEST_POSTGRES_PASSWORD}@{TEST_POSTGRES_HOST}:{TEST_POSTGRES_PORT}/{TEST_POSTGRES_DB}"
    with psycopg.connect(conn_str, autocommit=True) as conn:
        yield conn


@pytest.fixture(scope="session")
def test_database_url(postgres_admin_conn: psycopg.Connection) -> Generator[str, None, None]:
    """Create an isolated database for this test worker.

    Each pytest-xdist worker gets its own database to enable parallel tests.
    """
    worker_id = get_worker_id()
    db_name = f"{TEST_POSTGRES_DB}_{worker_id}"

    # Drop and recreate the worker database
    postgres_admin_conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
    postgres_admin_conn.execute(f"CREATE DATABASE {db_name}")

    db_url = get_test_database_url(worker_id)

    # Apply migrations in order
    with psycopg.connect(db_url) as conn:
        migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
        migration_files = sorted(
            f for f in os.listdir(migrations_dir) if f.endswith(".sql")
        )
        for migration_file in migration_files:
            migration_path = os.path.join(migrations_dir, migration_file)
            with open(migration_path) as f:
                conn.execute(f.read())
        conn.commit()

    yield db_url

    # Cleanup: drop the worker database
    postgres_admin_conn.execute(f"DROP DATABASE IF EXISTS {db_name}")


@pytest.fixture
def db_conn(test_database_url: str) -> Generator[psycopg.Connection, None, None]:
    """Provide a database connection for a single test.

    Uses a savepoint that rolls back after each test for isolation.
    """
    with psycopg.connect(test_database_url, autocommit=False) as conn:
        # Create a savepoint we can rollback to
        conn.execute("SAVEPOINT test_savepoint")
        try:
            yield conn
        finally:
            # Rollback to savepoint to undo test changes
            conn.execute("ROLLBACK TO SAVEPOINT test_savepoint")
            conn.commit()


@pytest.fixture(scope="session")
def s3_client() -> Generator[S3Client, None, None]:
    """Session-scoped S3 client for MinIO."""
    client = boto3.client(
        "s3",
        endpoint_url=TEST_MINIO_ENDPOINT,
        aws_access_key_id=TEST_MINIO_ACCESS_KEY,
        aws_secret_access_key=TEST_MINIO_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )

    # Ensure test bucket exists
    try:
        client.head_bucket(Bucket=TEST_MINIO_BUCKET)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=TEST_MINIO_BUCKET)

    yield client


@pytest.fixture
def s3_prefix() -> str:
    """Generate a unique S3 prefix for this test to ensure isolation."""
    worker_id = get_worker_id()
    run_id = uuid.uuid4().hex[:8]
    return f"test/{worker_id}/{run_id}"


@pytest.fixture
def s3_cleanup(s3_client: S3Client, s3_prefix: str) -> Generator[str, None, None]:
    """Provide S3 prefix and clean up objects after test."""
    yield s3_prefix

    # Cleanup: delete all objects with this prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=TEST_MINIO_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            s3_client.delete_object(Bucket=TEST_MINIO_BUCKET, Key=obj["Key"])


# Markers for test categorization
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: integration tests requiring Docker deps")
    config.addinivalue_line("markers", "unit: unit tests with no external deps")
