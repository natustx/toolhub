"""Test the test fixtures themselves work correctly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_database_connection(db_conn):
    """Verify database connection and schema are set up."""
    # Check we can query
    result = db_conn.execute("SELECT 1 as test").fetchone()
    assert result[0] == 1

    # Check tables exist
    result = db_conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """).fetchall()
    table_names = [row[0] for row in result]

    expected_tables = [
        "chunks",
        "entities",
        "entity_types",
        "evidence",
        "source_artifacts",
        "sources",
    ]
    for table in expected_tables:
        assert table in table_names, f"Missing table: {table}"


def test_database_extensions(db_conn):
    """Verify required extensions are loaded."""
    result = db_conn.execute("""
        SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')
    """).fetchall()
    extensions = {row[0] for row in result}

    assert "vector" in extensions, "pgvector extension not loaded"
    assert "pg_trgm" in extensions, "pg_trgm extension not loaded"


def test_s3_connection(s3_client, s3_cleanup):
    """Verify S3/MinIO connection works."""
    from tests.conftest import TEST_MINIO_BUCKET

    # Upload a test object
    test_key = f"{s3_cleanup}/test-object.txt"
    s3_client.put_object(
        Bucket=TEST_MINIO_BUCKET,
        Key=test_key,
        Body=b"test content",
    )

    # Read it back
    response = s3_client.get_object(Bucket=TEST_MINIO_BUCKET, Key=test_key)
    content = response["Body"].read()
    assert content == b"test content"


def test_database_isolation(db_conn):
    """Verify test isolation via transaction rollback."""
    # Insert data
    db_conn.execute("""
        INSERT INTO sources (canonical_url, source_type, collection)
        VALUES ('https://test.com', 'website', 'test')
    """)

    # Verify it's there in this test
    result = db_conn.execute("SELECT COUNT(*) FROM sources").fetchone()
    assert result[0] == 1

    # After test, transaction rolls back, so next test starts clean


def test_database_isolation_verify(db_conn):
    """Verify previous test's data was rolled back."""
    # Should be empty because previous test's transaction was rolled back
    result = db_conn.execute("SELECT COUNT(*) FROM sources").fetchone()
    assert result[0] == 0, "Data from previous test leaked through"
