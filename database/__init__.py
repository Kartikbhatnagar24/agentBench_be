from typing import Optional, Any
import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase.client import ClientOptions

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

url: str = os.environ["SUPABASE_URL"]
key: str = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(
    url,
    key,
    options=ClientOptions(
        postgrest_client_timeout=10,
        storage_client_timeout=10,
        schema="public",
    ),
)


def _apply_filters(builder, filters: dict):
    """
    Applies a dictionary of equality filters to a Postgrest query builder.
    Each key-value pair becomes a `.eq(column, value)` call.
    """
    for column, value in filters.items():
        builder = builder.eq(column, value)
    return builder


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------

def insert_one(table_name: str, data: dict) -> Optional[Any]:
    """
    Inserts a single record into *table_name*.

    Args:
        table_name: The target table.
        data:       A dict whose keys are column names.

    Returns:
        The inserted record as a dict, or None on failure.
    """
    try:
        response = supabase.table(table_name).insert(data).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error("insert_one failed on '%s': %s", table_name, e)
        raise


def insert_many(table_name: str, data: list[dict]) -> list[Any]:
    """
    Inserts multiple records into *table_name* in a single call.

    Args:
        table_name: The target table.
        data:       A list of dicts to insert.

    Returns:
        A list of inserted records.
    """
    try:
        response = supabase.table(table_name).insert(data).execute()
        return response.data or []
    except Exception as e:
        logger.error("insert_many failed on '%s': %s", table_name, e)
        raise


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def find_one(
    table_name: str,
    filters: dict,
    select: str = "*",
) -> Optional[Any]:
    """
    Returns the first record that matches *filters*, or None if not found.

    Args:
        table_name: The target table.
        filters:    Dict of {column: value} equality conditions.
        select:     Comma-separated columns to return (default all).

    Returns:
        A single record dict or None.
    """
    try:
        builder = supabase.table(table_name).select(select)
        builder = _apply_filters(builder, filters)
        response = builder.limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error("find_one failed on '%s': %s", table_name, e)
        raise


def find_many(
    table_name: str,
    filters: Optional[dict] = None,
    select: str = "*",
    order_by: Optional[str] = None,
    descending: bool = False,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> list[Any]:
    """
    Returns all records that match *filters*.

    Args:
        table_name:  The target table.
        filters:     Optional dict of {column: value} equality conditions.
        select:      Comma-separated columns to return (default all).
        order_by:    Column name to sort by.
        descending:  If True, sorts in descending order.
        limit:       Maximum number of records to return.
        offset:      Number of records to skip (for pagination).

    Returns:
        A list of record dicts.
    """
    try:
        builder = supabase.table(table_name).select(select)
        if filters:
            builder = _apply_filters(builder, filters)
        if order_by:
            builder = builder.order(order_by, desc=descending)
        if limit is not None:
            builder = builder.limit(limit)
        if offset is not None:
            builder = builder.offset(offset)
        response = builder.execute()
        return response.data or []
    except Exception as e:
        logger.error("find_many failed on '%s': %s", table_name, e)
        raise


def find_by_id(
    table_name: str,
    record_id: Any,
    select: str = "*",
    id_column: str = "id",
) -> Optional[Any]:
    """
    Convenience wrapper – fetches a single record by its primary key.

    Args:
        table_name:  The target table.
        record_id:   The primary key value.
        select:      Columns to return (default all).
        id_column:   Name of the PK column (default "id").

    Returns:
        The matching record dict or None.
    """
    return find_one(table_name, {id_column: record_id}, select=select)


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------

def update_one(
    table_name: str,
    data: dict,
    filters: Optional[dict] = None,
    record_id: Optional[Any] = None,
    id_column: str = "id",
) -> Optional[Any]:
    """
    Updates the first record matching *record_id* or *filters*.

    Provide either *record_id* (shorthand for ``{id_column: record_id}``) or
    a *filters* dict.  If both are given, *record_id* takes precedence.

    Args:
        table_name:  The target table.
        data:        Dict of {column: new_value} pairs to update.
        filters:     Dict of {column: value} equality conditions.
        record_id:   Primary-key value (uses *id_column* as the column name).
        id_column:   Name of the PK column (default "id").

    Returns:
        The updated record dict or None.
    """
    try:
        builder = supabase.table(table_name).update(data)
        if record_id is not None:
            builder = builder.eq(id_column, record_id)
        elif filters:
            builder = _apply_filters(builder, filters)
        else:
            raise ValueError("update_one requires either 'record_id' or 'filters'.")
        response = builder.execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error("update_one failed on '%s': %s", table_name, e)
        raise


def update_many(
    table_name: str,
    data: dict,
    filters: dict,
) -> list[Any]:
    """
    Updates all records matching *filters*.

    Args:
        table_name: The target table.
        data:       Dict of {column: new_value} pairs to update.
        filters:    Dict of {column: value} equality conditions (required –
                    prevents accidental full-table updates).

    Returns:
        A list of updated record dicts.
    """
    if not filters:
        raise ValueError("update_many requires at least one filter to prevent full-table updates.")
    try:
        builder = supabase.table(table_name).update(data)
        builder = _apply_filters(builder, filters)
        response = builder.execute()
        return response.data or []
    except Exception as e:
        logger.error("update_many failed on '%s': %s", table_name, e)
        raise


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

def delete_one(
    table_name: str,
    filters: Optional[dict] = None,
    record_id: Optional[Any] = None,
    id_column: str = "id",
) -> Optional[Any]:
    """
    Deletes the first record matching *record_id* or *filters*.

    Args:
        table_name:  The target table.
        filters:     Dict of {column: value} equality conditions.
        record_id:   Primary-key value shorthand.
        id_column:   Name of the PK column (default "id").

    Returns:
        The deleted record dict or None.
    """
    try:
        builder = supabase.table(table_name).delete()
        if record_id is not None:
            builder = builder.eq(id_column, record_id)
        elif filters:
            builder = _apply_filters(builder, filters)
        else:
            raise ValueError("delete_one requires either 'record_id' or 'filters'.")
        response = builder.execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error("delete_one failed on '%s': %s", table_name, e)
        raise


def delete_many(
    table_name: str,
    filters: dict,
) -> list[Any]:
    """
    Deletes all records matching *filters*.

    Args:
        table_name: The target table.
        filters:    Dict of {column: value} equality conditions (required –
                    prevents accidental full-table deletes).

    Returns:
        A list of deleted record dicts.
    """
    if not filters:
        raise ValueError("delete_many requires at least one filter to prevent full-table deletes.")
    try:
        builder = supabase.table(table_name).delete()
        builder = _apply_filters(builder, filters)
        response = builder.execute()
        return response.data or []
    except Exception as e:
        logger.error("delete_many failed on '%s': %s", table_name, e)
        raise


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def count(
    table_name: str,
    filters: Optional[dict] = None,
) -> int:
    """
    Returns the number of records matching *filters* (or total rows if none).

    Args:
        table_name: The target table.
        filters:    Optional dict of {column: value} equality conditions.

    Returns:
        Row count as an integer.
    """
    try:
        builder = supabase.table(table_name).select("*", count="exact")
        if filters:
            builder = _apply_filters(builder, filters)
        response = builder.execute()
        return response.count or 0
    except Exception as e:
        logger.error("count failed on '%s': %s", table_name, e)
        raise


def exists(
    table_name: str,
    filters: dict,
) -> bool:
    """
    Returns True if at least one record matching *filters* exists.

    Args:
        table_name: The target table.
        filters:    Dict of {column: value} equality conditions.

    Returns:
        True if a matching record exists, False otherwise.
    """
    return find_one(table_name, filters) is not None


# ---------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------

def storage_upload(
    bucket: str,
    path: str,
    file: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    """
    Upload raw bytes to a Supabase Storage bucket.

    Args:
        bucket:       The storage bucket name (e.g. "rag_project").
        path:         The destination path inside the bucket.
        file:         Raw bytes to upload.
        content_type: MIME type of the file (default: application/octet-stream).

    Raises:
        HTTPException 500 on failure (re-raised for FastAPI handlers).
        Exception for all other callers.
    """
    try:
        supabase.storage.from_(bucket).upload(
            path=path,
            file=file,
            file_options={"content-type": content_type},
        )
    except Exception as e:
        logger.error("storage_upload failed (bucket=%s, path=%s): %s", bucket, path, e)
        raise


def storage_download(bucket: str, path: str) -> bytes:
    """
    Download a file from a Supabase Storage bucket.

    Args:
        bucket: The storage bucket name.
        path:   The file path inside the bucket.

    Returns:
        Raw bytes of the downloaded file.
    """
    try:
        return supabase.storage.from_(bucket).download(path)
    except Exception as e:
        logger.error("storage_download failed (bucket=%s, path=%s): %s", bucket, path, e)
        raise


def storage_list(bucket: str, prefix: str) -> list:
    """
    List files inside a bucket under a given path prefix.

    Args:
        bucket: The storage bucket name.
        prefix: The folder prefix to list (e.g. "user_id/session_id").

    Returns:
        A list of file metadata dicts returned by Supabase.
    """
    try:
        return supabase.storage.from_(bucket).list(prefix) or []
    except Exception as e:
        logger.error("storage_list failed (bucket=%s, prefix=%s): %s", bucket, prefix, e)
        raise


def storage_delete(bucket: str, paths: list[str]) -> None:
    """
    Delete one or more files from a Supabase Storage bucket.

    Args:
        bucket: The storage bucket name.
        paths:  List of file paths inside the bucket to remove.
    """
    try:
        supabase.storage.from_(bucket).remove(paths)
    except Exception as e:
        logger.error("storage_delete failed (bucket=%s, paths=%s): %s", bucket, paths, e)
        raise
