"""Abstract interface for request persistence."""

from __future__ import annotations

import abc
import contextlib
import os
import time
from typing import (AsyncGenerator, Generator, List, Optional, Set, Tuple,
                    TYPE_CHECKING)

from sky.skylet import constants

if TYPE_CHECKING:
    from sky.server import daemons as daemons_lib
    from sky.server.requests.requests import Request
    from sky.server.requests.requests import RequestStatus
    from sky.server.requests.requests import RequestTaskFilter
    from sky.server.requests.requests import StatusWithMsg


class RequestBackend(abc.ABC):
    """Abstract interface for request persistence and lifecycle."""

    @abc.abstractmethod
    def get_request(self,
                    request_id: str,
                    fields: Optional[List[str]] = None) -> Optional[Request]:
        """Get a request by ID with appropriate locking."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_request_async(
            self,
            request_id: str,
            fields: Optional[List[str]] = None) -> Optional[Request]:
        """Async version of get_request."""
        raise NotImplementedError

    @abc.abstractmethod
    @contextlib.contextmanager
    def update_request(
            self, request_id: str) -> Generator[Optional[Request], None, None]:
        """Atomic read-modify-write with appropriate locking.

        Yields the request object. Caller modifies it in-place. On context
        exit, the modified request is persisted. If the request doesn't exist,
        yields None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    @contextlib.asynccontextmanager
    async def update_request_async(
            self, request_id: str) -> AsyncGenerator[Optional[Request], None]:
        """Async version of update_request."""
        del request_id
        yield None

    @abc.abstractmethod
    async def create_if_not_exists_async(self, request: Request) -> bool:
        """Create a request if it does not exist.

        Returns:
            True if a new request was created, False if it already exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def create_or_refresh_internal_daemon_async(
            self, request: 'Request') -> bool:
        """For an internal daemon request: insert a fresh PENDING row or
        refresh env-bearing columns on an existing row.

        Returns True if a new row was inserted (caller should enqueue
        the request onto the task queue), False if an existing row was
        refreshed in-place (the task_queue entry from the original
        creator stays in place; do NOT enqueue again).

        Atomic + idempotent under concurrent callers. Replaces
        `create_if_not_exists_async` on the daemon submission path:
        the dedup contract is identical (exactly one concurrent caller
        gets True), but losing callers also UPDATE `request_body`,
        `name`, and `schedule_type` on the existing row so the
        persisted `env_vars` reflect the current process's
        `os.environ` rather than whatever the original creator
        captured (which may be from a previous deployment generation
        in HA setups).
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_orphan_internal_daemons_async(
        self,
        internal_daemons: List['daemons_lib.InternalRequestDaemon'],
    ) -> None:
        """Delete daemon-shaped rows whose `request_id` is not in
        `internal_daemons` (daemon was renamed / removed in code),
        along with any task_queue entries (for backends with a
        persistent queue).

        Idempotent under concurrent callers.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query_requests(self, req_filter: RequestTaskFilter) -> List[Request]:
        """Query requests matching the filter."""
        raise NotImplementedError

    @abc.abstractmethod
    async def query_requests_async(
            self, req_filter: RequestTaskFilter) -> List[Request]:
        """Async version of query_requests."""
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_requests(self, request_ids: List[str]) -> None:
        """Delete requests by their IDs."""
        raise NotImplementedError

    @abc.abstractmethod
    async def update_status_async(self, request_id: str,
                                  status: RequestStatus) -> None:
        """Update the status of a request."""
        raise NotImplementedError

    @abc.abstractmethod
    async def update_status_msg_async(self, request_id: str,
                                      status_msg: str) -> None:
        """Update the status message of a request."""
        raise NotImplementedError

    @abc.abstractmethod
    def kill_requests(self,
                      request_ids: Optional[List[str]] = None,
                      user_id: Optional[str] = None) -> List[str]:
        """Kill requests and set their status to CANCELLED.

        Returns:
            A list of request IDs that were cancelled.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def kill_request_async(self, request_id: str) -> bool:
        """Kill a single request and set its status to cancelled.

        Returns:
            True if the request was killed, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_latest_request_id_async(self) -> Optional[str]:
        """Get the most recent request ID."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_requests_with_prefix(
            self,
            request_id_prefix: str,
            fields: Optional[List[str]] = None) -> Optional[List[Request]]:
        """Get all requests matching an ID prefix."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_requests_async_with_prefix(
            self,
            request_id_prefix: str,
            fields: Optional[List[str]] = None) -> Optional[List[Request]]:
        """Async version of get_requests_with_prefix."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_request_status_async(
            self,
            request_id: str,
            include_msg: bool = False) -> Optional[StatusWithMsg]:
        """Get the status (and optionally status_msg) of a request."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_api_request_ids_start_with(self,
                                             incomplete: str) -> List[str]:
        """Get request IDs for shell completion."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_active_file_mounts_blob_ids(self) -> Set[str]:
        """Get blob IDs referenced by active (PENDING/RUNNING) requests."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_shutdown_active_requests(self) -> List[Tuple[str, str]]:
        """Get (request_id, name) pairs to wait for during graceful shutdown."""
        raise NotImplementedError

    def mark_running(self, request_id: str, pid: int) -> bool:
        """Transition a request to RUNNING with the given pid.

        Returns True if the transition succeeded (row was in PENDING/WAITING).
        Default implementation uses read-modify-write; backends with
        efficient targeted UPDATEs should override.
        """
        with self.update_request(request_id) as request:
            if request is None:
                return False
            if request.status not in (RequestStatus.PENDING,
                                      RequestStatus.WAITING):
                return False
            request.status = RequestStatus.RUNNING
            request.pid = pid
            request.status_msg = None
            return True

    async def mark_running_async(self, request_id: str, pid: int) -> bool:
        """Async version of mark_running."""
        async with self.update_request_async(request_id) as request:
            if request is None:
                return False
            if request.status not in (RequestStatus.PENDING,
                                      RequestStatus.WAITING):
                return False
            request.status = RequestStatus.RUNNING
            request.pid = pid
            request.status_msg = None
            return True

    def mark_succeeded(self, request_id: str, return_value) -> None:
        """Transition a RUNNING request to SUCCEEDED with a return value.

        Default implementation uses read-modify-write; backends with
        efficient targeted UPDATEs should override.
        """
        with self.update_request(request_id) as request:
            if request is not None:
                request.status = RequestStatus.SUCCEEDED
                request.finished_at = time.time()
                if return_value is not None:
                    request.set_return_value(return_value)

    async def mark_succeeded_async(self, request_id: str, return_value) -> None:
        """Async version of mark_succeeded."""
        async with self.update_request_async(request_id) as request:
            if request is not None:
                request.status = RequestStatus.SUCCEEDED
                request.finished_at = time.time()
                if return_value is not None:
                    request.set_return_value(return_value)

    def mark_failed(self, request_id: str, error: BaseException) -> None:
        """Transition a RUNNING request to FAILED with an error.

        Default implementation uses read-modify-write; backends with
        efficient targeted UPDATEs should override.
        """
        with self.update_request(request_id) as request:
            if request is not None:
                request.status = RequestStatus.FAILED
                request.finished_at = time.time()
                request.set_error(error)

    async def mark_failed_async(self, request_id: str,
                                error: BaseException) -> None:
        """Async version of mark_failed."""
        async with self.update_request_async(request_id) as request:
            if request is not None:
                request.status = RequestStatus.FAILED
                request.finished_at = time.time()
                request.set_error(error)

    def reset_on_startup(self) -> None:
        """Called on server startup for backend-specific initialization."""

    def try_acquire_daemon_leader_lock(self) -> bool:
        """Try to acquire the daemon leader lock.

        In multi-replica setups, only the leader should run internal daemons
        (status refresh, volume refresh, etc.) to avoid duplicate work and
        DB contention. Single-process backends (SQLite) always return True.

        Returns:
            True if this instance is the daemon leader.
        """
        return True

    async def close(self) -> None:
        """Release resources (engines, connections, etc.)."""


_storage_backend: Optional[RequestBackend] = None


def get_request_backend() -> RequestBackend:
    """Get the registered request backend."""
    global _storage_backend
    if _storage_backend is None:
        # pylint: disable=import-outside-toplevel
        from sky.server.requests.requests import PostgresRequestBackend
        from sky.server.requests.requests import SqliteRequestBackend

        backend = os.environ.get(constants.ENV_VAR_API_REQUEST_DB_BACKEND, '')
        if backend.lower() == 'postgres':
            _storage_backend = PostgresRequestBackend()
        else:
            _storage_backend = SqliteRequestBackend()
    return _storage_backend


def set_request_backend(backend: RequestBackend) -> None:
    """Set the request backend."""
    global _storage_backend
    _storage_backend = backend
