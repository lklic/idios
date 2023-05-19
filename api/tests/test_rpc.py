import pytest
from unittest.mock import MagicMock, patch
import threading
import time
from ..rpc_client import RpcClient
from ..worker import RpcServer
from commands import commands

TEST_JOB_QUEUE_NAME = "test_rpc_queue"


@pytest.fixture
def rpc_server_thread():
    rpc_server = RpcServer(queue_name=TEST_JOB_QUEUE_NAME)
    server_thread = threading.Thread(target=rpc_server.start)
    yield server_thread
    # add_callback_threadsafe is the only call that can be safely done
    # from a different thread
    rpc_server.connection.add_callback_threadsafe(rpc_server.connection.close)
    server_thread.join()


def test_rpc_interaction(rpc_server_thread):
    with patch.dict(commands, {"command": MagicMock(return_value="result")}):
        rpc_client = RpcClient(queue_name=TEST_JOB_QUEUE_NAME)

        rpc_server_thread.start()

        assert "result" == rpc_client("command", ["argument"])

        commands["command"].assert_called_once_with("argument")


def test_parallel_workers(rpc_server_thread):
    blocking_client = RpcClient(queue_name=TEST_JOB_QUEUE_NAME)
    blocking_client_thread = threading.Thread(
        target=blocking_client.__call__, args=("slow", [])
    )
    blocking_client_thread.start()

    client_threads = []
    for i in range(5):
        rpc_client = RpcClient(queue_name=TEST_JOB_QUEUE_NAME)
        client_thread = threading.Thread(target=rpc_client.__call__, args=("fast", [i]))
        client_thread.start()
        client_threads.append(client_thread)

    def fast(i):
        print(f"{i} running")

    blocked = True

    def slow():
        print("waiting")
        while blocked:
            time.sleep(0.1)
        print("unblocked")

    second_rpc_server = RpcServer(queue_name=TEST_JOB_QUEUE_NAME)
    second_server_thread = threading.Thread(target=second_rpc_server.start)

    with patch.dict(commands, dict(slow=slow, fast=fast)):
        rpc_server_thread.start()

        blocking_client_thread.is_alive()
        for client_thread in client_threads:
            assert client_thread.is_alive()

        second_server_thread.start()

        for client_thread in client_threads:
            client_thread.join()

        assert blocking_client_thread.is_alive()

        blocked = False
        blocking_client_thread.join()

    second_rpc_server.connection.add_callback_threadsafe(
        second_rpc_server.connection.close
    )
    second_server_thread.join()
