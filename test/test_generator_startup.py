import contextlib
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generator_ipc  # noqa: E402


class GeneratorStartupTest(unittest.TestCase):

    def test_client_preserves_state_published_during_another_startup(self):
        # The real server publishes its port before its PID. Interleave a
        # second client's probe at precisely that boundary, without sleeps.
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown), \
                    tempfile.TemporaryDirectory() as temp_dir:
                pid_file = str(Path(temp_dir) / "server.pid")
                port_file = str(Path(temp_dir) / "server.port")
                stderr = io.StringIO()
                probe_results = []

                def publish_server(*args):
                    generator_ipc._write_server_port_file(port_file, 3701)
                    probe_results.append(generator_ipc.run_generator_client(
                        "127.0.0.1", 1, [], temp_dir, "generator.py", shutdown,
                        auto_start=False, pid_file=pid_file,
                        port_file=port_file, port_range="3701-3710"))
                    generator_ipc._write_pid_file(pid_file, os.getpid())

                response = {"returncode": 0, "stdout": "", "stderr": ""}
                with contextlib.redirect_stderr(stderr), \
                        mock.patch.object(generator_ipc,
                                          "_start_generator_server",
                                          side_effect=publish_server), \
                        mock.patch.object(generator_ipc,
                                          "_ping_generator_server"), \
                        mock.patch.object(generator_ipc,
                                          "_connect_and_request",
                                          return_value=response):
                    result = generator_ipc.run_generator_client(
                        "127.0.0.1", 1, [], temp_dir, "generator.py", False,
                        server_program="generator.py", pid_file=pid_file,
                        port_file=port_file, port_range="3701-3710")

                self.assertEqual(0, result, stderr.getvalue())
                self.assertEqual([0 if shutdown else 1], probe_results)
                self.assertEqual(3701,
                                 generator_ipc._read_server_port_file(port_file))
                self.assertEqual(os.getpid(),
                                 generator_ipc._read_pid_file(pid_file))

    def test_startup_lock_excludes_other_processes_and_recovers_after_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_file = str(Path(temp_dir) / "server.startup.lock")
            code = """
import os
import sys
sys.path.insert(0, sys.argv[1])
import generator_ipc
lock = generator_ipc._acquire_generator_server_startup_lock(
    sys.argv[2], 1)
if not lock:
    raise RuntimeError("could not acquire startup lock")
print("locked:{0}".format(os.getpid()), flush=True)
sys.stdin.readline()
"""
            # Kill the actual lock owner, not a Windows venv redirector that
            # could leave its interpreter child alive until stdin closes.
            interpreter = generator_ipc._get_process_image_path(os.getpid())
            with subprocess.Popen(
                    [interpreter or sys.executable, "-c", code,
                     str(SCRIPT_DIR), lock_file],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                    **generator_ipc.get_subprocess_no_window_kwargs()) as owner:
                try:
                    self.assertEqual("locked:{0}\n".format(owner.pid),
                                     owner.stdout.readline())
                    contender = generator_ipc._acquire_generator_server_startup_lock(
                        lock_file, 0.1)
                    if contender:
                        generator_ipc._release_generator_server_startup_lock(contender)
                    self.assertFalse(contender)
                finally:
                    owner.kill()
                    owner.wait(timeout=5)

            recovered = generator_ipc._acquire_generator_server_startup_lock(
                lock_file, 1)
            self.assertTrue(recovered)
            generator_ipc._release_generator_server_startup_lock(recovered)

            reacquired = generator_ipc._acquire_generator_server_startup_lock(
                lock_file, 1)
            self.assertTrue(reacquired)
            generator_ipc._release_generator_server_startup_lock(reacquired)

    def test_concurrent_clients_start_one_server_and_complete_requests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "generator.py"
            pid_file = str(Path(temp_dir) / "server.pid")
            port_file = str(Path(temp_dir) / "server.port")
            # Exercise real auto-start (including pythonw on Windows), file
            # locking, sockets and shutdown. Only generation is a tiny fixture.
            fixture.write_text("""
import argparse
import os
import sys
sys.path.insert(0, {script_dir!r})
import generator_ipc
parser = argparse.ArgumentParser()
parser.add_argument('--server-mode', action='store_true')
parser.add_argument('--server-address', default='127.0.0.1')
parser.add_argument('--server-port-range', default='0')
parser.add_argument('--server-idle-timeout', default='3')
parser.add_argument('--server-pid-file', default={pid_file!r})
parser.add_argument('--server-port-file', default={port_file!r})
options = parser.parse_args()
if options.server_mode:
    def generate(argv=None, display_argv=None, allow_ipc=False):
        print(os.getpid())
        return 0
    result = generator_ipc.run_generator_server(
        options.server_address, options.server_idle_timeout,
        lambda request: generator_ipc.run_generation_request(request, generate),
        options.server_pid_file, options.server_port_file,
        options.server_port_range)
else:
    print('ready', flush=True)
    sys.stdin.readline()
    result = generator_ipc.run_generator_client(
        options.server_address, 5, [], os.getcwd(), __file__, False,
        idle_timeout=3, server_program=__file__,
        pid_file=options.server_pid_file, port_file=options.server_port_file,
        port_range=options.server_port_range)
sys.exit(result)
""".format(script_dir=str(SCRIPT_DIR), pid_file=pid_file,
           port_file=port_file), encoding="utf-8")

            clients = []
            try:
                for _ in range(8):
                    clients.append(subprocess.Popen(
                        [sys.executable, str(fixture)], cwd=os.getcwd(),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True,
                        **generator_ipc.get_subprocess_no_window_kwargs()))
                for client in clients:
                    self.assertEqual("ready\n", client.stdout.readline())
                for client in clients:
                    client.stdin.write("go\n")
                    client.stdin.flush()
                server_pids = set()
                for client in clients:
                    stdout, stderr = client.communicate(timeout=15)
                    self.assertEqual(0, client.returncode, stderr)
                    server_pids.add(int(stdout.strip()))
                self.assertEqual(1, len(server_pids))
                self.assertEqual(server_pids.pop(),
                                 generator_ipc._read_pid_file(pid_file))
                self.assertIsNotNone(
                    generator_ipc._read_server_port_file(port_file))
            finally:
                for client in clients:
                    if client.poll() is None:
                        client.kill()
                    client.communicate(timeout=5)
                server_pid = generator_ipc._read_pid_file(pid_file)
                generator_ipc.run_generator_client(
                    "127.0.0.1", 1, [], temp_dir, str(fixture), True,
                    auto_start=False, pid_file=pid_file, port_file=port_file,
                    port_range="0")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if not os.path.exists(pid_file) and not os.path.exists(port_file) \
                            and (server_pid is None or
                                 generator_ipc._get_process_image_path(server_pid) is None):
                        break
                    time.sleep(0.01)
                self.assertFalse(os.path.exists(pid_file))
                self.assertFalse(os.path.exists(port_file))
                if server_pid is not None:
                    self.assertIsNone(generator_ipc._get_process_image_path(server_pid))


if __name__ == "__main__":
    unittest.main()
