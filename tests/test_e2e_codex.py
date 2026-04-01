"""E2E tests: codex CLI → codex-proxy (Rust binary) → upstream provider.

These tests spawn the compiled Rust proxy binary on a random port, generate a
Codex config.toml from the shared template under config/codex/, then run
`codex exec` to verify the full request path works for Python and Rust coding
tasks.

Prerequisites:
    - `codex` binary in PATH (codex-cli >= 0.117.0)
    - Rust proxy binary at `target/debug/codex-proxy`
    - Valid Z.AI credentials for the configured local account

Marked with pytest.mark.e2e_codex so they can be run selectively.
"""

import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUST_BINARY = PROJECT_ROOT / "target" / "debug" / "codex-proxy"
TEMPLATE_CONFIG = PROJECT_ROOT / "config" / "codex" / "config.e2e.toml"
CODEX_BIN = shutil.which("codex")
E2E_PROXY_KEY = "cpk_e2e_admin_local"

CODEX_TIMEOUT = 180  # seconds — generous for real provider round-trips


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_codex_config(
    port: int, config_path: Path, model: str = "glm-5-turbo"
) -> None:
    """Write a codex config.toml that routes through the proxy on *port*."""
    template = TEMPLATE_CONFIG.read_text()
    rendered = template.replace("{port}", str(port)).replace(
        'model = "glm-5-turbo"', f'model = "{model}"', 1
    )
    config_path.write_text(rendered)


def _wait_for_proxy(port: int, timeout: float = 15.0) -> None:
    """Poll until the proxy responds on the health/root endpoint."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return
        except ConnectionRefusedError:
            time.sleep(0.3)
        except urllib.error.HTTPError as exc:
            if exc.code == 200:
                return
            time.sleep(0.3)
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    raise TimeoutError(f"Proxy did not start on port {port} within {timeout}s")


def _load_project_env() -> dict[str, str]:
    """Load .env from project root without overriding the current process env."""
    env = os.environ.copy()
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    return env


def _require_codex_cli_support(codex_binary: str) -> None:
    """Fail fast if the installed Codex CLI no longer matches this harness."""
    version = subprocess.run(
        [codex_binary, "--version"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(PROJECT_ROOT),
    )
    if version.returncode != 0:
        pytest.skip(f"Unable to read `codex --version`: {version.stderr.strip()}")

    help_result = subprocess.run(
        [codex_binary, "exec", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(PROJECT_ROOT),
    )
    if help_result.returncode != 0:
        pytest.skip(f"Unable to read `codex exec --help`: {help_result.stderr.strip()}")

    help_text = help_result.stdout
    required_flags = [
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c, --config <key=value>",
    ]
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        pytest.skip(
            "Installed Codex CLI no longer supports required e2e flags: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def codex_binary():
    """Skip the whole module if codex is not available or CLI flags changed."""
    if not CODEX_BIN:
        pytest.skip("codex binary not found in PATH")
    _require_codex_cli_support(CODEX_BIN)
    yield CODEX_BIN


@pytest.fixture(scope="module")
def rust_binary():
    """Skip if Rust proxy binary is not built."""
    if not RUST_BINARY.exists():
        pytest.skip(
            f"Rust proxy binary not found at {RUST_BINARY}; run `cargo build` first"
        )
    yield RUST_BINARY


@pytest.fixture(scope="module")
def proxy_env():
    """Environment for launching the proxy from the repo root."""
    env = _load_project_env()
    if "CODEX_PROXY_ZAI_API_KEY" not in env and "ZAI_API_KEY" in env:
        env["CODEX_PROXY_ZAI_API_KEY"] = env["ZAI_API_KEY"]
    if not env.get("CODEX_PROXY_ZAI_API_KEY"):
        pytest.skip(
            "Z.AI credentials not found. Set CODEX_PROXY_ZAI_API_KEY or ZAI_API_KEY."
        )
    env["CODEX_PROXY_LOG_LEVEL"] = "info"
    return env


@pytest.fixture(scope="module")
def proxy_port(rust_binary, proxy_env):
    """Start the Rust proxy on a random port, return the port number."""
    port = _free_port()
    env = proxy_env.copy()
    env["CODEX_PROXY_PORT"] = str(port)

    proc = subprocess.Popen(
        [str(rust_binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    try:
        _wait_for_proxy(port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture(scope="module")
def codex_config_dir(proxy_port, tmp_path_factory):
    """Create a temp directory with codex config pointing at the proxy."""
    config_dir = tmp_path_factory.mktemp("codex-e2e-config")
    config_path = config_dir / "config.toml"
    _build_codex_config(proxy_port, config_path)
    return config_dir


@pytest.fixture(scope="module")
def codex_env(codex_config_dir, proxy_env):
    """Environment for Codex subprocesses."""
    env = proxy_env.copy()
    env["CODEX_HOME"] = str(codex_config_dir)
    env["CODEX_PROXY_API_KEY"] = E2E_PROXY_KEY
    return env


def _run_codex(
    codex_binary: str,
    config_dir: Path,
    codex_env: dict[str, str],
    prompt: str,
    workdir: Path,
    model: str | None = None,
    timeout: int = CODEX_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run `codex exec` with the e2e config."""
    cmd = [
        codex_binary,
        "exec",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c",
        f'config_file="{config_dir / "config.toml"}"',
        "-c",
        'model_provider="codex-proxy"',
    ]
    if model:
        cmd.extend(["-c", f'model="{model}"'])
    cmd.append(prompt)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(workdir),
        env=codex_env,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestPreflight:
    """Check the shared config path and auth wiring before longer scenarios."""

    def test_shared_template_exists(self):
        assert TEMPLATE_CONFIG.exists(), f"Missing shared Codex template: {TEMPLATE_CONFIG}"
        template = TEMPLATE_CONFIG.read_text()
        assert 'env_key = "CODEX_PROXY_API_KEY"' in template
        assert 'requires_openai_auth = false' in template
        assert '{port}' in template

    def test_codex_proxy_smoke(self, codex_binary, codex_config_dir, codex_env, tmp_path):
        workdir = tmp_path / "smoke"
        workdir.mkdir()

        result = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            "Create a file called smoke.txt containing exactly OK and nothing else. Do not run any extra commands.",
            workdir,
            timeout=90,
        )

        smoke_file = workdir / "smoke.txt"
        assert smoke_file.exists(), (
            f"smoke.txt was not created.\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}\n"
            f"returncode: {result.returncode}"
        )
        assert smoke_file.read_text().strip() == "OK"


class TestPythonCoding:
    """Verify codex can complete a Python coding task through the proxy."""

    def test_python_hello_world(self, codex_binary, codex_config_dir, codex_env, tmp_path):
        """Ask codex to write a simple Python script, verify the file exists."""
        workdir = tmp_path / "python_test"
        workdir.mkdir()

        result = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            "Create a file called hello.py that prints 'Hello from codex-proxy e2e' and nothing else. Do not run it.",
            workdir,
        )

        hello_py = workdir / "hello.py"
        if not hello_py.exists():
            pytest.fail(
                f"hello.py was not created.\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}\n"
                f"returncode: {result.returncode}"
            )

        content = hello_py.read_text()
        assert "Hello from codex-proxy e2e" in content, (
            f"Expected greeting in hello.py.\n"
            f"Got:\n{content}\n"
            f"stdout: {result.stdout[-500:]}"
        )

    def test_python_with_test(self, codex_binary, codex_config_dir, codex_env, tmp_path):
        """Ask codex to write a Python module + test, then run the test."""
        workdir = tmp_path / "python_with_test"
        workdir.mkdir()

        result = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            textwrap.dedent("""\
                Write a Python module `adder.py` with a function `add(a, b)` that returns a + b.
                Then write a test file `test_adder.py` that tests add() with at least 3 cases.
                Then run `python -m pytest test_adder.py -v` and make sure all tests pass.
            """),
            workdir,
            timeout=CODEX_TIMEOUT + 60,
        )

        test_file = workdir / "test_adder.py"
        module_file = workdir / "adder.py"

        assert module_file.exists(), (
            f"adder.py was not created.\nstdout: {result.stdout[-2000:]}"
        )
        assert test_file.exists(), (
            f"test_adder.py was not created.\nstdout: {result.stdout[-2000:]}"
        )

        content = module_file.read_text()
        assert "def add" in content

        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_adder.py", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workdir),
        )
        assert test_result.returncode == 0, (
            f"Tests did not pass.\n"
            f"stdout: {test_result.stdout}\n"
            f"stderr: {test_result.stderr}"
        )


class TestRustCoding:
    """Verify codex can complete a Rust coding task through the proxy."""

    def test_rust_hello_world(self, codex_binary, codex_config_dir, codex_env, tmp_path):
        """Ask codex to init a minimal Rust project and verify Cargo.toml."""
        workdir = tmp_path / "rust_test"
        workdir.mkdir()

        result = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            "Run `cargo init --name e2e_test` in the current directory. That is the only thing you need to do.",
            workdir,
            timeout=60,
        )

        cargo_toml = workdir / "Cargo.toml"
        if not cargo_toml.exists():
            pytest.fail(
                f"Cargo.toml was not created.\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}\n"
                f"returncode: {result.returncode}"
            )

        content = cargo_toml.read_text()
        assert "e2e_test" in content

    def test_rust_simple_lib(self, codex_binary, codex_config_dir, codex_env, tmp_path):
        """Ask codex to write a Rust lib function and run tests."""
        workdir = tmp_path / "rust_lib_test"
        workdir.mkdir()

        result = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            textwrap.dedent("""\
                Initialize a Rust library project with `cargo init --lib --name mathlib`.
                In src/lib.rs, implement a public function `multiply(a: i32, b: i32) -> i32` that returns a * b.
                In the default test module, write 3 tests for multiply().
                Then run `cargo test` and make sure all tests pass.
            """),
            workdir,
            timeout=CODEX_TIMEOUT + 120,
        )

        lib_rs = workdir / "src" / "lib.rs"
        cargo_toml = workdir / "Cargo.toml"

        assert cargo_toml.exists(), (
            f"Cargo.toml not created.\nstdout: {result.stdout[-2000:]}"
        )
        assert lib_rs.exists(), (
            f"src/lib.rs not created.\nstdout: {result.stdout[-2000:]}"
        )

        content = lib_rs.read_text()
        assert "multiply" in content

        test_result = subprocess.run(
            ["cargo", "test"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(workdir),
        )
        assert test_result.returncode == 0, (
            f"cargo test failed.\n"
            f"stdout: {test_result.stdout[-2000:]}\n"
            f"stderr: {test_result.stderr[-2000:]}"
        )


class TestMultiTurn:
    """Verify multi-turn interaction (conversation chaining) through the proxy."""

    def test_python_two_turns(self, codex_binary, codex_config_dir, codex_env, tmp_path):
        """First turn: create a file. Second turn: modify it."""
        workdir = tmp_path / "multi_turn"
        workdir.mkdir()

        result1 = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            "Create a file `counter.py` with a class Counter that has increment() and get() methods.",
            workdir,
        )

        assert (workdir / "counter.py").exists(), (
            f"counter.py not created in turn 1.\nstdout: {result1.stdout[-1000:]}"
        )

        result2 = _run_codex(
            codex_binary,
            codex_config_dir,
            codex_env,
            "Read counter.py and add a reset() method to the Counter class.",
            workdir,
        )

        content = (workdir / "counter.py").read_text()
        assert "reset" in content.lower(), (
            f"reset() not found after turn 2.\n"
            f"Content:\n{content}\n"
            f"stdout: {result2.stdout[-1000:]}"
        )
