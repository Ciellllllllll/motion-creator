import threading

import pytest

from motionbricks_blender import setup_env as s


def _fake(tmp, fail_on=None, log=None):
    log = log if log is not None else []

    def mk(key):
        def done(home):
            return (home / f"{key}.flag").exists()

        def run(home, say):
            log.append(key)
            if fail_on and key == fail_on[0]:
                fail_on.clear()
                raise RuntimeError("boom")
            (home / f"{key}.flag").write_text("ok")

        return s.Step(key, f"ラベル{key}", 1.0, done, run)

    return [mk(k) for k in ("a", "b", "c", "d")], log


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "テスト ユーザー" / "mb"
    monkeypatch.setenv("MOTIONBRICKS_BLENDER_HOME", str(h))
    return h


def test_skips_done_steps(home, monkeypatch):
    steps, log = _fake(home)
    home.mkdir(parents=True)
    for k in "abcd":
        (home / f"{k}.flag").write_text("ok")
    monkeypatch.setattr(s, "STEPS", steps)
    s.install(lambda p, t: None, threading.Event())
    assert log == []


def test_resumes_after_failure(home, monkeypatch):
    steps, log = _fake(home, fail_on=["c"])
    monkeypatch.setattr(s, "STEPS", steps)
    with pytest.raises(s.SetupError):
        s.install(lambda p, t: None, threading.Event())
    assert log == ["a", "b", "c"]
    s.install(lambda p, t: None, threading.Event())
    assert log == ["a", "b", "c", "c", "d"]


def test_progress_monotonic(home, monkeypatch):
    steps, _ = _fake(home)
    monkeypatch.setattr(s, "STEPS", steps)
    seen = []
    s.install(lambda p, t: seen.append(p), threading.Event())
    assert seen == sorted(seen) and seen[0] >= 0.0 and seen[-1] == 1.0


def test_japanese_home_path(home):
    cmd = s.server_command(50123)
    assert str(home / "motionbricks") in cmd
    assert cmd[-2:] == ["--port", "50123"]
    assert any(c.endswith("mb_server.py") for c in cmd)


def test_cancel(home, monkeypatch):
    steps, log = _fake(home)
    monkeypatch.setattr(s, "STEPS", steps)
    ev = threading.Event()
    ev.set()
    with pytest.raises(s.SetupError) as e:
        s.install(lambda p, t: None, ev)
    assert "中止" in str(e.value) and log == []


def test_real_steps_have_japanese_labels():
    assert [st.key for st in s.STEPS] == ["tools", "clone", "checkout", "lfs", "sync"]
    assert all(any(ord(ch) > 0x3000 for ch in st.label) for st in s.STEPS)
