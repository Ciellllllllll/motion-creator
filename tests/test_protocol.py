import random
import socket

from motionbricks_blender import protocol


def _pair():
    return socket.socketpair()


def test_roundtrip_large():
    a, b = _pair()
    rnd = random.Random(0)
    frames = [[[round(rnd.uniform(-1, 1), 6) for _ in range(7)] for _ in range(77)] for _ in range(300)]
    import threading
    t = threading.Thread(target=protocol.send, args=(a, {"ok": True, "frames": frames}))
    t.start()
    got = protocol.recv(b)
    t.join()
    assert got["frames"] == frames


def test_japanese_error():
    a, b = _pair()
    protocol.send(a, {"ok": False, "error": "生成に失敗しました"})
    assert protocol.recv(b) == {"ok": False, "error": "生成に失敗しました"}


def test_closed_connection_raises():
    a, b = _pair()
    a.close()
    try:
        protocol.recv(b)
        assert False
    except ConnectionError:
        pass
