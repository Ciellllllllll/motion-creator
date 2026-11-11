import json
import pathlib

import pytest

from motionbricks_blender import client

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def server():
    proc, port = client.start_server()
    c = client.Client(port)
    yield c
    try:
        c.shutdown()
    except client.ClientError:
        pass
    proc.wait(timeout=20)


@pytest.fixture(scope="session")
def skel(server):
    s = server.skeleton()
    DATA.mkdir(exist_ok=True)
    (DATA / "skeleton.json").write_text(json.dumps(s), encoding="utf-8")
    return s
