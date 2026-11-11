"""長さ付き JSON の送受信（4 バイト big-endian の長さ + UTF-8 JSON）。"""

import json
import struct

MAX_BYTES = 256 * 1024 * 1024


def send(sock, obj: dict) -> None:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _read(sock, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(1 << 20, n - got))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv(sock) -> dict:
    (size,) = struct.unpack(">I", _read(sock, 4))
    if size > MAX_BYTES:
        raise ConnectionError("message too large")
    return json.loads(_read(sock, size).decode("utf-8"))
