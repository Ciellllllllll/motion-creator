"""同梱ポーズ集（Kimodo で生成した動き）の一覧。bpy 非依存。"""

import json
from pathlib import Path

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
_items = None


def items() -> list:
    global _items
    if _items is None:
        index = LIBRARY_DIR / "index.json"
        _items = json.loads(index.read_text(encoding="utf-8"))["items"] if index.is_file() else []
    return _items


def item(item_id: str) -> dict:
    return next(i for i in items() if i["id"] == item_id)
