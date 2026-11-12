"""標準入力から HF トークンを受け取り、権限を確認してからローカル認証へ保存する。

トークンは表示・ログ出力しない。コマンド引数にも載せない。
使い方: <トークンファイル> を標準入力へ流し込む
    Get-Content <file> -Raw | python -I hf_login_check.py
"""
import sys

from huggingface_hub import HfApi, login

BASE_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_REV = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"


def main():
    token = sys.stdin.read().strip()
    if not token:
        sys.exit("トークンが空です")
    api = HfApi(token=token)
    me = api.whoami()
    print("account:", me.get("name"))
    role = (me.get("auth") or {}).get("accessToken", {}).get("role")
    print("token role:", role)
    info = api.model_info(BASE_REPO, revision=BASE_REV, files_metadata=True)
    print("base revision ok:", info.sha)
    # 実ファイルの取得権限（ゲート承認）を小さなファイルで確認する
    path = api.hf_hub_download(BASE_REPO, "config.json", revision=BASE_REV,
                               cache_dir=None, local_dir=None)
    print("gated file download ok:", bool(path))
    login(token=token, add_to_git_credential=False)
    print("stored in local HF auth")


if __name__ == "__main__":
    main()
