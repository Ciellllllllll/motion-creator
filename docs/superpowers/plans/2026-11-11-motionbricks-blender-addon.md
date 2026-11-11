# MotionBricks Blender アドオン 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Blender の N パネルだけで、MotionBricks によるポーズ間の動き生成から Mixamo 形式 FBX の書き出しまでを行えるアドオンを作る。

**Architecture:** Blender アドオン（UI・3D 表示・ポーズ編集・Mixamo 転送・FBX 書き出し）と、MotionBricks の uv 環境（Python 3.12）で動く推論サーバーの 2 プロセス構成。アドオンがサーバーを子プロセスとして起動し、127.0.0.1 の TCP で長さ付き JSON をやり取りする。ポーズ集は Kimodo で生成した SOMA 形式 BVH を同梱し、サーバー側で MotionBricks の `load_bvh` により MotionBricks の骨格へ読み込む。

**Tech Stack:** Blender 5.0（bpy, Python 3.11）、MotionBricks（NVlabs/motionbricks、コミット `54fc563aeaaf819a753d72cc9fdc6713989cdd43`、warp・TensorRT 10.8）、uv、Git LFS、Kimodo（ポーズ集の生成時のみ）。

**Spec:** `docs/superpowers/specs/2026-11-11-motionbricks-blender-addon-design.md`（本リポジトリ）。実装者は必ず設計書も読むこと。

## Global Constraints

- 新規リポジトリ `C:\Users\cielg\dev\motionbricks-blender`（git init、GitHub への公開は利用者の指示があるまでしない）。
- MotionBricks のサンプル動き（`assets/motions/` 以下の BVH/CSV）は、アドオン・サーバー・ツール・テストのどこからも読み込まない。サーバーは起動時にこのフォルダを読み込み対象から外すこと（パスを渡されたら拒否する）。
- UI のラベル・ボタン・状態表示・エラー表示はすべて日本語。説明文（ツールチップ以外の文章）はパネルに置かない。
- パネルは設計書 6.1 の並びと一致させる（見出し：MotionBricks／ポーズ／生成／書き出し）。
- ユーザーにコマンド入力、パス設定、仮想環境の操作、モデル選択を求めない。
- 生成・書き出しで利用者のキャラクター FBX ファイルと既存アクションを変更しない（新しいアクション・オブジェクトとして作る）。
- 30 fps。SOMA モデルの関節 30 個（設計書 5 章）。
- MotionBricks の作業フォルダ：`%LOCALAPPDATA%\MotionBricksBlender\`（以下 `MB_HOME`）。リポジトリは `MB_HOME\motionbricks`。
- 自動で足す中間目標の間隔：`MAX_STEP_M = 2.0`（m、水平距離）。
- 到達の判定（完成の条件 2）：各ポーズの腰の水平位置の差 10 cm 以内。
- Mixamo 往復の関節位置誤差 1 cm 以内（設計書 7 章 16）。
- Blender 側のテストは `blender -b --factory-startup --python tests/blender/run.py -- <test_module>` で実行し、失敗時は終了コード 1。純 Python のテストは Blender 同梱 Python ではなく `uv run pytest`（アドオンリポジトリの dev 環境）で実行。GPU テストは MotionBricks 環境で実行。

## Review Focus

- **サーバーが残る**：Blender を閉じる・アドオンを無効にする・Blender が落ちる場合でも MotionBricks のプロセスが残らないこと（Task 10 にテスト追加）。
- **サーバーが途中で落ちる／ポートが使用中**：日本語 1 行のエラーを出し、「起動」をもう一度押せば復帰できること（Task 2・Task 10 にテスト追加）。
- **保存して開き直す**：ポーズの配置・編集内容・生成結果が残り、サーバー未起動の状態では「起動」ボタンだけが出て、生成は起動後にそのまま押せること（Task 6 にテスト追加）。
- **Mixamo でないキャラクター、接頭辞の違うキャラクター**（`mixamorig1:`、接頭辞なし）：Mixamo でなければ日本語エラーを出し、読み込みかけのオブジェクトを残さない。接頭辞違いは正しく転送されること（Task 5・Task 9 にテスト追加）。
- **日本語を含むパス**（ユーザー名、FBX の保存先）：インストール・起動・書き出しが壊れないこと（Task 1・Task 9 にテスト追加）。

---

## ファイル構成

```
motionbricks-blender/
  motionbricks_blender/            # アドオン本体（Blender 拡張機能）
    blender_manifest.toml
    __init__.py                    # register / unregister、終了時のサーバー停止
    setup_env.py                   # インストールと起動の手順（bpy 非依存）
    client.py                      # サーバーとの通信（bpy 非依存）
    protocol.py                    # 長さ付き JSON の送受信（bpy 非依存、サーバーと共用）
    soma_rig.py                    # SOMA 骨格の作成、ポーズ配列 ⇔ Blender ポーズ
    rig_transfer.py                # Mixamo 転送・In Place（Kimodo Bridge の motion_tools.py から移植）
    poses.py                       # ポーズ一覧のデータ、ポーズ人形の表示、ポーズ集からの取得
    limb_edit.py                   # 手足の編集（IK ハンドル）と編集結果の反映
    chain.py                       # 生成の連鎖の組み立て（中間目標の追加、bpy 非依存）
    generate.py                    # 生成・再生・もう一度の operator、結果のアクション作成
    export.py                      # キャラクター読み込み、Mixamo 転送、FBX 書き出し
    ui.py                          # パネルと起動・停止 operator、進み具合の表示
    library/                       # 同梱ポーズ集
      index.json
      *.bvh
    server/
      mb_server.py                 # MotionBricks 環境で動く推論サーバー（アドオンと一緒に配布）
  tools/
    build_library.py               # Kimodo でポーズ集を生成（開発時のみ）
  tests/
    test_protocol.py  test_setup_env.py  test_chain.py      # 純 Python（pytest）
    gpu/test_server.py                                       # MotionBricks 環境（pytest）
    blender/run.py  blender/test_*.py                        # Blender（blender -b）
  docs/README.md                   # 日本語の使い方
```

---

### Task 1: リポジトリ作成・自動インストール・リスク確認

**Files:**
- Create: `motionbricks_blender/setup_env.py`, `motionbricks_blender/blender_manifest.toml`, `motionbricks_blender/__init__.py`（空の register）, `pyproject.toml`（dev 用：pytest）, `tests/test_setup_env.py`, `docs/risk-check-2026-11-11.md`

**Interfaces:**
- Produces:
  - `MB_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"`
  - `mb_home() -> pathlib.Path`（`%LOCALAPPDATA%\MotionBricksBlender`、環境変数 `MOTIONBRICKS_BLENDER_HOME` で上書き可＝テスト用）
  - `class Step(NamedTuple): key: str; label: str; weight: float; done: Callable[[Path], bool]; run: Callable[[Path, Callable[[str], None]], None]`
  - `STEPS: list[Step]`：順に `tools`（git・git-lfs・uv の確認）, `clone`, `checkout`, `lfs`, `sync`。ラベルは日本語（「ツールを確認中」「MotionBricksを取得中」「バージョンを固定中」「モデルを取得中」「環境を準備中」）。
  - `install(progress: Callable[[float, str], None], cancel: threading.Event) -> None`：`done` が真のステップは飛ばす（途中から再開できる）。失敗時は `SetupError(message_ja: str)` を送出。
  - `server_command(port: int) -> list[str]`：`["uv", "run", "--project", <MB_HOME>/motionbricks, "python", <アドオン>/server/mb_server.py, "--port", str(port)]`。

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_setup_env.py`
  - `test_skips_done_steps`：一時フォルダを `MOTIONBRICKS_BLENDER_HOME` にし、全ステップの `done` を真にした偽 `STEPS` で `install` を呼ぶと `run` が一度も呼ばれない。
  - `test_resumes_after_failure`：3 番目の `run` が例外を出す偽ステップで `install` が `SetupError` を出し、2 回目の呼び出しでは 1・2 番目が飛ばされ 3 番目から再実行される。
  - `test_progress_monotonic`：`progress` に渡る値が 0→1 で単調増加し、最後が 1.0。
  - `test_japanese_home_path`：`MOTIONBRICKS_BLENDER_HOME` を `…/テスト ユーザー/mb` にしても `server_command` の要素が正しいパスを含む。
  - `test_cancel`：`cancel` を立てると次のステップに進まず `SetupError("中止しました")`。
- [ ] **Step 2: 失敗を確認** — Run: `uv run pytest tests/test_setup_env.py -v` / Expected: FAIL（モジュールなし）
- [ ] **Step 3: `setup_env.py` を実装**。外部コマンドは `subprocess.Popen`（`CREATE_NO_WINDOW`）で実行し、出力の最終行を `progress` のラベルに流す。`done` は各ステップのマーカーファイル（`MB_HOME\.state\<key>.ok`）で判定し、`sync` の `done` は `.venv` の存在も確認する。
- [ ] **Step 4: テスト通過を確認** — Run: `uv run pytest tests/test_setup_env.py -v` / Expected: PASS
- [ ] **Step 5: 実機でインストールを実行** — Run: `uv run python -c "from motionbricks_blender import setup_env as s; s.install(print, __import__('threading').Event())"` / Expected: エラーなく完了し、`MB_HOME\motionbricks\.venv` ができる。
- [ ] **Step 6: リスク確認（設計書 8 章）**。以下を実行して結果を `docs/risk-check-2026-11-11.md` に記録する。
  - RTX 5080 で TensorRT が動くか：`MB_HOME\motionbricks` で `uv run python docs/examples/generate_once.py`（※このスクリプトはサンプル歩行クリップを読むため、**実行前に** `_rig.py` を使わない自前の確認スクリプト `tests/gpu/check_engine.py` を作り、骨格の基本姿勢 `skeleton.reference_local_transforms` の 4 ポーズを context、腰を +1 m 前に置いた同じ姿勢を target にして `generate_animation` を 1 回呼ぶ。サンプル動きは使わない）。Expected: フレーム数 18〜58 の結果が返る。失敗したら設計書 7 章 15 に従い TensorRT の版を上げて再試行し、記録する。
  - 骨格の一致：Kimodo で生成した SOMA BVH（`%USERPROFILE%\.kimodo-venv` で 2 秒の歩行を 1 本生成）を MotionBricks の `load_bvh(path, skeleton)` で読み、30 関節の名前がすべて対応すること、読み込んだポーズを FK した足首・手首の位置が Kimodo 側の FK と 2 cm 以内で一致すること（体格差は腰の高さで正規化）を確認する。一致しなければ記録し、Task 3 の変換方式を決める。
  - ライセンス確認（設計書 7 章 14）：Hugging Face の `nvidia/Kimodo-SOMA-RP-v1` のライセンス表記が NVIDIA Open Model License のままか確認して記録。変わっていたら作業を止めて利用者に報告。
- [ ] **Step 7: コミット** — `git init` 済みのリポジトリで `git add -A && git commit -m "Add resumable MotionBricks installer and record risk checks"`

---

### Task 2: 推論サーバーと通信

**Files:**
- Create: `motionbricks_blender/protocol.py`, `motionbricks_blender/client.py`, `motionbricks_blender/server/mb_server.py`, `tests/test_protocol.py`, `tests/gpu/test_server.py`

**Interfaces:**
- Consumes: `setup_env.server_command(port)`
- Produces:
  - `protocol.send(sock, obj: dict) -> None` / `protocol.recv(sock) -> dict`：4 バイト big-endian 長さ＋UTF-8 JSON。
  - ポーズ配列の表現（以後すべてのタスク共通）：`list[list[list[float]]]` 形状 `[ポーズ数, 30, 7]`、各関節は `[tx, ty, tz, qx, qy, qz, qw]` の**ローカル変換**、関節順はサーバーの `joint_names`、ワールドは Z 上（`model_to_runtime` 適用済み）、関節 0（Hips）の変換がワールドでの配置。
  - サーバーのコマンド（`{"cmd": ...}` → `{"ok": true, ...}` または `{"ok": false, "error": "<日本語>"}`）：
    - `ping` → `{"ok": true, "version": "<motionbricks の版>", "ready": bool}`（`ready` は TensorRT プラン構築済み）
    - `skeleton` → `{"joint_names": [...30], "parents": [...30], "rest": [30][7]}`
    - `sample` `{"clip": "<library 内ファイル名>", "time": float, "anchor": 0|3, "placement": [x, y, yaw_rad]}` → `{"poses": [4][30][7]}`（anchor=3：その時刻で終わる 4 ポーズ＝最初のポーズ用、anchor=0：その時刻から始まる 4 ポーズ＝目標用。公式 `sample_pose_window` と同じ意味）
    - `generate` `{"context": [4][30][7], "targets": [{"poses": [k][30][7], "use": 1..4, "early": int}], "token_counts": [int] | null}` → `{"frames": [N][30][7], "contacts": [N][4], "segments": [[start, end], ...]}`（各目標で生成し、`frames - early` まで再生したところで最後の 4 ポーズを次の context にする。`segments` は各目標の区間）
    - `shutdown`
  - `client.Client(port: int)`：`ping()`, `skeleton()`, `sample(...)`, `generate(...)`, `shutdown()`。接続できない・タイムアウト・`ok: false` は `ClientError(message_ja)`。
  - `client.start_server(port: int | None = None) -> (subprocess.Popen, int)`：空きポートを選んで起動し、`ping` が返るまで待つ（最大 15 分、TensorRT 構築中は `ready: false` を返しつつ応答する）。

- [ ] **Step 1: 失敗するテストを書く**
  - `tests/test_protocol.py::test_roundtrip_large`：`[300][30][7]` の配列を含む dict を socketpair で送受信して一致。
  - `tests/test_protocol.py::test_japanese_error`：`{"ok": false, "error": "生成に失敗しました"}` が文字化けせず届く。
  - `tests/gpu/test_server.py::test_skeleton`：`joint_names` が設計書 5 章の 30 個と同じ順序・名前。
  - `test_rejects_sample_motions`：`sample` に `"../motionbricks/assets/motions/..."` を渡すと `ok: false`（ライブラリ外のパス拒否）。
  - `test_generate_reaches_target`：基本姿勢の context と、腰を (1.5, 0) に置いた基本姿勢の target 1 個（`use: 1, early: 0`）で `generate`。最後のフレームの Hips の水平位置が (1.5, 0) から 10 cm 以内。`len(frames)` が 18〜58。
  - `test_generate_chain_two_targets`：目標 2 個で `segments` が 2 個、連続していて、全体の最後が 2 個目の目標から 10 cm 以内。
  - `test_server_survives_bad_request`：不正な JSON 形状を送ると `ok: false` が返り、続けて `ping` が成功する。
- [ ] **Step 2: 失敗を確認** — Run: `uv run pytest tests/test_protocol.py -v`（PASS 以外）、`MB_HOME\motionbricks` の環境で `uv run --project %MB_HOME%\motionbricks pytest <addon repo>/tests/gpu/test_server.py -v`（FAIL）
- [ ] **Step 3: 実装**。サーバーは公式 `docs/examples/generate_once.py` と `app/inbetweener_demo.py` の流れ（`create_input`/`create_output`、`update_from_poses`、`generate_animation`、`MotionBricksInferenceConfig(allowed_token_counts=...)`、`TargetPoseInferenceParams`）を、サンプル動きを使わずに書く。ライブラリ BVH は `load_bvh(path, skeleton)` で読み、`clip.sample(time, pose, root_offset=placement ∘ model_to_runtime)`。ポーズ配列 ⇔ warp 配列の変換を 1 か所にまとめる。1 接続 1 リクエストずつ処理する単純な TCP サーバーでよい。
- [ ] **Step 4: 通過を確認**（Step 2 と同じコマンドで PASS）
- [ ] **Step 5: コミット** — `git commit -m "Add MotionBricks inference server and client"`

---

### Task 3: ポーズ集の生成（Kimodo）

**Files:**
- Create: `tools/build_library.py`, `motionbricks_blender/library/index.json`, `motionbricks_blender/library/*.bvh`, `tests/gpu/test_library.py`

**Interfaces:**
- Produces: `index.json` の形式 `{"items": [{"id": "walk", "name": "歩き", "file": "walk.bvh", "duration": float, "default_time": float, "prompt": str, "seed": int, "model": "Kimodo-SOMA-RP-v1"}]}`。先頭は `{"id": "base", "name": "基本姿勢", ...}`（静止した基本姿勢 1 秒）。

項目（設計書 7 章 4）：基本姿勢, 立ち, 歩き, 小走り, 走り, しゃがみ, しゃがみ歩き, ジャンプ, 座る, 手を振る, 振り返る。各 3〜5 秒、30 fps。`default_time` は動きが安定した中ほどの時刻。

- [ ] **Step 1: 失敗するテストを書く** — `tests/gpu/test_library.py`
  - `test_index_complete`：11 項目、名前が上の日本語と一致、全ファイルが存在。
  - `test_all_clips_load`：サーバーの `sample` で全項目の `default_time` が取得でき、ポーズに NaN がない。
  - `test_no_sample_motion_origin`：`index.json` の全項目に `prompt` と `seed` があり、`model` が Kimodo のモデル名（＝サンプル動き由来でないことの記録）。
- [ ] **Step 2: 失敗を確認** — Run: `uv run --project %MB_HOME%\motionbricks pytest tests/gpu/test_library.py -v` / Expected: FAIL
- [ ] **Step 3: `tools/build_library.py` を実装**し、Kimodo 環境（`%USERPROFILE%\.kimodo-venv\Scripts\python.exe`）で実行して BVH を生成する。Kimodo の BVH 書き出しは Task 1 の骨格確認で決めた方式に従う（そのまま読めるなら Kimodo の SOMA BVH をそのまま、読めなければ MotionBricks 骨格へ変換して保存）。基本姿勢は MotionBricks 骨格の `rest` から作る。プロンプトは英語、名前は日本語。
- [ ] **Step 4: 通過を確認** — Step 2 と同じ / Expected: PASS
- [ ] **Step 5: コミット** — `git commit -m "Add Kimodo-generated pose library"`

---

### Task 4: SOMA 骨格とポーズ配列の変換（Blender）

**Files:**
- Create: `motionbricks_blender/soma_rig.py`, `tests/blender/run.py`, `tests/blender/test_soma_rig.py`

**Interfaces:**
- Consumes: サーバーの `skeleton` 応答（Task 2）
- Produces:
  - `build_armature(skel: dict, name: str, color: tuple[float, float, float]) -> bpy.types.Object`：関節名＝ボーン名、ワールド Z 上、ボーンの向きは子の方向（末端は親と同じ向きで 5 cm）、色はボーンカラー（カスタム色）で付ける。オブジェクトに `obj["mb_skeleton"] = json` を保存。
  - `apply_pose(arm, pose: list[list[float]]) -> None`：1 ポーズ（`[30][7]`）を pose bone に設定。
  - `read_pose(arm) -> list[list[float]]`：現在のポーズを `[30][7]` で返す。
  - `write_action(arm, frames: list, name: str, start: int = 1) -> bpy.types.Action`：`[N][30][7]` をキーフレームとして新しいアクションに書く。

- [ ] **Step 1: 失敗するテストを書く** — `tests/blender/test_soma_rig.py`（サーバーの `skeleton` 応答を JSON ファイル `tests/data/skeleton.json` に保存しておき、それを使う。Task 2 の GPU テスト実行時に保存するコードを足す）
  - `test_roundtrip_pose`：ランダムな関節回転（各関節 ±60°）と腰の位置を `apply_pose` → `read_pose` して、回転差 0.01° 以内、位置差 0.1 mm 以内。
  - `test_world_positions_match_fk`：同じポーズをサーバー側 FK（`tests/data/fk_expected.json` に保存した期待値）と比べ、各関節のワールド位置が 1 mm 以内。
  - `test_action_frames`：60 フレームの配列を `write_action` し、30 フレーム目を評価すると入力と一致。
- [ ] **Step 2: 失敗を確認** — Run: `blender -b --factory-startup --python tests/blender/run.py -- test_soma_rig` / Expected: 終了コード 1
- [ ] **Step 3: 実装**
- [ ] **Step 4: 通過を確認** — Step 2 と同じ / Expected: 終了コード 0
- [ ] **Step 5: コミット** — `git commit -m "Add SOMA armature and pose conversion"`

---

### Task 5: Mixamo 転送の移植

**Files:**
- Create: `motionbricks_blender/rig_transfer.py`, `tests/blender/test_rig_transfer.py`
- 移植元：Kimodo Bridge フォーク `motion_tools.py`（`RigTransfer`, `RestFrame`, `PoseSampler`, `write_bases`, `bones_parent_first`, `rest_relative`, `detect_mixamo_prefix`, `apply_in_place`, `find_motion_root`, `bind_action`, `_new_action_for`, `unique_action_name` とその依存）。Kimodo 固有の文言・キーは `mb_` 接頭辞に改名。

**Interfaces:**
- Produces:
  - `mb_to_mixamo(source_arm, mixamo_arm) -> RigTransfer`：対応表は `motion_tools._soma_to_mixamo_pairs()` のうち MotionBricks の 30 関節に存在するもの。手の向きの基準は `LeftHand → LeftHandMiddleEnd`。目印は `{"hips": "Hips", "head": "Head", "left_hip": "LeftLeg", "right_hip": "RightLeg", "feet": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]}`。
  - `mixamo_from_mb(mixamo_arm, source_arm) -> RigTransfer`（逆方向、往復誤差の確認用）
  - `transfer_action(source_arm, mixamo_arm, action_name: str) -> bpy.types.Action`：既存アクションは `use_fake_user=True` で残し、新しいアクションに書く。指は基本姿勢。
  - `NotMixamoError(message_ja)`：Mixamo でない骨格。
  - `apply_in_place(arm) -> tuple[bool, str]`

- [ ] **Step 1: 失敗するテストを書く** — `tests/blender/test_rig_transfer.py`（キャラクター：`tests/data/player_mixamo_repaired.fbx` に利用者の FBX をコピー。`.gitignore` に入れ、リポジトリには含めない。無ければテストは SKIP ではなく FAIL にして置き場所を表示）
  - `test_roundtrip_error_under_1cm`：SOMA 骨格に Task 4 の歩行 60 フレームを付け、Mixamo へ転送→ SOMA へ逆転送。対応する関節のワールド位置誤差の最大が 1 cm 以内（体格差は腰の高さ比で正規化）。
  - `test_prefix_variants`：`mixamorig:` を `mixamorig1:` と接頭辞なしに改名した複製でも転送でき、対応ボーン数が 22 以上。
  - `test_not_mixamo`：ボーン名を `bone_000…` に変えた骨格で `NotMixamoError`、メッセージが日本語。
  - `test_existing_action_kept`：転送前に付いていたアクションが残り、内容が変わっていない。
- [ ] **Step 2: 失敗を確認** — `blender -b --factory-startup --python tests/blender/run.py -- test_rig_transfer` / Expected: 終了コード 1
- [ ] **Step 3: 移植して実装**
- [ ] **Step 4: 通過を確認** / Expected: 終了コード 0
- [ ] **Step 5: コミット** — `git commit -m "Port Mixamo transfer for the MotionBricks skeleton"`

---

### Task 6: ポーズ一覧・ポーズ人形・ポーズ集からの取得

**Files:**
- Create: `motionbricks_blender/poses.py`, `tests/blender/test_poses.py`

**Interfaces:**
- Consumes: `soma_rig.build_armature/apply_pose/read_pose`, `client.Client.sample`
- Produces:
  - `class MBPoseEntry(PropertyGroup)`：`kind: Enum('START','MIDDLE','END')`, `library_id: Enum（index.json から動的）`, `time: Float`（スライダー、0〜`duration`、変更で再取得）, `armature: Pointer(Object)`, `window: String`（取得した 4 ポーズの JSON）, `edited: String`（編集差分の JSON、空なら未編集）
  - `class MBSceneState(PropertyGroup)`（`Scene.mb`）：`poses: Collection[MBPoseEntry]`, `index: Int`, `server_port: Int`, `status: Enum('STOPPED','INSTALLING','STARTING','RUNNING','ERROR')`, `progress: Float`, `progress_label: String`, `error: String`, `character_path: String(FILE_PATH)`, `export_mode: Enum('BAKE','MOTION')`, `in_place: Bool`, `variant: Int`, `result_action: Pointer(Action)`
  - 初期状態：起動後に一覧が空なら、最初（`立ち`）を原点、最後（`立ち`）を (0, -2 m) に作る（公式 Inbetweener の初期配置に合わせる）。
  - ポーズ人形の配置：人形オブジェクトの位置（XY）と Z 回転が配置。ユーザーがドラッグした値を `placement = [x, y, yaw]` として `sample` に渡す。人形は `MB_ポーズ` コレクションに入れ、色は 最初＝(0.24, 0.81, 0.38), 途中＝(0.88, 0.63, 0.23), 最後＝(0.31, 0.56, 0.91)。
  - operator：`mb.pose_add`（選択中の後ろに途中を追加、ポーズ集・時刻は選択中のものを複製、位置は直前と次の中間）, `mb.pose_remove`（最初・最後は不可）, `mb.pose_move(direction: 'UP'|'DOWN')`（最初・最後の位置は固定）
  - `refresh_pose(entry, client)`：`sample`（START は anchor=3、それ以外は anchor=0）→ `window` に保存 → 編集差分を適用 → 人形に表示（START は 4 ポーズ目、他は 1 ポーズ目）。
  - `entry_window(entry) -> list[[30][7]] `：編集差分を適用済みの 4 ポーズ。

- [ ] **Step 1: 失敗するテストを書く** — `tests/blender/test_poses.py`（サーバーは Task 2 の `Client` を偽物に差し替え、`tests/data/` に保存した `sample` の応答を返す）
  - `test_initial_layout`：初期化で 2 件、最初が原点、最後が (0, -2)。
  - `test_add_remove_move`：途中を 2 件追加して 4 件、最初と最後は削除・移動できない、途中同士は入れ替わる。
  - `test_time_slider_resamples`：`time` を変えると偽クライアントの `sample` が新しい時刻で呼ばれ、人形の姿勢が変わる。
  - `test_drag_changes_placement`：人形を (1, 2) に動かし 90° 回すと、次の `sample` の `placement` が `[1, 2, π/2]`。
  - `test_persist_after_reload`：保存して開き直すと、ポーズ一覧・時刻・編集差分・人形が残り、`status` は `STOPPED` に戻る。
- [ ] **Step 2: 失敗を確認** — `blender -b --factory-startup --python tests/blender/run.py -- test_poses` / Expected: 終了コード 1
- [ ] **Step 3: 実装**
- [ ] **Step 4: 通過を確認** / Expected: 終了コード 0
- [ ] **Step 5: コミット** — `git commit -m "Add pose list, pose puppets and library sampling"`

---

### Task 7: 手足の編集

**Files:**
- Create: `motionbricks_blender/limb_edit.py`, `tests/blender/test_limb_edit.py`

**Interfaces:**
- Consumes: `MBPoseEntry`, `soma_rig.read_pose`
- Produces:
  - `mb.limb_edit_begin`：選択中のポーズ人形に、両手首（`LeftHand`/`RightHand`）・両足首（`LeftFoot`/`RightFoot`）の IK ハンドル（Empty）と肘・膝のポールターゲット（Empty）を作り、IK 制約（チェーン長 2：前腕＋上腕、すね＋太もも）を付けてポーズモードに入る。腰・胸・頭は回転で直接動かす。
  - `mb.limb_edit_end`（編集を確定）：IK の結果を FK に焼き込み、ハンドルと制約を削除し、編集差分を `entry.edited` に保存。パネルの「手足を編集」はトグル表示（編集中は押し込み状態）。
  - `mb.limb_edit_reset`（元に戻す）：`edited` を空にして `refresh_pose`。
  - 編集差分の定義と適用（この方式で固定）：
    ```
    編集後の表示ポーズ P'、編集前の表示ポーズ P（START は window[3]、他は window[0]）について
      関節 j ごとに  d_j = rot(P'_j) * rot(P_j)^-1          （ローカル回転の差）
      腰の位置差     d_root = pos(P'_0) - pos(P_0)
    window の 4 ポーズ全部に  rot(W_k,j) ← d_j * rot(W_k,j)、pos(W_k,0) ← pos(W_k,0) + d_root
    ```
    これで選んだ瞬間の勢い（4 ポーズ間の差）を保ったまま姿勢だけ変わる。

- [ ] **Step 1: 失敗するテストを書く** — `tests/blender/test_limb_edit.py`
  - `test_drag_foot_moves_knee`：右足首ハンドルを 10 cm 前に動かして確定すると、右足首のワールド位置が 1 cm 以内で目標に一致し、右膝も動いている。左脚は変わらない。
  - `test_edit_keeps_velocity`：歩きの中ほどのポーズで左手首を動かして確定。window の 4 ポーズ間の腰の移動量が編集前と同じ（差 1 mm 以内）。
  - `test_reset`：元に戻すと `edited` が空になり、姿勢が編集前と一致。
  - `test_no_leftovers`：確定後、ハンドル・ポールの Empty と IK 制約が残っていない。
- [ ] **Step 2: 失敗を確認** / Expected: 終了コード 1
- [ ] **Step 3: 実装**
- [ ] **Step 4: 通過を確認** / Expected: 終了コード 0
- [ ] **Step 5: コミット** — `git commit -m "Add limb editing with IK handles"`

---

### Task 8: 生成（連鎖の組み立て・結果の表示・もう一度）

**Files:**
- Create: `motionbricks_blender/chain.py`, `motionbricks_blender/generate.py`, `tests/test_chain.py`, `tests/blender/test_generate.py`

**Interfaces:**
- Consumes: `entry_window`, `Client.generate`, `Client.sample`, `soma_rig.write_action`, `rig_transfer.transfer_action`
- Produces:
  - `chain.build_request(windows: list[list], kinds: list[str], walk_sampler: Callable[[float, float, float], list], variant: int) -> dict`（bpy 非依存）：
    - context＝最初の window（4 ポーズ）。
    - 途中ポーズ → 目標 `{"use": 4, "early": 0}`、最後 → `{"use": 1, "early": 0}`。
    - 直前の目標（または最初）との腰の水平距離が `MAX_STEP_M` を超える区間には、等間隔（各 `MAX_STEP_M` 以下）に中間目標を足す。中間目標の姿勢は `walk_sampler(x, y, yaw)`（ポーズ集「歩き」の `default_time` を、進行方向を向けて配置した 4 ポーズ）で、`{"use": 2, "early": 8}`。
    - `token_counts`：`variant == 0` なら `None`（モデル既定）、それ以外は `[[8, 14], [10, 14], [6, 12], [12, 16]][(variant - 1) % 4]` の範囲の整数リスト。
  - `generate.py` の operator：
    - `mb.generate`（生成）：全ポーズの window を集めて `build_request` → 別スレッドで `Client.generate` → 終わったら `MB_結果` 骨格（なければ作成、色は白）に `write_action(name="MB_生成_###")`。キャラクター FBX が指定されていれば Task 9 の `ensure_character()` で読み込み、`transfer_action` してキャラクターで再生。ポーズ人形は生成後に非表示、ポーズを選ぶと再表示。シーンのフレーム範囲を結果に合わせ、再生を開始。
    - `mb.play`（再生）：フレーム 1 から再生／停止を切り替え。
    - `mb.regenerate`（もう一度）：`variant += 1` して `mb.generate`。
  - 生成中は「生成」ボタンを押せない。失敗時は `MBSceneState.error` に日本語 1 行。

- [ ] **Step 1: 失敗するテストを書く**
  - `tests/test_chain.py::test_no_intermediate_when_close`：最初 (0,0)、最後 (0,-2) → 目標 1 個。
  - `test_intermediates_when_far`：最初 (0,0)、途中 (0,-5)、最後 (3,-5) → 0→-5 の間に中間目標 2 個（間隔 ≤ 2.0 m）、-5→(3,-5) の間に 1 個。中間目標は `use: 2, early: 8`、元の途中は `use: 4, early: 0`、最後は `use: 1`。
  - `test_variant_token_counts`：`variant=0` で `None`、`variant=1` で `list(range(8, 15))`、`variant=5` も同じ。
  - `tests/blender/test_generate.py::test_generate_end_to_end`（GPU、実サーバー起動）：最初・途中 2・最後をポーズ集から置き（途中 1 つは手足を編集）、生成。結果アクションの各ポーズ区間（`segments`）の最後で、腰の水平位置が対応するポーズから 10 cm 以内。キャラクター指定時は Mixamo キャラクターに新しいアクションが付き、元の FBX の更新日時が変わっていない。
  - `test_generate_from_base_pose`：全ポーズを「基本姿勢」から手足編集だけで作っても生成できる（完成の条件 3）。
  - `test_regenerate_differs`：もう一度で結果のフレーム数か姿勢が変わる。
  - `test_server_dies_midway`：生成中にサーバープロセスを強制終了すると、`status == 'ERROR'`、エラーが日本語、既存の結果アクションが壊れていない。
- [ ] **Step 2: 失敗を確認** — `uv run pytest tests/test_chain.py -v` と `blender -b --factory-startup --python tests/blender/run.py -- test_generate` / Expected: FAIL / 終了コード 1
- [ ] **Step 3: 実装**
- [ ] **Step 4: 通過を確認** / Expected: PASS / 終了コード 0
- [ ] **Step 5: 足の補正の要否を判断**（設計書 7 章 11）：`test_generate_end_to_end` の結果を Mixamo キャラクターで目視し、足滑り（接地確信度 > 0.5 の区間の足の移動）を測る。平均 3 cm/s を超えるなら、Kimodo Bridge の `motion_refine.apply_foot_lock` を移植して自動適用するタスクを追加し、利用者に一言伝える。判断と数値を `docs/risk-check-2026-11-11.md` に追記。
- [ ] **Step 6: コミット** — `git commit -m "Add chained generation, playback and regenerate"`

---

### Task 9: キャラクター読み込みと FBX 書き出し

**Files:**
- Create: `motionbricks_blender/export.py`, `tests/blender/test_export.py`

**Interfaces:**
- Consumes: `rig_transfer.transfer_action`, `rig_transfer.apply_in_place`, `MBSceneState`
- Produces:
  - `ensure_character(path: str) -> bpy.types.Object`：FBX を `MB_キャラクター` コレクションに読み込む（同じパス・更新日時なら再利用）。Mixamo でなければ読み込んだものを全部削除して `NotMixamoError`。
  - `mb.export`（Mixamo形式で保存）：Blender 標準の保存ダイアログ（`ExportHelper`、拡張子 `.fbx`、初期ファイル名 `MB_<日時>.fbx`）。
    - 書き出す動き：結果アクションを複製し、最初のフレームの腰の水平位置が原点に来るよう平行移動。In Place ならさらに `apply_in_place`。元の結果アクションは変えない。
    - `BAKE`：キャラクターの骨格とメッシュを選択して書き出し。`MOTION`：骨格のみ。
    - FBX 設定（固定）：`use_selection=True, bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, add_leaf_bones=False, primary_bone_axis='Y', secondary_bone_axis='X', axis_forward='-Z', axis_up='Y', apply_scale_options='FBX_SCALE_ALL', bake_anim_simplify_factor=0.0`、シーンの fps は 30。
    - キャラクター未指定のときは保存ボタンを押せない（`poll` が偽）。パネル側で欄を赤く表示（`alert=True`）。

- [ ] **Step 1: 失敗するテストを書く** — `tests/blender/test_export.py`
  - `test_bake_export_reimport`：書き出した FBX を新しいシーンに読み込むと、`mixamorig:` ボーンとメッシュがあり、10 フレームごとの全ボーンのワールド位置が書き出し前と 1 mm 以内（原点への平行移動分を除く）。
  - `test_motion_only_has_no_mesh`：`MOTION` で書き出した FBX にメッシュがなく、骨格の動きは同じ。
  - `test_in_place`：In Place で書き出すと腰の水平移動が 0（1 mm 以内）、高さは同じ。
  - `test_starts_at_origin`：最初のフレームの腰の水平位置が (0, 0) から 1 mm 以内。
  - `test_source_untouched`：書き出し後、キャラクター FBX ファイルの更新日時・サイズ、結果アクションのキーが変わっていない。
  - `test_japanese_path`：保存先 `…/書き出し テスト/歩き.fbx` で書き出せて読み込める。
  - `test_not_mixamo_cleanup`：Mixamo でない FBX を指定するとエラーになり、シーンのオブジェクト数が指定前と同じ。
- [ ] **Step 2: 失敗を確認** / Expected: 終了コード 1
- [ ] **Step 3: 実装**
- [ ] **Step 4: 通過を確認** / Expected: 終了コード 0
- [ ] **Step 5: コミット** — `git commit -m "Add character import and Mixamo FBX export"`

---

### Task 10: パネル・起動／停止・後片付け

**Files:**
- Create: `motionbricks_blender/ui.py`, `tests/blender/test_ui.py`
- Modify: `motionbricks_blender/__init__.py`（全モジュールの register、`bpy.app.handlers.load_post` で `status` を戻す、`atexit` とアドオン無効化でサーバー停止）

**Interfaces:**
- Consumes: すべての operator、`setup_env.install`, `client.start_server`
- Produces:
  - パネル `MB_PT_main`（タブ「MotionBricks」）とサブパネル `MB_PT_poses`（ポーズ）, `MB_PT_generate`（生成）, `MB_PT_export`（書き出し）。起動前と起動中は `MB_PT_main` だけ表示。表示文言は設計書 6.1 のとおり：「▶ MotionBricksを起動」「中止」「✔ 起動中」「■ 停止」「ポーズ集」「タイミング」「✋ 手足を編集」「↺ 元に戻す」「▶ 生成」「⏵ 再生」「⟳ もう一度」「キャラに焼き込み」「動きのみ」「キャラクター」「📌 In Place」「💾 Mixamo形式で保存」。一覧の項目名は「最初」「途中 1…」「最後」。アイコンは Blender 標準アイコンで代用してよい（絵文字が描画できない場合）。
  - `mb.start`（起動）：別スレッドで `install`（進み具合を `progress`/`progress_label` へ）→ `start_server` → `skeleton` 取得 → 初期ポーズ作成。`bpy.app.timers` で UI を更新。
  - `mb.cancel`（中止）、`mb.stop`（停止：`shutdown` → 5 秒で終わらなければ強制終了）。
  - エラーは `MB_PT_main` に赤字 1 行（`error`）と「▶ MotionBricksを起動」ボタン（再試行）。

- [ ] **Step 1: 失敗するテストを書く** — `tests/blender/test_ui.py`
  - `test_panel_order_and_labels`：偽の `layout` で各状態の draw を呼び、出てくるラベルの並びが設計書 6.1 と一致。ASCII のみのラベルがない（「In Place」「Mixamo」を含むラベルは日本語との組み合わせなので許可リストで扱う）。
  - `test_states`：`STOPPED` では起動ボタンだけ、`INSTALLING` では進み具合と中止、`RUNNING` で全サブパネル、`ERROR` で赤字＋起動ボタン。
  - `test_server_killed_on_disable`（GPU）：起動 → アドオン無効化 → 10 秒以内にサーバープロセスが存在しない。
  - `test_port_in_use_and_restart`（GPU）：起動中にサーバーを強制終了 → `status == 'ERROR'` → 起動を押すと別ポートで復帰して `RUNNING`。
  - `test_second_launch_under_1min`（GPU）：インストール済み・プラン構築済みの状態で、起動ボタンから `RUNNING` まで 60 秒以内（完成の条件 1）。
- [ ] **Step 2: 失敗を確認** / Expected: 終了コード 1
- [ ] **Step 3: 実装**
- [ ] **Step 4: 通過を確認** / Expected: 終了コード 0
- [ ] **Step 5: コミット** — `git commit -m "Add Japanese panel, launch and shutdown"`

---

### Task 11: 受け入れ確認・説明書・インストール

**Files:**
- Create: `docs/README.md`（日本語：インストール方法、使い方の流れ、書き出し形式、ライセンス上の注意＝サンプル動き不使用とポーズ集の出所）, `tests/blender/test_acceptance.py`, `tools/install_addon.py`（Blender 5.0 のユーザー拡張フォルダへコピー）
- Modify: `docs/risk-check-2026-11-11.md`（最終結果）

- [ ] **Step 1: 受け入れテストを書く** — `tests/blender/test_acceptance.py`：設計書 9 章の 1〜7 を順に自動実行（新しいホームフォルダ `MOTIONBRICKS_BLENDER_HOME` を一時フォルダにした完全な初回インストールを含む）。条件 6 は、サーバーとアドオンのファイルアクセスを記録するフック（`builtins.open` と `load_bvh` のラッパー）で `assets/motions` へのアクセスが 0 回であることを確認。
- [ ] **Step 2: 実行** — `blender -b --factory-startup --python tests/blender/run.py -- test_acceptance` / Expected: 終了コード 0
- [ ] **Step 3: 目視確認（条件 8）**：受け入れテストで書き出した FBX を Blender で開き、Blender MCP の `look` で正面・横から各 12 フレーム確認。膝の逆曲がり・大きな足滑り・体の突き抜けを記録（MotionBricks 自体の癖と変換で生じたものを区別）。
- [ ] **Step 4: 全テスト再実行** — 純 Python・GPU・Blender のすべて / Expected: 全 PASS
- [ ] **Step 5: 利用者の Blender にインストール** — `blender -b --python tools/install_addon.py` / Expected: 拡張機能一覧に「MotionBricks」が有効で表示
- [ ] **Step 6: コミット** — `git commit -m "Add acceptance test, Japanese README and installer"`
