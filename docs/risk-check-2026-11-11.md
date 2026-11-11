# リスク確認の記録（2026-11-11）

設計書 8 章の確認結果。

## 1. RTX 5080 で TensorRT が動くか → 動く

- `tests/gpu/check_engine.py`（サンプル動きは使わない。基本姿勢 4 ポーズを context、腰をずらした基本姿勢を target）
- TensorRT 10.8.0.43 のまま、依存関係の変更なし。
- 初回のプラン構築 85.5 秒。生成 1 回 4〜5 ミリ秒（初回のみ約 1.2 秒）。
- 目標 (1,0)・(0,1)・(0,-1.5) のいずれも 50 フレームで、最後の腰の位置が目標と一致。NaN なし。

## 2. 骨格の一致 → 完全に一致

- MotionBricks の SOMA 骨格（`soma_base_skel_minimal_no_root.usd`）は **77 関節**で、Kimodo の `somaskel77` と同じ関節名。モデルが使うのは 30 関節だが、ポーズ配列は 77 関節。
  - **計画との違い**：計画ではポーズ配列を `[n][30][7]` としていたが、実際は `[n][77][7]`。以後 77 を使う。
- `tests/gpu/check_kimodo_bvh.py`：Kimodo の BVH（標準 T ポーズ**オフ**＝SOMA ネイティブ形式）を `load_bvh(path, skeleton)` で読むと、不足関節 0、腰・頭・手・足の位置差 0.00 cm。
  - Kimodo の BVH にある `Root` 関節は読み飛ばされる（Kimodo の Root は原点固定）。
  - ポーズ集は Kimodo の BVH をそのまま同梱できる。変換は不要。

## 3. Kimodo モデルのライセンス → 変更なし

- Hugging Face `nvidia/Kimodo-SOMA-RP-v1`：`license_name: nvidia-open-model-license`。モデルカードに "This model is ready for commercial use." と記載（2026-11-11 確認）。

## 4. サンプル動きの扱い

- インストール時に `git lfs pull --exclude "assets/motions/**,docs/**"` とし、サンプル動きの実体はダウンロードしない（ポインタファイルのみ）。

## 5. 足の補正の要否（計画タスク 8 ステップ 5）→ 必要。自動でかける

`tests/blender/measure_slide.py` で、接地の確信度 0.5 超の区間の足首の水平移動を測った。

| 条件 | SOMA の結果 | Mixamo へ転送後（補正なし） |
|---|---|---|
| 最初・途中 2・最後（歩き、しゃがみを含む） | 平均 12.1 cm/s | 平均 8.1 cm/s |
| 6 m 離した立ち→立ち（中間目標つき） | 平均 8.7 cm/s | 平均 5.9 cm/s |

基準（平均 3 cm/s）を超えたので、Kimodo Bridge の足の固定処理を `foot_lock.py` に移植し、
Mixamo キャラクターへの転送後に自動でかける。接地中に足が床へ届かないフレームでは腰を最大 4 cm 下げる。
補正後（6 m の条件）：母指球の水平移動 平均 0.03 cm/s、接地中の足裏の高さ 0〜0.17 cm、腰の道筋は変化なし。
パネルには設定を出さない。
