# MotionBricks for Blender

ポーズを置いて、その間の動きを NVIDIA MotionBricks に作らせ、Mixamo 形式の FBX として保存する Blender アドオンです。操作はすべて Blender のサイドバー（N キー）の「MotionBricks」タブで行います。

## 必要なもの

- Windows、Blender 4.2 以降（5.0 で確認）
- NVIDIA の GPU（RTX 5080 で確認）
- Git、Git LFS、uv（PATH から実行できること）
- Mixamo の骨格を持つキャラクターの FBX（`mixamorig:Hips` などのボーン名）

## インストール

```
blender -b --python tools/install_addon.py
```

Blender のユーザー拡張機能フォルダへコピーして有効にします。

## 使い方

1. **MotionBricksを起動** を押します。
   - 初回だけ、MotionBricks 本体とモデルの取得、環境の準備を自動で行います（約 5 分、約 3 GB）。2 回目以降は数秒です。
2. **ポーズ** を決めます。3D ビューに「最初」（緑）と「最後」（青）の人形が出ます。
   - 人形をドラッグ（G）・回転（R → Z）すると、位置と向きが変わります。足元の矢印が正面です。
   - 一覧の ＋ で「途中」（オレンジ）を追加、－ で削除、▲▼ で順番を入れ替えます。
   - **ポーズ集** で姿勢の元になる動きを選び、**タイミング** でその動きのどの瞬間を使うかを選びます。
   - **手足を編集** を押すと、手首・足首の白い球と、ひじ・ひざの向きを決める黄色い目印が出ます。つかんで動かす（G）と手足がついてきます。腰や胸、頭は回転（R）で動かせます。終わったら **編集を終える** を押します。**元に戻す** で編集を取り消します。
3. **生成** を押します。最初から最後までをつなぐ動きができ、再生が始まります。
   - **もう一度** で、同じポーズから長さの違う別の動きを作ります。
4. **書き出し** で保存します。
   - **キャラクター** に Mixamo のキャラクター FBX を指定します。指定すると、生成した動きがそのキャラクターで再生されます。
   - **キャラに焼き込み**：メッシュ付きの FBX。**動きのみ**：骨格だけの FBX。
   - **In Place**：生成結果とキャラクターの腰の水平移動を止め、その場で再生・書き出しします。生成前でも生成後でも切り替えられ、オフにすると元の移動に戻ります。上下動と回転は変わりません。
   - **Mixamo形式で保存** を押して保存先を選びます。

## 書き出される FBX

- 30 fps、0 秒から始まります。
- 最初のフレームで腰が原点にあり、最初のポーズの正面が前（Blender の -Y、FBX の +Z）を向きます。
- 指は Mixamo キャラクターの基本姿勢のままです（MotionBricks は指を生成しません）。
- 接地中の足が滑らないように、脚を自動で補正しています。
- キャラクターの元の FBX と、Blender 内の既存のアクションは変更しません。

## ポーズ集とライセンス

- ポーズ集（`motionbricks_blender/library/`）は、**NVIDIA Kimodo（Kimodo-SOMA-RP-v1）で生成した動き**です。プロンプトとシードは `index.json` に記録しています。作り直すには `python tools/build_library.py` を実行します（Kimodo の環境が必要）。
- **MotionBricks に同梱されているサンプル動き（`assets/motions/`）は使っていません。** これらは評価・テスト専用のライセンス（NVIDIA Sample Data Evaluation License）で、製品に使えないためです。このアドオンはサンプル動きの実体をダウンロードせず、サーバーもポーズ集以外のファイルを読み込みません。
- MotionBricks のモデルは NVIDIA Open Model Agreement、Kimodo のモデルは NVIDIA Open Model License です（どちらも商用利用可と記載されています）。利用の前に、各ライセンスの本文を確認してください。

## 保存される場所

- MotionBricks 本体と環境：`%LOCALAPPDATA%\MotionBricksBlender\`（約 3 GB）。削除すると次回の起動時に取得し直します。
- ユーザー名などに日本語が含まれる場合は、モデルファイルへのリンクを `C:\ProgramData\MotionBricksBlender\onnx\` に作ります（TensorRT が日本語を含むパスを開けないため）。

## うまくいかないとき

- 起動に失敗したら、もう一度 **MotionBricksを起動** を押してください。途中まで済んだ準備は飛ばして続きから進みます。
- サーバーの記録は `%LOCALAPPDATA%\MotionBricksBlender\server.log` にあります。
- 遠くに置いたポーズへは、間に歩きの中間目標を自動で足してつなぎます（2 m ごと）。

## 開発

```
uv run pytest                      # 純 Python のテスト
uv run pytest tests/gpu            # サーバーのテスト（GPU、MotionBricks のインストールが必要）
bash tests/bl.sh test_poses        # Blender 内のテスト（test_soma_rig / test_rig_transfer / test_poses /
                                   #   test_limb_edit / test_generate / test_export / test_ui / test_acceptance）
```

`tests/data/player_mixamo_repaired.fbx` にテスト用の Mixamo キャラクターを置いてください（リポジトリには含めません）。
