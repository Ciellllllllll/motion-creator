# MotionBricks for Blender

キャラクターの最初と最後の姿勢を指定すると、その間の動きを自動で作るBlender用の追加機能です。作った動きは、Mixamoのキャラクターに適用し、FBXファイルとして保存できます。

## 必要なもの

- WindowsとBlender 4.2以降
- NVIDIA製のGPU
- Git、Git LFS、uvをコマンドから実行できる環境
- Mixamo形式の骨格を持つキャラクターのFBXファイル

Blender 5.0とRTX 5080で動作を確認しています。

## インストール

このフォルダで、次のコマンドを実行します。

```sh
blender -b --python tools/install_addon.py
```

## 使い方

Blenderの3D画面でNキーを押し、「MotionBricks」タブを開きます。

1. 「MotionBricksを起動」を押します。初回の準備には約5分かかり、約3GBのファイルを取得します。
2. 最初と最後の人形を動かし、位置や向き、姿勢を決めます。途中の姿勢も追加できます。
3. 「生成」を押すと、姿勢をつなぐ動きが作られ、再生されます。
4. 「キャラクター」にFBXファイルを指定し、「Mixamo形式で保存」で保存します。

姿勢は「ポーズ集」から選べます。「手足を編集」では、画面上の目印を動かして姿勢を調整できます。

保存時は、キャラクターごと保存するか、動きだけを保存するかを選べます。「In Place」を有効にすると、水平方向に移動せず、その場で動くようになります。

## 保存する動きについて

毎秒30フレームで、0秒から始まる動きとして保存します。指の動きは生成されません。元のキャラクターファイルは変更しません。

## 使用するモデルとデータ

動きの生成にはNVIDIA MotionBricksを使います。ポーズ集はNVIDIA Kimodoで生成したもので、MotionBricksに付属する評価・テスト専用のサンプル動作は使っていません。

利用前に、MotionBricksのNVIDIA Open Model Agreementと、KimodoのNVIDIA Open Model Licenseを確認してください。

## 起動できないとき

「MotionBricksを起動」をもう一度押すと、準備の続きから再開します。詳しい記録は`%LOCALAPPDATA%\MotionBricksBlender\server.log`にあります。
