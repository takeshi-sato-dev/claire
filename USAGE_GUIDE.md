# CLAIRE 使い方ガイド

## セットアップ

### 1. config.pyの設定

まず、`config.py`を編集してトポロジーとトラジェクトリファイルのパスを設定します：

```python
# config.py

# ===== Input Files =====
TOPOLOGY_FILE = "/path/to/your/system.psf"
TRAJECTORY_FILE = "/path/to/your/trajectory.xtc"

# 例：EphA2の場合
# TOPOLOGY_FILE = "/Users/takeshi/data/epha2/system.psf"
# TRAJECTORY_FILE = "/Users/takeshi/data/epha2/trajectory.xtc"

# ===== Frame Selection =====
FRAME_START = 20000    # 開始フレーム
FRAME_STOP = 50000     # 終了フレーム（None = 最後まで）
FRAME_STEP = 10        # フレームステップ

# 例：全フレームを解析する場合
# FRAME_START = 0
# FRAME_STOP = None
# FRAME_STEP = 1
```

### 2. 脂質タイプの設定（オプション）

解析する脂質タイプも設定できます：

```python
# config.py

# ===== Lipid types =====
DEFAULT_LIPID_TYPES = ['CHOL', 'DIPC', 'DPSM']  # 解析する脂質
TARGET_LIPID = 'DPG3'  # メディエーター脂質（例：GM3）
```

## 基本的な使い方

### config.pyを使う場合（推奨）

`config.py`にファイルパスを設定しておけば、シンプルなコマンドで実行できます：

```bash
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output epha2_results \
    --parallel
```

### コマンドラインで指定する場合

`config.py`を使わない場合は、コマンドラインで全て指定できます：

```bash
python run_claire.py \
    --topology /path/to/system.psf \
    --trajectory /path/to/trajectory.xtc \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output results \
    --parallel
```

## 主要なオプション

### 必須パラメータ
- `--lipids`: 解析する脂質タイプ（スペース区切り）
- `--target-lipid`: メディエーター脂質名（例：GM3はDPG3）

### よく使うオプション
- `--output DIR`: 出力ディレクトリ（デフォルト: claire_output）
- `--parallel`: 並列処理を有効化
- `--n-workers N`: 並列処理のワーカー数
- `--cutoff FLOAT`: コンタクトカットオフ（Å、デフォルト: 15.0）

### フレーム選択
- `--start N`: 開始フレーム（デフォルト: config.FRAME_START）
- `--stop N`: 終了フレーム（デフォルト: config.FRAME_STOP）
- `--step N`: フレームステップ（デフォルト: config.FRAME_STEP）

**ヒント**: `config.py`で設定しておけば、毎回指定する必要はありません

### 解析のスキップ
- `--skip-temporal`: 時間発展解析をスキップ
- `--skip-spatial`: 空間解析をスキップ
- `--skip-ml`: 機械学習解析をスキップ

## 出力ファイル

### データファイル
- `composition_data.csv`: フレームごとの組成データ
- `temporal_windows.csv`: スライディングウィンドウ組成
- `frame_data.pkl`: キャッシュデータ（再解析用）

### 図（PNGとSVG両方）
全ての図は**PNGとSVG両方**で保存されます：

- `composition_changes.png` / `.svg`: 組成変化の棒グラフ
- `temporal_composition.png` / `.svg`: 時間変化
- `radial_profiles.png` / `.svg`: 動径プロファイル
- `ml_predictions.png` / `.svg`: 機械学習予測
- `comprehensive_summary.png` / `.svg`: 総合図

**利点**：
- PNG: プレゼンテーション、論文のドラフト用
- SVG: 最終論文、編集可能なベクター形式

## 使用例

### 例1: EphA2の完全解析

```bash
# config.pyで設定済みの場合
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output epha2_full \
    --parallel \
    --n-workers 8
```

### 例2: Notchの高速解析（空間・ML解析スキップ）

```bash
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output notch_quick \
    --parallel \
    --skip-spatial \
    --skip-ml
```

### 例3: フレームサブセットのテスト

```bash
# config.pyの設定を上書き
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output test_run \
    --start 0 \
    --stop 100 \
    --step 10

# または、config.pyで設定済みならそのまま実行
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output test_run
```

### 例4: 上部リーフレットのみ解析

```bash
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output upper_leaflet \
    --leaflet upper \
    --parallel
```

## Python APIの使い方

スクリプトから使う場合：

```python
import claire

# トラジェクトリ読み込み
u = claire.load_universe('system.psf', 'trajectory.xtc')
upper, lower = claire.identify_lipid_leaflets(u)

# 組成解析
analyzer = claire.CompositionAnalyzer(['CHOL', 'DIPC', 'DPSM'], target_lipid='DPG3')
df = analyzer.frames_to_dataframe(frame_data_list)
df = analyzer.calculate_conservation_ratios(df)
results = analyzer.analyze_composition_changes(df)

# 時間発展解析
temporal = claire.TemporalAnalyzer(window_size=100, step_size=10)
window_df = temporal.sliding_window_composition(df, ['CHOL', 'DIPC', 'DPSM'])

# 空間解析
spatial = claire.SpatialAnalyzer(radii=[5.0, 10.0, 15.0, 20.0])
radial_data = spatial.calculate_radial_composition(u, frame_idx, proteins, lipids, leaflet, lipid_types)

# 機械学習
predictor = claire.CompositionPredictor()
ml_results = predictor.predict_composition_changes(df, mediator_cols, lipid_types)
```

## トラブルシューティング

### エラー: "No topology file specified"
→ `config.py`で`TOPOLOGY_FILE`を設定するか、`--topology`オプションを使用

### エラー: "No trajectory file specified"
→ `config.py`で`TRAJECTORY_FILE`を設定するか、`--trajectory`オプションを使用

### 並列処理が動かない
→ `--n-workers`を減らす、または`--parallel`なしで実行

### メモリ不足
→ `--step`を大きくしてフレーム数を減らす、または`--stop`で範囲を制限

## パフォーマンスのヒント

1. **並列処理**: 大規模トラジェクトリには`--parallel`を使用
2. **キャッシュ**: `frame_data.pkl`が生成されたら再解析は高速
3. **フレーム選択**: まず少数フレームでテスト、その後全体を解析
4. **解析のスキップ**: 不要な解析は`--skip-*`でスキップ

## 次のステップ

- 出力CSVファイルで詳細データを確認
- PNGファイルでプレゼンテーション資料作成
- SVGファイルで論文用の図を編集
- 異なるパラメータで再解析（キャッシュから高速）
