# CLAIRE 変更内容まとめ

## 実装した変更

### 1. config.pyでのファイルパスとフレーム範囲の設定

**変更内容**:
- `config.py`に`TOPOLOGY_FILE`と`TRAJECTORY_FILE`を追加
- `FRAME_START`, `FRAME_STOP`, `FRAME_STEP`を追加
- デフォルト値としてこれらを使用可能に
- コマンドラインオプションでの上書きも可能

**使い方**:
```python
# config.py
TOPOLOGY_FILE = "/path/to/system.psf"
TRAJECTORY_FILE = "/path/to/trajectory.xtc"

# Frame selection
FRAME_START = 20000  # 開始フレーム
FRAME_STOP = 50000   # 終了フレーム（None = 全て）
FRAME_STEP = 10      # フレームステップ
```

```bash
# config.pyの設定を使う
python run_claire.py --lipids CHOL DIPC DPSM --target-lipid DPG3

# または直接指定
python run_claire.py --topology system.psf --trajectory traj.xtc --lipids CHOL DIPC DPSM
```

### 2. PNG/SVG両方での図の保存

**変更内容**:
- 新しい関数`save_figure()`を追加（`visualization/plots.py`）
- 全てのプロット関数を修正してPNGとSVG両方で保存
- `config.py`で`FIGURE_FORMATS = ['png', 'svg']`を設定

**効果**:
```
出力例:
  ✓ Saved composition_changes.png
  ✓ Saved composition_changes.svg
  ✓ Saved temporal_composition.png
  ✓ Saved temporal_composition.svg
  ...
```

**利点**:
- PNG: プレゼンテーション、ドラフト用（300 DPI高解像度）
- SVG: 論文用、編集可能なベクター形式

## 変更されたファイル

### 1. config.py
```python
# 追加された設定
TOPOLOGY_FILE = None
TRAJECTORY_FILE = None
FRAME_START = 0
FRAME_STOP = None
FRAME_STEP = 1
FIGURE_FORMATS = ['png', 'svg']  # 元: FIGURE_FORMAT
```

### 2. visualization/plots.py
```python
# 新規追加
def save_figure(fig, output_path, formats=['png', 'svg'], dpi=300):
    """Save figure in multiple formats"""
    # PNG, SVG両方で保存

# 変更: 全てのプロット関数
def plot_composition_changes(...):
    ...
    if output_path:
        save_figure(fig, output_path)  # 変更前: plt.savefig()
```

### 3. run_claire.py
```python
# 追加: config.pyからの設定読み込み
topology = args.topology if args.topology else TOPOLOGY_FILE
trajectory = args.trajectory if args.trajectory else TRAJECTORY_FILE
frame_start = args.start if args.start is not None else FRAME_START
frame_stop = args.stop if args.stop is not None else FRAME_STOP
frame_step = args.step if args.step is not None else FRAME_STEP

# バリデーション追加
if topology is None:
    print("ERROR: No topology file specified...")
```

### 4. visualization/__init__.py
```python
# save_figureをエクスポートに追加
from .plots import (..., save_figure)
__all__ = [..., 'save_figure']
```

### 5. 新規ファイル
- `USAGE_GUIDE.md`: 日本語の詳細使用ガイド

## 互換性

### 後方互換性: ✅ 完全に保持

既存の使い方も全て動作します：

```bash
# これまで通り動作
python run_claire.py --topology system.psf --trajectory traj.xtc ...

# 新しい方法も使える
python run_claire.py --lipids CHOL DIPC DPSM ...  # config.pyから読み込み
```

## 使用方法の改善

### 以前
```bash
python run_claire.py \
    --topology /very/long/path/to/system.psf \
    --trajectory /very/long/path/to/trajectory.xtc \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output results \
    --parallel
```

### 現在（config.py設定後）
```bash
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output results \
    --parallel
```

### さらに、config.pyで全て設定すれば
```python
# config.py
TOPOLOGY_FILE = "/path/to/system.psf"
TRAJECTORY_FILE = "/path/to/trajectory.xtc"
FRAME_START = 20000
FRAME_STOP = 50000
FRAME_STEP = 10
DEFAULT_LIPID_TYPES = ['CHOL', 'DIPC', 'DPSM']
TARGET_LIPID = 'DPG3'
```

```bash
# 最短コマンド
python run_claire.py --lipids CHOL DIPC DPSM --target-lipid DPG3 --parallel
```

## 出力の改善

### 以前
```
出力ディレクトリ/
├── composition_changes.png
├── temporal_composition.png
├── radial_profiles.png
└── ...
```

### 現在
```
出力ディレクトリ/
├── composition_changes.png  ← プレゼン用
├── composition_changes.svg  ← 論文用（NEW）
├── temporal_composition.png
├── temporal_composition.svg  （NEW）
├── radial_profiles.png
├── radial_profiles.svg  （NEW）
└── ...
```

## テスト方法

### 1. 最小限のテスト
```bash
# config.pyを編集
TOPOLOGY_FILE = "/path/to/test.psf"
TRAJECTORY_FILE = "/path/to/test.xtc"

# 実行
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --output test_output \
    --start 0 --stop 10 --step 1
```

### 2. 確認項目
- [ ] プログラムが正常に起動
- [ ] トラジェクトリが読み込まれる
- [ ] 各ステップが実行される
- [ ] 出力ディレクトリに.pngと.svg両方が生成される
- [ ] CSVファイルが生成される

## まとめ

### 主な改善点
1. ✅ **config.pyでのファイルパス設定** - コマンドが短く、管理が簡単
2. ✅ **config.pyでのフレーム範囲設定** - start/stop/stepをデフォルト化
3. ✅ **PNG/SVG両方での保存** - プレゼンと論文の両方に対応
4. ✅ **後方互換性の維持** - 既存のスクリプトも動作
5. ✅ **使いやすさの向上** - より少ないタイピングで実行可能

### 次のステップ
1. `config.py`でファイルパスを設定
2. テストランで動作確認
3. 本番データで解析実行
4. SVGファイルをIllustratorなどで論文用に編集
