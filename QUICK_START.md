# CLAIRE クイックスタート

## エラーが出た場合の対処法

### エラー: "No topology file specified"

**原因**: `config.py`で`TOPOLOGY_FILE`と`TRAJECTORY_FILE`が設定されていません

**解決方法1**: config.pyを編集

```bash
# config.pyを編集
nano config.py
```

または

```bash
vi config.py
```

以下のように変更：

```python
# コメントアウトを外す
TOPOLOGY_FILE = "/Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step5_assembly.psf"
TRAJECTORY_FILE = "/Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step7_production.xtc"
```

**解決方法2**: コマンドラインで直接指定

```bash
python run_claire.py \
    --topology /path/to/system.psf \
    --trajectory /path/to/trajectory.xtc \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --parallel
```

### エラー: "zsh: command not found: --parallel"

**原因**: コマンドを改行する際にバックスラッシュ（`\`）が必要です

**間違った例**:
```bash
python run_claire.py --lipids CHOL DIPC DPSM --target-lipid DPG3
  --parallel
```

**正しい例**:
```bash
# 1行で書く
python run_claire.py --lipids CHOL DIPC DPSM --target-lipid DPG3 --parallel
```

または

```bash
# バックスラッシュで改行
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --parallel
```

## 実行手順

### ステップ1: config.pyを編集

```bash
cd /Users/takeshi/Library/CloudStorage/OneDrive-学校法人京都薬科大学/manuscript/JOSS/conservedxx5done/claire_new

# config.pyを編集（お好みのエディタで）
nano config.py
```

以下の行を編集：

```python
# コメントアウト（#）を外して有効化
TOPOLOGY_FILE = "/Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step5_assembly.psf"
TRAJECTORY_FILE = "/Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step7_production.xtc"
```

保存して終了（nanoの場合: Ctrl+X, Y, Enter）

### ステップ2: テスト実行

```bash
# まずは少数フレームでテスト（config.pyより優先される）
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --start 0 \
    --stop 100 \
    --step 10 \
    --output test_output
```

### ステップ3: 本番実行

config.pyで設定したフレーム範囲で実行：

```bash
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output epha2_results \
    --parallel
```

## よくある問題と解決法

### 問題1: ファイルが見つからない

```
ERROR: Topology file not found: /path/to/system.psf
```

**解決**: ファイルパスを確認

```bash
# ファイルが存在するか確認
ls -la "/Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step5_assembly.psf"
```

### 問題2: メモリ不足

```bash
# フレーム数を減らす
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --start 20000 \
    --stop 30000 \
    --step 20
```

### 問題3: 並列処理が動かない

```bash
# 並列処理なしで実行
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output results
```

## 最小限のテストコマンド

config.pyを編集したくない場合：

```bash
python run_claire.py \
    --topology /Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step5_assembly.psf \
    --trajectory /Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step7_production.xtc \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --start 0 \
    --stop 10 \
    --step 1 \
    --output quick_test
```

## 出力の確認

```bash
# 出力ディレクトリを確認
ls -la epha2_results/

# 生成されたファイル
# - composition_data.csv
# - composition_changes.png
# - composition_changes.svg
# - temporal_composition.png
# - temporal_composition.svg
# など
```

## 次のステップ

1. テスト実行で動作確認
2. config.pyで本番設定
3. 本番データで実行
4. SVGファイルで論文用の図を作成
