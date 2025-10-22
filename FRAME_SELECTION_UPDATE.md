# フレーム選択機能の追加

## 実装した変更

### config.pyに追加した設定

```python
# ===== Frame Selection =====
# Set default frame range for analysis
FRAME_START = 0          # Start frame (default: 0)
FRAME_STOP = None        # Stop frame (default: None = all frames)
FRAME_STEP = 1           # Frame step (default: 1 = every frame)
```

### 使い方

#### 1. config.pyで設定する（推奨）

```python
# config.py
TOPOLOGY_FILE = "/path/to/system.psf"
TRAJECTORY_FILE = "/path/to/trajectory.xtc"

# 例：フレーム20000-50000を10フレームごとに解析
FRAME_START = 20000
FRAME_STOP = 50000
FRAME_STEP = 10
```

実行：
```bash
# config.pyの設定を使う
python run_claire.py --lipids CHOL DIPC DPSM --target-lipid DPG3 --parallel
```

#### 2. コマンドラインで上書き

```bash
# config.pyの設定を上書きする
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --start 0 \
    --stop 1000 \
    --step 10 \
    --parallel
```

### メリット

1. **一度設定すれば繰り返し使える**
   - 複数回実行するときに毎回フレーム範囲を指定する必要がない

2. **プロジェクトごとに管理しやすい**
   - EphA2用のconfig.py、Notch用のconfig.pyなど

3. **柔軟性も維持**
   - 必要に応じてコマンドラインで上書き可能

### 典型的な使用例

#### ケース1: 平衡化後のデータのみ解析
```python
# config.py
FRAME_START = 20000  # 平衡化完了
FRAME_STOP = None    # 最後まで
FRAME_STEP = 10      # 10フレームごと
```

#### ケース2: 特定期間の詳細解析
```python
# config.py
FRAME_START = 30000
FRAME_STOP = 40000
FRAME_STEP = 1  # 全フレーム
```

#### ケース3: 高速テスト
```python
# config.py
FRAME_START = 0
FRAME_STOP = 1000
FRAME_STEP = 100  # 粗くサンプリング
```

### 実行時の出力例

```
================================================================================
CLAIRE - Composition-based Lipid Analysis
================================================================================
Topology: /path/to/system.psf
Trajectory: /path/to/trajectory.xtc
Output: results
Lipids: ['CHOL', 'DIPC', 'DPSM']
Target lipid: DPG3
Frame range: 20000 to 50000 (step 10)
Contact cutoff: 15.0 Å
================================================================================

### STEP 1: Loading Trajectory ###
✓ Trajectory loaded: 123456 atoms, 80000 frames

Analyzing 3000 frames: 20000 to 49990 (step 10)
```

## 変更されたファイル

1. **config.py**
   - `FRAME_START`, `FRAME_STOP`, `FRAME_STEP`を追加

2. **run_claire.py**
   - コマンドライン引数のデフォルト値を変更
   - config.pyからの読み込み処理を追加
   - 出力表示にフレーム範囲を追加

3. **README.md**
   - フレーム選択の説明を追加

4. **USAGE_GUIDE.md**
   - 日本語の詳細説明を追加

5. **CHANGES_SUMMARY.md**
   - 変更内容のまとめに追加

## 後方互換性

✅ 完全に保持されています

既存の使い方も全て動作します：

```bash
# これまで通り動作
python run_claire.py --topology system.psf --trajectory traj.xtc \
    --start 0 --stop 1000 --step 10 --lipids CHOL DIPC DPSM

# 新しい方法
python run_claire.py --lipids CHOL DIPC DPSM  # config.pyから読み込み
```

## テスト

### 推奨テスト手順

1. config.pyを編集
```python
TOPOLOGY_FILE = "/path/to/test.psf"
TRAJECTORY_FILE = "/path/to/test.xtc"
FRAME_START = 0
FRAME_STOP = 100
FRAME_STEP = 10
```

2. 実行
```bash
python run_claire.py --lipids CHOL DIPC DPSM --target-lipid DPG3 --output test
```

3. 確認
- [ ] 正しいフレーム範囲が表示される
- [ ] 指定した範囲のみ解析される
- [ ] 出力ファイルが生成される

### コマンドライン上書きのテスト

```bash
python run_claire.py --lipids CHOL DIPC DPSM --start 50 --stop 150 --step 5 --output test2
```

確認：
- [ ] config.pyの設定が上書きされる
- [ ] 出力に「Frame range: 50 to 150 (step 5)」と表示される
