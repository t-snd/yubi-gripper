# yubi-gripper

YUBI で収集したグリッパのエンコーダ角度を、ロボット側グリッパのモータ
**Dynamixel XM460-W350-R** に位置指令として流し込み、ちゃんと動くかを確かめるためのツールです。

入力データは次の2種類に対応しています。

| 入力 | 中身 | 既定で読む場所 |
|---|---|---|
| **mcap**（yubi-app / yubi-sw で収録） | `sensor_msgs/JointState`、約100 Hz | `/yubi/gripper/{left,right}/joint_state`（無ければ `/joint_states` から関節名で選択） |
| **LeRobot**（rebake で変換したもの） | 30 fps、`[left, right]` の2要素 | `observation.joint_states`（`--key action.joint_states` も指定可） |

データの決まりごと（[yubi-sw](https://github.com/airoa-org/yubi-sw) の `encoder_node.py`、
[rebake](https://github.com/airoa-org/rebake) の `config/robot_model/yubi.yaml` から確認済み）:

- 単位は rad。セットアップ GUI の *Capture MIN* で取った値を原点にしているので、**全閉 = 0 rad**、開くほど増えます。
  URDF 上の指関節の範囲は 0.00〜0.94 rad です。
- 関節名は `left_right_finger_joint` / `right_right_finger_joint`。

## 仕組み

```
encoder angle [rad] --(線形マップ・範囲外はクリップ)--> Goal Position [tick] --> XM460-W350-R
 encoder_closed_rad (0.0)  -> motor_closed_tick
 encoder_open_rad   (0.94) -> motor_open_tick
```

- 収録時刻どおりに（`--speed` で倍速も可）Goal Position を送り、毎ステップ Present Position / Present Current を読んで CSV に記録します。
- 終了時に追従誤差（RMS / 最大、tick と rad 換算）と最大電流を表示します。
- 既定は **Current-based Position モード（5）** で、Goal Current を上限にして把持力を抑えます（物を噛んでもモータが無理に押し続けない）。
- 開始位置にはゆっくり移動してから（`approach_profile_velocity`）、Enter を待って再生を始めます。Ctrl+C で中断すると必ずトルクを OFF にします。

## インストール

```bash
pip install -e .            # グラフ出力も使うなら: pip install -e '.[plot]'
cp config/gripper.example.yaml config/gripper.yaml
```

U2D2 等で PC とモータを接続し、`config/gripper.yaml` の `port` / `baudrate` / `id` を合わせてください
（XM460-W350-R の工場出荷値は ID 1、57600 bps）。シリアルポートの権限が必要です（`dialout` グループ等）。
Linux の FTDI は latency timer を 1 ms にすると周期が安定します:
`echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`

## 使い方

### 1. データの中身を確認

```bash
yubi-gripper info path/to/episode.mcap        # rosbag2 ディレクトリでも可
yubi-gripper info path/to/lerobot_yubi        # rebake の出力ディレクトリ
```

左右それぞれのサンプル数・周期・角度範囲が出ます。範囲が 0〜0.94 rad から大きく外れる場合は
`mapping.encoder_closed_rad` / `encoder_open_rad` を実データに合わせてください。

### 2. モータ側の全閉・全開位置をキャリブレーション

```bash
yubi-gripper calibrate --write config/gripper.yaml
```

トルク OFF になるので、ロボット側グリッパを手で **全閉 → Enter → 全開 → Enter**。
取得した `motor_closed_tick` / `motor_open_tick` が設定ファイルに保存されます。

単発で動作確認したい場合:

```bash
yubi-gripper goto --state open
yubi-gripper goto --state closed
yubi-gripper goto --rad 0.3 --hold
```

### 3. まずはモータを動かさずに確認（dry-run）

```bash
yubi-gripper replay episode.mcap --side left --dry-run --plot logs/dry.png
```

### 4. 再生

```bash
# mcap（左グリッパ）
yubi-gripper replay episode.mcap --side left --plot logs/left.png

# LeRobot（エピソード 3、右グリッパ、半速）
yubi-gripper replay lerobot_yubi --episode 3 --side right --speed 0.5

# 一部区間だけ、3回ループ
yubi-gripper replay episode.mcap --start 5 --end 15 --loops 3
```

主なオプション:

| オプション | 説明 |
|---|---|
| `--side left/right` | どちらのグリッパの角度を使うか |
| `--key` | LeRobot の列（`observation.joint_states` / `action.joint_states`） |
| `--topic` | mcap のトピックを明示 |
| `--speed` | 再生速度倍率 |
| `--rate HZ` | 一定周期に再サンプリング（ゼロ次ホールド）して送る |
| `--operating-mode 3/4/5` | 位置 / 拡張位置 / 電流制限付き位置 |
| `--goal-current` | モード5の電流上限（単位 2.69 mA、Current Limit でクリップ） |
| `--profile-velocity` | 再生中の Profile Velocity（0 = 制限なし） |
| `--no-feedback` | Present Position の読み出しを省略（送信周期を詰めたい時） |
| `--hold` | 終了後もトルクを保持し Enter で解放 |
| `--log` / `--plot` | CSV ログ / 指令 vs 実位置のグラフ |

ログ（既定 `logs/replay_*.csv`）の列: `t_cmd, t_actual, cmd_rad, cmd_tick, present_tick, present_rad, current`。
`present_rad` はモータ位置をエンコーダ角度に逆変換した値なので、`cmd_rad` と直接比べられます。

## 注意

- YUBI 側エンコーダとロボット側グリッパの機構（リンク比）が違うので、角度をそのまま送るのではなく、
  **全閉・全開の2点で線形に対応づけ**ています。非線形な機構ならキャリブレーション点を増やす拡張が必要です。
- 30 fps の LeRobot データは mcap（100 Hz）より粗いので、追従が段付きに見えることがあります。
- モード 3 のときは 0〜4095（1回転内）しか指令できません。キャリブレーション値がこの範囲外ならモード 4/5 を使ってください。
- Goal Current 等の単位・上限は ROBOTIS e-Manual で XM460-W350 の値を確認してください。

## テスト

```bash
pip install -e '.[dev]'
pytest
```

合成した mcap / LeRobot データと疑似モータで、読み込み・変換・再生ループを検証しています。
