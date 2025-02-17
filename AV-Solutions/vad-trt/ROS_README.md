# TODO

- [ ] trajectory, bboxの座標変換
- [ ] can_busを追加(/tf, /sensing/imu/tamagawa/imu_raw, /localization/kinematic_stateから取得)
- [ ] /tf_staticからlidar2imgを作る
- [ ] pluginsのbuildとnodeのbuildを同時に実行できるようにCMakeLists.txtを変更
- [ ] /planning/mission_planning/routeの出力を使って，ego_hist_trajをダミーで作り，1フレーム目のego_hist_trajとして入力する
- [ ] rangeを30m以遠にする
- [ ] 現状(prev_bevをGPU->CPUに移し，CPU-> GPUに移している)の実装を修正し，「prev_bevをGPUに置いたままにする」実装に変える

## can_bus

- client
  - https://github.com/Shin-kyoto/VAD/blob/859ef575b21b45d5761a36b5d0381b5347bec5d3/vad_client.py#L561-L573
- can_bus
  - https://github.com/Shin-kyoto/VAD/blob/859ef575b21b45d5761a36b5d0381b5347bec5d3/vad_server.py#L411-L446

```python
# ego2global_translation(/tfから取得)
can_bus[0:3] = nsego2global_translation

# ego2global_rotation(/tfから取得)
can_bus[3:7] = [
    nsego2global_rotation[0],  # roll
    nsego2global_rotation[1],  # pitch
    nsego2global_rotation[2],  # yaw
    0.0
]

# 加速度(IMUから取得)
can_bus[7] = ax
can_bus[8] = ay

# yaw角速度(/localization/kinematic_stateから取得)
can_bus[12] = ego_w

# 速度
can_bus[13] = ego_vx
can_bus[14] = ego_vy

# patch_angle（座標系の変換を考慮して90度加算）(/tfから取得)
patch_angle = request.can_bus.patch_angle + 90.0
can_bus[-2] = patch_angle / 180 * np.pi
can_bus[-1] = patch_angle
```
