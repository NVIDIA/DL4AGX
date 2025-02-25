# can_bus.binを読み込んで，値を確認する．
import numpy as np
import yaml

can_bus_data = {}
for frame_id in range(1, 30):
    can_bus_path = f"/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data/{frame_id}/img_metas.0[can_bus].bin"

    # バイナリファイルをバイト単位で読み込む
    with open(can_bus_path, "rb") as f:
        data = f.read()

    # バイナリデータをnumpy配列に変換
    can_bus_data[frame_id] = np.frombuffer(data, dtype=np.float32)

# can_busのdataをyamlに吐き出す
# can_bus[0:3]はtranslation, [3:7]はrotation, 
# can_bus[7]はax, [8]はay
# can_bus[12]はego_w(各速度)
# can_bus[13]はego_vx, [14]はego_vy
# can_bus[-2]はpatch_angle[rad], can_bus[-1]はpatch_angle[deg]
with open(f"can_bus.yaml", "w") as f:
    for frame_id, data in can_bus_data.items():
        f.write(f"{frame_id}:\n")
        for i in range(len(data)):
            if i < 3:
                f.write(f"  - {data[i]} # translation \n")
            elif 3 <= i < 7:
                f.write(f"  - {data[i]} # rotation \n")
            elif i == 7:
                f.write(f"  - {data[i]} # ax \n")
            elif i == 8:
                f.write(f"  - {data[i]} # ay \n")
            elif i == 12:
                f.write(f"  - {data[i]} # ego_w(yaw方向角速度) \n")
            elif i == 13:
                f.write(f"  - {data[i]} # ego_vx \n")
            elif i == 14:
                f.write(f"  - {data[i]} # ego_vy \n")
            elif i == 16:
                f.write(f"  - {data[i]} # patch_angle[rad] \n")
            elif i == 17:
                f.write(f"  - {data[i]} # patch_angle[deg] \n")
            else:
                f.write(f"  - {data[i]} \n")
