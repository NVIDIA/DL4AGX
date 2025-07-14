# `perv_can_bus` handling

## Context and Problem Statement

- `prev_can_bus`を誰が保持すべきか
　- `prev_can_bus`は，`VadInputData`(正確にはその中のcan_busとshift)を計算する際に必要である
  - これを，`VadNode`, `VadInterface`, `VadModel`のどこで持つべきか？
- 参考: `prev_bev`は`VadModel`が保持している

## Considered Options

### 1. `VadModel`で持つ

- こうすると，`VadInterface`の責任範囲である，「`VadInputTopicData`を`VadInputData`に変換する」が`VadModel`にも散らばってしまう．
- よってこれは微妙

### 2. `VadInterface`で持つ

- すると，`VadInterface`が状態を持つことになる．
- 「`VadInputTopicData`を`VadInputData`に変換する」という処理は複雑であり，ROSとVADのTensorRT実装の両方に依存度が高く，debug頻度が高い．
- 状態を持つ変換処理はdebug難易度が上がる．よってなるべく避けたい．

### 3. `VadNode`で持つ

- Nodeが状態を持つのはよくある話であり，素直
- ただし，"can_bus"という，VADの世界の用語がNodeまで浸透するのはいただけない．

## Decision Outcome

- 「3. `VadNode`で持つ」を選択．
    - 理由: `VadInterface`の責任範囲を守ること，(ROSとVAD TensorRTの２つの世界にまたがっているために)もっともbugが出やすい`VadInterface`を混沌から守ることが最優先と考えた．そのために，`VadNode`に，"can_bus"という，VADの世界の用語が浸透することを許容する．

### Consequences

- 2025/07/14: 「3. `VadNode`で持つ」を選択．
