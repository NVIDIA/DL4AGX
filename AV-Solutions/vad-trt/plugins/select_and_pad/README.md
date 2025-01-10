### Pseudo code for select_and_pad plugin
```python
import torch

def select_and_pad(feat, flag, invalid, P):
    B, Q, C = feat.shape
    B, Q = flag.shape
    C = invalid.shape
    out = zeros(B, P, C)
    for i in range(B):
        temp = feat[i, flag]  # pick values from feat according to given flag
        temp_len = len(temp)  # decide number of padding elements
        out[i, :temp_len] = temp  # move valid items from feature to out
        out[i, temp_len:] = invalid # pad the remaining items with 'invalid' features
    return out
```