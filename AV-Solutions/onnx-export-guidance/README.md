# TorchScript-based ONNX Export Guidance for TensorRT

## Table of contents
- [ONNX export from Pytorch](#onnx-export-from-pytorch)
- [Python Build-in Functions and Dtypes](#python-build-in-functions-and-dtypes)
- [If-Else Control Flow](#if-else-control-flow)
- [For Loop](#for-loop)
- [Bitwise Ops](#bitwise-ops)
- [Bool Indexing](#bool-indexing)
- [None Tensors](#none-tensors)
- [Empty Tensors](#empty-tensors)
- [Dynamic Shape Tensors](#dynamic-shape-tensors)
- [Update Instance Variables (Attributes)](#update-instance-variables-attributes)
- [Shape Tensors and Execution Tensors](#shape-tensors-and-execution-tensors)
- [Unsupported Operators](#unsupported-operators)
- [Missing Operators](#missing-operators)
- [Constant Folding Failure](#constant-folding-failure)
- [Miscelleneous](#miscelleneous)
- [Helpful Links](#helpful-links)

### ONNX export from Pytorch
#### What is TorchScript-based ONNX Exporter?
In order to deploy model with TensorRT, the model needs to be exported from Pytorch to ONNX. The TorchScript-based ONNX exporter is available since PyTorch 1.2.0.
TorchScript-based ONNX exporter is leveraged to trace (through torch.jit.trace()) the model and capture a static computation graph. The exporter also supports TorchScript scripting (through torch.jit.script()), which adds support for data-dependent control-flow. 

- Pros: It exports static ONNX graphs, which are easy to postprocess (simplify, fold, ...) and use (by tools such as TensorRT to build fast inference engine). 
- Cons: It does not record any control-flow, like if-statements or loops; Does not truly handle dynamic inputs; Does not handle nuances between training and eval mode.

#### Basic Usage Example
```
import torch

# Define model
model = MyModel()

# Use TorchScript-based ONNX exporter to export the model to ONNX
torch.onnx.export(
    model,
    input_tuple,
    "model.onnx",
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)
```
The guidance covers only limited basic use cases and several edge cases when exporting ONNX for TensorRT using TorchScript-based ONNX exporter. It does not serve as a documentation for TorchScript-based ONNX exporter, nor guaranteed to always work. All experiences are based `Pytorch-1.12.1` and `TensorRT-8.6.12+`. Users are recommended to focus on `Pytorch-1.12` or above version, and `TensorRT-8.6` or above version. Please note expected performance also varies across different platforms and hardwares.
For more basic usage details (such as assigning dynamic axes, inserting custom ops, ...), please refer to [TorchScript-based ONNX Exporter](https://pytorch.org/docs/stable/onnx_torchscript.html).

### Python Build-in Functions and Dtypes
Python build-in functions (such as `min()`, `max()`, `len()`) and dtypes(such as `str`, `dict`) are usually not fully supported, it’s better to convert them to Pytorch operations/dtypes or move the usage of them out of ONNX graph creation.
#### Example1: convert to pytorch ops
`len(tensorA)` will not work for tensors with dynamic shape on the 0-th axes, consider replacing `len(tensorA)` with `tensorA.shape[0]` or `tensorA.size(0)`.
#### Example2: move out of ONNX graph
Comparison between strings: see the example and solutions [below](#special-case-1).

### If-Else Control Flow
#### Static if-else
When the choice of `if` path and `else` path do not depend on model inputs, just keep the `static if-else`, `torch.onnx.export` will pick one control flow path during the runtime.
#### Dynamic if-else
##### Preferred solution: vectorize if-else
When the choice of `if` path and `else` path depend on model inputs, vectorize the `dynamic if-else`.

Take the following dynamic if-else as an example
```
if condA:
  a=funcA(...)
elif condB:
  a=funcB(...)
else:
  a=funcC(...)
```

Convert `condA`, `condB` to tensor bools before the forward pass if they are python bools.
Then avoid if-else control flows by vectorizing them
```
condC = ~(condA|condB) 
a = condA*funcA(...) + condB*funcB(...) + condC*funcC(...)
```
##### Other solutions: 
Using `@torch.jit.script` to define the if-else part as a separate torchscript function is also acceptable, see [Tracing vs Scripting](https://pytorch.org/docs/stable/onnx_torchscript.html#tracing-vs-scripting), but control flow nodes are not preferred in ONNX by TensorRT.


##### Special Case 1: 
Sometimes it's not possible to convert python bools to tensors inside ONNX graph (e.g., the python bool is generated from two string comparison like `if stringA!=stringB`, or comparisons between tensor and None tensor such as `if tensor is None`), consider moving the comparisons outside ONNX and sending an indicator tensor (a bool or int tensor) as an auxiliary ONNX input. A good indicator tensor example is `use_prev_bev` in [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt/blob/303d3140c14016047c07f9db73312af364f0dd7c/tools/bevformer/onnx2trt.py#L183).

Say in the following case
```
# assuming prev_bev and bev_embed always have a shape of [bevh**2, 1, 256], where bevh is a static number
if prev_bev is None or current_scene_token != prev_scene_token:
  prev_bev = torch.zeros(bevh**2, 1, 256)
else:
  prev_bev = bev_embed
```
To convert a TensorRT-friendly ONNX, there are two steps
1. move the comparison outside the ONNX graph creation
```
if prev_bev is None or current_scene_token != prev_scene_token:
  use_prev_bev = 0
else:
  use_prev_bev = 1
```
and pass `use_prev_bev` into the ONNX graph as an auxiliary input.\
2. inside the ONNX graph creation
```
prev_bev = torch.zeros(bevh**2, 1, 256)*(1-use_prev_bev) + bev_embed*use_prev_bev
```

##### Special Case 2:
Sometimes we may need to deal with if-else with different output shapes:\
Say there is a if-else with different shape outputs at axes 0
```
if scene_token_changed:
  output = test_track_instances
else:
  output = empty_track_instances
```
Follow the steps to vectorize such if-else
1. pad the smaller shape tensor with special value
2. do multiplication and addition (to replace if-else as above example)
3. create index by detecting the special value
4. do querying(slicing)

Select a special value `-10**4` (or whatever values, just make sure normal values in `output` never reach the special value, and the value is in the allowed range of the dtype of `output`, such as `float`, `int`, `long`, depending on your use case), then do step 1-4
```
len_test_track_instances = test_track_instances.shape[0]
len_empty_track_instances = empty_track_instances.shape[0]
padded_empty_track_instances = torch.cat([empty_track_instances, \
                                        torch.ones(len_test_track_instances-len_empty_track_instances).to(empty_track_instances)*(-10**4)], \
                                        dim=0)
track_instances = padded_empty_track_instances*scene_token_changed + test_track_instances*(1-scene_token_changed)
index = track_instances!=(-10**4)
output = track_instances[index]
```

#### torch.where
It's recommended to replace `torch.where` with multiplication and addition to avoid potential `IffCondition` node which may cause TensorRT engine build failure.
Assuming `x` and `y` have the same shape
```
z = torch.where(condition.bool(), x, y)
```
replace  `torch.where` by
```
z = condition.float()*x + (1-condition.float())*y
```
`condition` can be either the same shape of `x`,`y` or a single item tensor with shape `[1]` , depending on your use case. 
`z` is the output, which should have the same shape of `x` and `y`


#### torch.Tensor.squeeze
Torch's squeeze operation inserts a if-else condition node into ONNX, since in ONNX the squeeze operation first checks if the desired dimension is 1 and only then will it apply the squeeze operation. This will lead to a TensorRT error: `IIfConditionalOutputLayer inputs must have the same shape`. To walk around, it is required to avoid using `.squeeze(x)` or `.unsqueeze(x)`. Replace all such ops with explicit indexing
```
x.squeeze(0) => x[0]
x.unsqueeze(0) => x[None, ...]
x.unsqueeze(2) => x[:, :, None, ...]
x.unsqueeze(-1) => x[..., None]
```
or reshaping if exact tensor shapes are known. For exmaple,
```
# y.shape -> [4,32,1,64,32]
x = y.squeeze(2)
```
Replace `torch.Tensor.squeeze` with indexing
```
x = y[:,:,0,:,:]
```
Or replace `torch.Tensor.squeeze` with reshaping
```
b,c,_,h,w = y.shape
x = y.reshape(b,c,h,w)
```

### For Loop
#### Parallelizable For Loops
For loops (whether dynamic or static number of loops) can be avoided, if future loops do not depend on the results of previous loops. 
##### Example: Parallelizable For Loops
before any computation in this funcA: 
`cond1` is a shape `[z]` bool tensor, where `z` is a positive dynamic value
`track_instances` is a shape `[z]` int tensor, where `z` is a positive dynamic value
`max_obj_id` is a shape `[1]` int tensor, with value `>=0`  in it.
```
def funcA(...):
  cond1 = cond1.long()
  for i in range(track_instances.shape[0]):
      track_instances[i] = (max_obj_id[0].long())*cond1[i] + track_instances[i]*(1-cond1[i])
      max_obj_id = (max_obj_id+1)*cond1[i] + max_obj_id*(1-cond1[i])
  return max_obj_id, track_instances
```
The for loop can be avoided because future steps computation does not depend on past steps computation results. After parallelizing,
```
def funcB(...):
  track_instances[cond1] = torch.arange(cond1.nonzero().shape[0], device=max_obj_id.device, dtype=torch.int32)+max_obj_id[0].int()
  max_obj_id = max_obj_id + cond1.long().sum()
  return max_obj_id, track_instances
```

#### Not Parallelizable For Loops
##### Static Numbers of Loops
Keep it as is. `torch.onnx.export` will unfold it.
##### Dynamic Numbers of Loops
Use `@torch.jit.script`, see [Tracing vs Scripting](https://pytorch.org/docs/stable/onnx_torchscript.html#tracing-vs-scripting)

For example, in Pytorch, assuming there are dynamic numbers of loops depending directly or indirectly on model's input tensors
```
def nonmergable_func(x):
    # non-mergable operations
    ...
    return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_loops):
        # Dynamic loop based on input
        for _ in range(num_loops):
            x = nonmergable_func(x)
        return x
```
and assuming that mathematically we cannot merge multiple runs of `nonmergable_func` into a single run, i.e., we must run `nonmergable_func` `num_loops` times one by one to get the final result. To make it exportable with correct logic, define the for loop separately as a torchscript function
```
# TorchScript function for the dynamic loop
@torch.jit.script
def dynamic_loop(x, num_loops):
    for _ in range(num_loops):
        x = nonmergable_func(x)
    return x

# Updated model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_loops):
        # Use the scripted function for the loop
        x = dynamic_loop(x, num_loops)
        return x
```


### Bitwise Ops
Usually bitwise such as `& | ~` are supported in tensor bools(not python bools) by TensorRT, but sometimes may not. However, bitwise ops can be converted to equavilent basic ops like `+-*` with dytpe cast, and it is always ok to use basic ops.\
For example \
`c = a&b` ⇔ `c = (a.int() * b.int()).bool()` \
`c = a|b` ⇔ `c = (a.int() + b.int()).bool()` \
`~c` ⇔ `(1 - c.int()).bool()` \
where `a,b,c` are bool tensors.

### Bool Indexing
Usually using bool tensors as indices to query tensors with valid shapes is supported by TensorRT, but sometimes may not, and in this case, try the following function to convert bool tesnor index to long tensor index. [In ONNX, bool tensor index query is translated to `Where` nodes, Long tensor index query is translated to `ScatterND` nodes]
```
# WARNING: this function is just designed for index tensors with shape length 1, i.e., say bool_index.shape is [z], where z>=1
# for other shape length index tensors, please generalize
def index_bool2long_trt(bool_index):
    long_index = bool_index.long()
    long_index = torch.arange(1, long_index.shape[0]+1, device=long_index.device) * long_index
    long_index = long_index[long_index.nonzero(as_tuple=True)[0]]
    long_index = long_index - torch.ones_like(long_index)
    return long_index

# bool query: assuming t.shape[0]==z
bool_queried_t = t[bool_index]  # this sometimes throws a bool indexing related error by TensorRT, but not always. 

# if TensorRT gives error, WAR: convert bool index to long index
long_quried_t = t[index_bool2long_trt(bool_index)]
```

### None Tensors
None tensors should be avoided. Such as
```
# assuming x, y are torch tensors with static shape [b, c, h, w]
if x is not None:
  y = y + x
```
Example WARs are:\
WAR1: move this comparison outside ONNX (recommended) \
WAR2: if `x` has static shape, initialize `x` as a special value tensor, and compare `x` with this special value tensor to allow if-else vectorization.
```
# initialize x as a special value tensor, please make sure normal values in x never reach this special_value and special_value is in the allowed range of the dtype
special_value = -10**4
x = torch.ones([b, c, h, w]) * special_value

...

# later when compare, vectorize the if-else as usual
cond = (x!=torch.ones([b, c, h, w])*(-10**4)).int()
y = (y + x)*cond + y*(1-cond)
```

### Empty Tensors
Sometimes, after querying with a fully False bool index, an empty tensor will be generated. TensorRT supports ops on empty tensor. 
#### Assign a value to an empty tensor
If `index` is a fully False bool tensor then `x1[index]` can be an empty tensor. \
When assigning a value `assigned_value` to `x1[index]` whether `x1[index]` is empty or not, it's allowed in pytorch to do
```
x1[index] = assigned_value
```
however, when exporting a TensorRT-friendly ONNX, creating a tensor of ones that has the same shape as the empty tensor, and multiplying it with the `assigned_value` is required (i.e., leading TensorRT to create an `IFillLayer`)
```
x1[index] = torch.ones_like(x1[index])*assigned_value
```

#### Slice on empty tensor
Directly slicing on non-empty axes of empty tensors is allowed. For example
```
x2 = torch.ones(0, 100)
x3 = x2[..., 1:9]
```

#### Support empty tensor reshape in TensorRT
ONNX reshape placeholder `allowzero` `0->1` using [`onnx_graphsurgeon`](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html) is useful for empty tensor reshaping in TensorRT, there is no harm to repair the ONNX after `torch.onnx.export` by
```
import onnx_graphsurgeon as gs
...

torch.onnx.export(onnx_file_name, ...)
graph = gs.import_onnx(onnx.load(onnx_file_name))
for node in graph.nodes:
    if node.op == "Reshape":
        node.attrs["allowzero"] = 1
onnx.save(gs.export_onnx(graph), onnx_file_name[:-4]+'repaired.onnx')
```

### Dynamic Shape Tensors
TensorRT does not always support dynamic shape tensors value assignment. \
For example, if `_mask` is a dynamic shape long tensor index or a dynamic value bool tensor index, then `A[_mask]` and `B[_mask]` are both tensors with dynamic shape at axes 0, in pytorch it is allowed to assign a dynamic shape tensor's values to another dynamic shape tensor
```
A[_mask] = B[_mask]
```
to export a TensorRT-friendly ONNX, it is recommended to avoid assigning a dynamic shape tensor's values to another dynamic shape tensor by cherry-picking values. (if it's possible to do so)
```
# assuming _mask is a dynamic value bool tensor
A = A*(~_mask) + B*(_mask)
```

### Update Instance Variables (Attributes)
In a forward function inside a class, all updated `self.v` should be passed out and passed in in the next frame. \
For example, avoid the following
```
Class ...:
  def forward(self, ...):
      ...
      self.v += 1
      ...
      return ...
```
Instead,
In the first frame, initialize `v` outside ONNX, and pass it in the forward function
```
Class ...:
	def forward(self, ..., v):
    ...
    v = v + 1
    ...
    return ..., v
```
Store the returned `v` in cache (update state, outside ONNX) \
In the next frame,
Pass in the updated `v` as the next input.

### Shape Tensors and Execution Tensors
In TensorRT, there is distinction between Shape Tensors (on host), and Execution Tensors (on device). Do not miss use them. \
Below are some common PyTorch operations and their typical ONNX equivalents for shape tensor manipulation when exported:
- Tensor shape retrieval: `x.size()` or `x.shape` in PyTorch → `Shape`
- Indexing a dimension: `x.size(dim)` → `Shape` + `Gather`
- Number of elements: `torch.numel(x)` → `Size` in ONNX
- Reshaping and dimension ops: `x.reshape(shape)` or `x.view(shape)` → `Reshape`
- Combining shapes: `torch.cat([shape_tensors...])` → `Concat`
- Filtering indices and dynamic shapes: `torch.nonzero(x)` → `NonZero`
- Expanding shapes: `x.expand(new_shape)` → `Expand`
- Creating tensors from shapes: 
  - `torch.zeros(shape)` or `torch.ones(shape)` when shape is derived from `x.size()` → `ConstantOfShape` (if used in a way that ONNX can infer the shape statically)
  - `torch.arange(start, limit)`  → `Range`
  - ...
- Slicing shape-related tensors: `x[:, ...]` or slicing on shape tensors → `Slice`

Sometimes it is allowed to use execution tensors as shape tensors in Pytorch, for example,
```
# assuming cond is a bool tensor with shape [z], track_instances is a int tensor with shape [z]
# both of them are execution tensors on CUDA device
track_instances[cond] = torch.arange(cond.sum().long(), device=track_instances.device, dtype=torch.int32)
```
However in TensorRT, Range Op accepts shape tensors on host (i.e., CPU) as input, thus to export a TensorRT-friendly ONNX, 
```
# considering feeding a shape tensor that has the same value of cond.sum() into torch.arange
track_instances[cond] = torch.arange(cond.nonzero().shape[0], device=track_instances.device, dtype=torch.int32)
```
Moreover, in rare cases if after running `trtexec` TensorRT shows not supporting assigning a dynamic shape tensor to another, leverage the experience in [Dynamic Shape Tensors Value Assignment](#dynamic-shape-tensors-value-assignment), thus have
```
# avoid assigning a dynamic shape tensor's values to another dynamic shape tensor by cherry-picking values
track_instances = ~cond * track_instances + cond * (cond.int().cumsum(0, dtype=torch.int32)-1)
```

### Unsupported Operators
#### Quick Solution: Re-write with TensorRT-supported Pytorch Operators
Example 1: `torch.unique(tensor)`
```
def torch_unique_trt(tensor):
    sorted_tensor, _ = torch.sort(tensor)
    diff = torch.ones_like(sorted_tensor)
    diff[1:] = sorted_tensor[1:] - sorted_tensor[:-1]
    unique_elements = sorted_tensor[diff != 0]
    return unique_elements
```
Example 2: `torch.atan2(y, x)`
```
import math
def torch_atan2_trt(y, x):
    '''
    reference: https://en.wikipedia.org/wiki/Atan2
    '''
    eps = 1e-8
    atan = torch.atan(y/(x+eps))
    x_eq_0 = x==0
    x_gt_0 = x>0
    x_ls_0 = x<0
    y_ge_0 = y>=0
    y_gt_0 = y>0
    y_ls_0 = y<0

    pi_div_2 = (torch.ones_like(atan))*(math.pi/2)
    negative_pi_div_2 = (torch.ones_like(atan))*(-math.pi/2)

    atan2 = (negative_pi_div_2)*(x_eq_0 & y_ls_0).int()\
            + (pi_div_2)*(x_eq_0 & y_gt_0).int()\
            + (atan-math.pi)*(x_ls_0 & y_ls_0).int()\
            + (atan+math.pi)*(x_ls_0 & y_ge_0).int()\
            + (atan) * x_gt_0.int()

    return atan2.float()
```

#### Standard Solution: ONNX Custom Operators and TensorRT Plugins
For Pytorch Operators that cannot be re-written with both [ONNX Supported TorchScript Operators](https://pytorch.org/docs/stable/onnx_torchscript_supported_aten_ops.html) and [TensorRT-supported Operators](https://github.com/onnx/onnx-tensorrt?tab=readme-ov-file#supported-operators), export ONNX with [Custom Operators](https://pytorch.org/docs/stable/onnx_torchscript.html#custom-operators), and write [TensorRT Plugins](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html) for it.


### Missing Operators
#### Missing `NonZero` operation in TensorRT
A common pattern in object detection networks is to threshold detections below a specified value to reduce downstream computations.  This will cause a TensorRT engine build to failure due to a missing `NonZero` operation.  The simplest solution for this is to replace thresholding with `topK`.
```
mask = confidence > threshold
detections = predictions[mask]
```
If the returned `detections` element order does not matter, replace the above code with:
```
mask = confidence > threshold
indices = torch.topk(input=confidence, k=mask.nonzero().shape[0])
detections = predictions[indices]
```
In other circumstances that `indices`/`detections` element order matters, an alternative and generalized solution would be converting bool indices to long indices, see [Bool Indexing](#bool-indexing).

#### Missing `resolve_conj` in ONNX opset
Pytorch can insert `resolv_conj` into the ONNX graph even when not working with conjugate values; in old versions of Pytorch this operation is missing from the ONNX export but in newer versions it is implemented is a [no op](https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset9.py#L6588-L6596).  For older versions of pytorch the solution is to backport the no op.
```
def noop_complex_operators(g, input):
    return input
torch.onnx.register_custom_op_symbolic("aten::resolve_conj", noop_complex_operators, 9)
torch.onnx.register_custom_op_symbolic("aten::resolve_neg", noop_complex_operators, 9)
```

### Constant Folding Failure
A common warning message received during ONNX export is as follows:
```
Warning: Constant folding in symbolic shape inference fails: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper__index_select)
```
This warning message can cause shape inference to fail for downstream parts of the network.  This error message is caused by a Tensor residing on the GPU while the values used for indexing residing on the CPU. IE:
```
pc_range.device -> cuda
points.device -> cuda
points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
```
In the above example, the slice ranges reside on the CPU (`3:6`, `0:3`) and the CPU -> GPU migration is not correctly handled in ONNX export. 
#### Solution 1: manual unroll
The range based indexing of this operation require using torch's indexing operations, but single element indexing does not; thus the above can be converted to.
```
points[:,0] = points[:,0] * (pc_range[3] - pc_range[0]) + pc_range[0]
points[:,1] = points[:,1] * (pc_range[4] - pc_range[1]) + pc_range[1]
points[:,2] = points[:,2] * (pc_range[5] - pc_range[2]) + pc_range[2]
```
#### Solution 2: reformulating broadcast operations
Broadcasting operations can be implemented as indexing operations which do not correctly migrate the indexing tensor to the GPU during onnx export.
```
update = (1.0 - x).view(B, 1, 1) * pseudo_reference_points
```
In the above code block `(1.0-x).view(B, 1, 1)` produces a [B,1,1] shaped tensor to be multiplied by a [B, 1024, 3] tensor.  The broadcasting of this multiplication produces an indexing array.
Broadcasting can be reformulated a few ways.
```
# Since batch size is 1, we collapse the 1d tensor to a 0d tensor and thus this becomes scalar multiplication.
update = (1 - x[0]) * pseudo_reference_points

# another option is to repeat the broadcasted element and multiply by a flattened array.  
N = pseudo_reference_points.numel()
update = (1 - x).repeat(N) * pseudo_reference_points.flatten()
```

#### Solution 3: manual index generation on the device
In the below code example, self.num_propagated is a constant uint64_t value. 
```
# memory_reference_point.shape -> [B, N, 3] N > self.num_propagated (1280 in the reference code)
# update.shape -> [B, self.num_propagated, 3] (For reference, this tensor comes from the previous example)
memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + update
```
In the above code example `memory_reference_point[:, :self.num_propagated]` produces an index_select and index_set operation with a CPU side indexing tensor.
The code can be rewritten as follows:
```
B,N, _ = memory_reference_point.shape # [B, N, 3]
# this creates indices on the device
indices= (torch.arange(B, device=device, dtype=torch.long)[:,None,None], 
          torch.arange(N, device=device, dtype=torch.long)[None,:,None], 
          torch.arange(3, device=device, dtype=torch.long)[None,None,:])
# this uses index_put_ with our device side indices and accumulate to update the values of memory_reference_point
memory_reference_point.index_put_(indices, update, accumulate=True)
```
Note that if `self.num_propagated` is a dynamic number on CPU, the above solution still works, see "Slice with dynamic dim" in [Miscelleneous](#miscelleneous). If `self.num_propagated` is an execution number on GPU, it's not recommended in TensorRT to use it as a shape tensor due to the [formal inference rules](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#formal_inference_rules), see also [Shape Tensors and Execution Tensors](#shape-tensors-and-execution-tensors).

#### Solution 4: ignore and perform constant folding after ONNX export.
It is possible to export an ONNX graph with this error present, and then perform constant folding on the ONNX model since once the ONNX model is reloaded the model will fully be on the CPU.  This is viable if shape inference is not needed for anything downstream.  Additionally constant folding and shape inference is done during TensorRT engine build so it may not be necessary during ONNX.


### Miscelleneous
1. Avoid using `torch.Tensor.item()`, otherwise the tensor will be treated as a constant

2. Slice with dynamic dim \
`tensorA[0:dim]` is not supported by TensorRT if `dim` is explicitly dynamically generated. \
Usually `dim` is static and can be pre-calculated & set as static. \
If `dim` is a truly dynamic value, consider converting it to a long or bool index and then query on `tensorA`. \
For example, replace
    ```
    tensorB = tensorA[0:dim]
    ```
    by
    ```
    index = torch.arange(dim)
    tensorB = tensorA[index]
    ```

3. List copy:
Deep copy is not allowed in `torch.onnx.export`. 
    ```
    listB = copy.deepcopy(listA)
    ```
    =>
    ```
    listB = []
    for item in listA:
      listB.append(item)
    ```

4. Issues with `torch.zeros_like` in ONNX export\
say `t` is a static shape `[x]` input, where `x>=1` \
option1: `torch.zeros_like(t[0])` gives ONNX graph: `t->gather(0)->shape->ConstantOfShape->out` \
option2: `torch.zeros_like(t)[0]` gives ONNX graph: `a full zero constant with shape () -> out` \
It's a special case in `torch.onnx.export` because `t[0]` has shape of `()`, which is treated as dynamic shape.

5. Avoid using List comprehensions. Instead, use standard for loop.\
    ```
    l1 = [i for i in range(10)]
    ```
    =>
    ```
    l1=[]
    for i in range(10):
      l1.append(i)
    ```

6. Avoid using `Flatten`, `Unflatten`, instead, use combinations of `reshape`(preferred over `view`) and `permute` and/or `einsum`.

7. It's highly recommended to export ONNX with real inputs in `torch.onnx.export`. 
- For models with dynamic shape inputs/outputs, or models that may generate empty intermediate/output tensors, it's recommended to export ONNX with real input samples that will not generate empty tensors. Usually the first several data samples from dataset will likely generate empty tensors.

8. Read [Avoiding Pitfalls](https://pytorch.org/docs/stable/onnx_torchscript.html#avoiding-pitfalls), [Limitations](https://pytorch.org/docs/stable/onnx_torchscript.html#limitations), [Support Custom Ops](https://pytorch.org/docs/stable/onnx_torchscript.html#adding-support-for-operators), and [FAQ](https://pytorch.org/docs/stable/onnx_torchscript.html#frequently-asked-questions) of TorchScript-based ONNX Exporter carefully.

### Helpful Links
[ONNX Supported TorchScript Operators](https://pytorch.org/docs/stable/onnx_torchscript_supported_aten_ops.html) \
[TensorRT Backend For ONNX](https://github.com/onnx/onnx-tensorrt) \
[ONNX Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md) \
[Pytorch API Categorization](https://gist.github.com/soumith/9851dc7327919fc42c18066a0a7c412d) \
[Symbolic Functions Opsets](https://github.com/pytorch/pytorch/tree/main/torch/onnx) \
[TorchScript-based ONNX Exporter](https://pytorch.org/docs/stable/onnx_torchscript.html) \
[`torch.onnx` Documentation](https://pytorch.org/docs/stable/onnx.html)