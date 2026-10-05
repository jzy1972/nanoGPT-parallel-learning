# nanoGPT TP+DP 验证包

本包验证你现有的 TP 模型和进程组函数。TP、DP 大小都通过参数传入。
脚本不会改写你的 model.py、train.py，也不会自动提交或 push。

## 你本地的 CPU 能验证什么

| 项目 | CPU + Gloo | CUDA + NCCL |
|---|---|---|
| TP/DP 进程组成员和 rank 映射 | 可以 | 可以 |
| 完整参数加载、QKV 分片和复制参数 | 可以 | 可以 |
| 前向 logits、全局平均 loss | 可以 | 可以 |
| 梯度累积、DDP 平均后的所有参数梯度 | 可以 | 可以 |
| AdamW 更新及一、二阶动量 | 可以 | 可以 |
| TP 全局梯度裁剪（额外选项） | 可以 | 可以 |
| 非有限梯度标记的全局一致性（额外选项） | 可以 | 可以 |
| 实际 train.py 的 dropout、AMP、CUDA RNG 和恢复训练 | 需要 GPU | verify_train_resume.py |

CPU 数值脚本真的启动多个进程并执行 Gloo 集合通信。
它固定 dropout=0，在测试进程内替换 CUDA RNG 跟踪器为无操作上下文。
这不会修改模型文件；CPU 结果覆盖零 dropout 的数学计算和通信组合。
GPU 的随机性、混合精度、NCCL 及训练入口集成另行实测。
本包的数值脚本采用全精度，CUDA 模式本身也不表示 AMP 已验证。

## 文件和放置方法

把整个 nanoGPT_tp_verification 文件夹放入你的 nanoGPT 根目录：
与 model.py、train.py、configurator.py 同级。保持文件夹内部文件在一起。

| 文件 | 作用 |
|---|---|
| check_project.py | 不依赖 PyTorch 的语法、接口和常见错误检查 |
| verify_tp_dp.py | CPU/CUDA 数值对齐验证 |
| verify_train_resume.py | 实际 GPU 训练恢复测试；也可在 CPU 上比较已取得的 checkpoint |
| verification_common.py | 测试工具：函数提取、独立切分参照、共同失败检查 |
| model_dense.py | 原版 nanoGPT 完整模型的参考副本 |
| reference_LICENSE.txt | 原版模型的 MIT 许可证 |
| VALIDATION_STATUS.json | 生成阶段实际做过的检查及未运行的项目 |

本包直接从 parallel_utils.py 或 train.py 提取你已写好的顶层函数：
build_parallel_groups；额外测试会使用 mark_tp_parameters、
clip_grad_norm_tp、has_nonfinite_grad。
函数由 AST 单独编译，不会执行 train.py 顶层的训练入口。
因此，无须为了本包再移动函数或修改训练入口。

参考模型来自本次工作区中你此前提供的原版 nanoGPT 副本。
如要换成你自己的原版参考文件，可使用 --reference 指定路径。
完整模型与 TP 模型采用相同配置。测试切分参照通过索引独立计算，
不会调用你的 shard_dense_tensor 来生成期望结果。

## 第一步：静态检查

以下命令在 nanoGPT 根目录的终端运行：

```bat
python nanoGPT_tp_verification/check_project.py --project .
```

如有 ERROR，先修正对应文件。最近上传版本中需要核对：

- expected_state = self.state_dict()，要调用方法。
- 多行 assert 的提示字符串用括号包住，左括号紧接条件后的逗号。
- F.linear 使用小写 l。
- CopyToTP、SumFromTP 通过 .apply 调用。
- 保存、恢复模型分片都使用 ckpt_tp{tp_rank:03d}.pt。
- control[1] 的名称完整；取标记使用 bool(control[1].item())。
- 使用 dist.barrier()。
- 保存决定的广播和最后的屏障由全部相关 rank 执行。
- runtime_rank 文件由所有 rank 分别保存。

静态检查不能发现所有逻辑错误，也不能代替数值测试。

## 第二步：确认 CPU PyTorch 和 Gloo

```bat
python -c "import torch; import torch.distributed as d; print(torch.__version__); print('Gloo:', d.is_gloo_available())"
```

未安装 PyTorch 时，可在你用于 nanoGPT 的 Python 环境安装 CPU 版本：

```bat
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

安装入口：https://pytorch.org/get-started/locally/
如已有可用 PyTorch，可直接使用现有环境。脚本会检查分布式后端是否可用。

## 第三步：本地 CPU 数值验证

```bat
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cpu --tp-size 2 --dp-size 2 --report cpu_tp_dp_report.json
```

这个示例表示 TP=2、DP=2，启动 4 个 CPU 进程。
进程数是 TP×DP，与 GPU 数量无关。数字可替换为其他正整数。
测试模型的 n_head=2×TP，n_embd=16×TP，保证按完整 head 切分。

更新版默认在主进程和子进程导入 PyTorch/NumPy 之前，
把 OpenBLAS、OpenMP、MKL 等计算库的线程上限设为 1。
子进程同时执行 torch.set_num_threads(1)。可用 --threads 修改上限；
小模型的 CPU 多进程验证建议先保持 1。

### Windows 上出现 OpenBLAS 内存分配错误

例如：OpenBLAS error: Memory allocation still failed after 10 retries。
这段信息本身不能判断 TP 数学计算是否正确，也不能判断根因一定是物理内存不足。
先限制计算库线程，避免多个进程分别创建过多线程和工作缓冲区。
即使使用旧版脚本，也可在 Windows CMD 中逐行运行：

```bat
set "OPENBLAS_NUM_THREADS=1"
set "OPENBLAS_DEFAULT_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cpu --tp-size 2 --dp-size 2 --threads 1
```

这些 set 命令只影响当前 CMD 窗口和它随后启动的程序。
设置环境变量要先于启动 Python；仅在导入库后调用 torch.set_num_threads
无法控制 OpenBLAS 导入时已经发生的初始化。
仍报错时保存完整输出，继续区分库导入、可用内存和进程启动问题：

```bat
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cpu --tp-size 2 --dp-size 2 --threads 1 > tp_verify_log.txt 2>&1
```

同一 CMD 中也可单独检查 NumPy/PyTorch 导入：

```bat
python -c "import numpy as np; import torch; print('NumPy:', np.__version__); print('PyTorch:', torch.__version__)"
```

此处保留 TP=2、DP=2 的测试规模；调整线程不会减少 TP/DP 进程数。
线程变量的说明见 [OpenBLAS 官方文档](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)。

默认设置：
- 小型 2 层模型，序列长度 8，词表 64。
- double 精度、dropout=0。
- 每个 DP 副本每个 micro-batch 为 2 条序列。
- 每次更新累积 2 个 micro-batch，连续检查 2 次 AdamW 更新。
- 原版模型每次处理完整的全局 batch，即 DP×micro-batch×累积次数。
- TP+DP 模型按 DP 取不同数据、TP 内广播相同数据。
- loss 只沿 DP 组求平均；不再把 TP 副本重复计入 batch。
- 前向和反向比较使用相同的数学 SDPA 后端，便于检查基本计算。

成功时预期会看到各阶段的 PASS：
进程组映射、参数加载、前向/loss、梯度、AdamW 参数与动量更新。
只有脚本正常结束并输出最后的 PASS，才记为本次数值验证通过。
report.json 中记录验证范围和参数。

### 增加裁剪和非有限梯度标记检查

```bat
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cpu --tp-size 2 --dp-size 2 --clip-norm 0.01 --check-overflow
```

这会调用你的实际裁剪和非有限梯度检测函数。
Inf 只在一个 rank 的测试梯度中临时注入，检查所有 rank 的标记一致，
然后恢复正常梯度。它检验标记传播，不等同于验证 AMP GradScaler。

### 可选：无 bias 和手工 Attention 分支

```bat
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cpu --tp-size 2 --dp-size 2 --no-bias --attention manual
```

### 可选：float32

```bat
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cpu --tp-size 2 --dp-size 2 --precision float32 --atol 1e-6 --rtol 1e-4
```

不同 TP、DP、梯度累积次数都可通过相应参数修改。
完整参数说明：

```bat
python nanoGPT_tp_verification/verify_tp_dp.py --help
```

## 第四步：之后在 GPU 环境补测

CUDA 数值验证的 TP×DP 进程分别使用独立 GPU。
在具备对应 GPU 和 NCCL 的环境运行，例如：

```text
python nanoGPT_tp_verification/verify_tp_dp.py --project . --device cuda --tp-size P --dp-size D
```

把 P、D 替换为你选择的正整数。也支持现有 torchrun 启动方式，
此时 --dp-size 可省略，从 WORLD_SIZE/TP 推导，适用于跨节点的数值验证。
CPU 自启动模式使用 FileStore，兼容 Windows 的 spawn 启动方式。

### 实际 train.py 恢复训练

```text
python nanoGPT_tp_verification/verify_train_resume.py --project . --tp-size P --dp-size D
```

这个自动启动器面向单机 GPU 环境，要求可见 GPU 数至少为 P×D。
它运行你真正的 train.py 三次：
1. 从零连续训练到目标 checkpoint。
2. 从零训练到中间 checkpoint。
3. 从中间 checkpoint 恢复，再训练到同一目标 checkpoint。

它生成独立的小数据集和输出目录 verification_runs，不使用你的原有训练输出。
三个运行的学习率 schedule 一致，dropout=0.2，compile=False。
默认 dtype=float32；之后可用 --dtype float16 或 --dtype bfloat16 补测。
NumPy 是实际 train.py 的依赖，如缺少可在该环境安装 numpy。

比较内容：
- 所有 TP 模型分片、optimizer、scaler、model_args。
- 训练相关配置、TP/DP 元数据、iteration、best_val_loss。
- 每个 rank 的 CPU/CUDA RNG、TP RNG、train/eval generator。
- 每个 rank 预取的 X/Y。
- RNG 字节和 X/Y 要求完全一致；模型浮点容差可通过参数调整。

比较的是目标 iteration 开始、完成 eval 后保存的 checkpoint 边界，
与当前 train.py 的保存位置一致。它不会把最后一次日志当作已保存模型。
不比较 out_dir、init_from 和本次停止位置 max_iters，因为三次运行这些值不同。

日志、数据、checkpoint 和 report.json 留在独立输出目录。
出错时先查看对应 .log 的最后几行。

如果 GPU 环境已经产出 continuous 和 resumed 两个目录，
可以把目录下载到本地 CPU 环境，只做文件比较：

```text
python nanoGPT_tp_verification/verify_train_resume.py --compare-only CONTINUOUS_DIR RESUMED_DIR --tp-size P --dp-size D
```

这些占位符需换成实际目录和并行度。

## Git 操作时间

CPU 数值检查通过后，可先记录一个本地提交，
标明“CPU 数值验证通过，GPU 训练集成待测”。

检查当前分支和修改：

```bat
git status
git diff -- model.py train.py
git add model.py train.py
git add nanoGPT_tp_verification
```

如 parallel_utils.py 是你新建或修改的，再单独执行 git add parallel_utils.py。
查看已暂存改动后提交：

```bat
git diff --cached
git commit -m "Add TP DP numerical verification"
```

report、verification_runs 和 checkpoint 是本地验证输出，提交时选择上述源文件。
GPU 检查还未做时，在进度说明中写“待测”。push 到自己的仓库前，
用 git remote -v 确认 origin 指向自己的 GitHub 仓库。

## 验证范围和生成时检查

本包覆盖当前 scratch 模型与 load_dense_state_dict 的参数映射。
GPT-2 from_pretrained 的转置、TP 分片加载仍是后续适配项。
compile、GPU 性能和长时间训练稳定性不在当前数学对齐测试的范围内。

生成环境未安装 PyTorch：本包已检查 Python 语法、CLI 和静态诊断，
实际 PyTorch 数值检查尚未在生成环境执行。
实际状态以 VALIDATION_STATUS.json 和你运行后输出的结果为准。

资料：
- PyTorch 分布式后端：https://docs.pytorch.org/docs/stable/distributed.html
- PyTorch DDP：https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- torch.testing：https://docs.pytorch.org/docs/stable/testing.html
- 原版 nanoGPT：https://github.com/karpathy/nanoGPT
