# R120-P2 新实例执行受阻记录

- 团队：team-anchor
- 记录日期：2026-11-26（Asia/Shanghai）
- 指定科研源码 commit：18d4a3ec7da6851ca632ff73cd2a3d92674a1fea
- 状态：NOT_STARTED / ENVIRONMENT_UNAVAILABLE
- diagnose.py 未在新实例启动；无 GPU 诊断结果、性能结果或诊断进程退出码。
- 未修改 research_code/；未重跑 R120-P1。

## 实际观察

新实例 SSH 登录成功。对 /root 和 /opt 的最大深度 5 搜索未返回 model.safetensors、pyvenv.cfg 或 diagnose.py。此前约定模型和虚拟环境路径不存在；本次查阅的恢复目录也不存在。
此搜索仅覆盖上述路径和深度，不能证明整个文件系统没有 Python 或模型。

尝试调用 python 导入 torch、transformers 时，远端 shell 返回：
```text
bash: line 1: python: command not found
```
SSH 子进程返回状态为 127。该值是环境探查命令的退出码，不是 diagnose.py 的退出码。完整探查命令和回显见 connection_check.log。

## 执行边界和后续前提

旧实例资产未保存，用户要求以 GitHub 为准恢复。当前轮明确要求不重新准备环境、下载数据或修改科研代码，因此本次探查后未启动环境恢复或科研诊断。
要执行指定诊断，需先恢复可用 Python/Transformers 4.48.3 环境及经过原哈希校验的 LayerSkip 1B 权重，再从指定 commit 执行。R120-P1 数据已在仓库，可直接复用。

仓库中的源码、实验数据与日志可以用于恢复；模型权重尚未备份至 GitHub，本记录不声称已保存模型。
