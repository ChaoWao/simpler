# A2/A3 worker 退出协议：必须先 CLOSE，再返回

[英文原文](../../troubleshooting/a2a3-worker-retirement.md)

A2/A3 的驻留 worker 不能仅仅因为已经确认收到 EXIT 就直接返回。
它必须等到 AICPU 关闭该组 worker 的 fast-path 寄存器窗口，
并通过全局内存（GM）明确发出放行信号后，才能返回。

这一要求同时适用于 `tensormap_and_ringbuffer` 和 `host_build_graph`。
本修复不新增借用外部 stream（borrowed-stream）/L1 支持，也不改变 A5 的退出协议。

## 这里的 fast path 是什么

AICPU 通过 MMIO 访问每个 AICore 的寄存器窗口：写入 `DATA_MAIN_BASE` 来分发任务，
读取 `COND` 来获取 ACK/FIN 状态。AICore 则读取自己的任务分发专用寄存器（SPR），
并写入自己的 COND SPR。
向 `FAST_PATH_ENABLE`（偏移 `0x18`）写入 `0xE` 表示打开窗口，写入 `0xF` 表示关闭窗口。
这是设备侧的任务分发机制，不是 torch allocator 配置、主机任务队列选项，也不是 ACL event。

## 旧协议缺少哪条顺序约束

旧协议建立了下面两条执行链：

```text
AICore: observe EXIT -> write COND=EXITED -> return
AICPU:                 observe EXITED   -> write IDLE -> CLOSE
```

两条链存在共同的前序事件，却没有约束各自后续操作之间的先后关系。
因此，下面的执行顺序并没有违反旧的软件协议：

1. AICore 观察到 EXIT，并发布 EXITED。
2. AICore 直接返回，此时它的 fast-path 窗口仍然打开。
3. AICPU 观察到 EXITED，写入 IDLE，然后关闭窗口。

EXITED 只能证明 worker 已经停止接收任务，不能同时证明 AICPU 已经完成窗口管理，
因为后者发生在收到 ACK **之后**。缺失的顺序约束就是 `CLOSE -> worker return`。

主机侧互斥锁、stream 之间的 event，或者等待整次启动（launch）完成，
都不能反过来为该次启动内部的这两个操作补上顺序约束。
发生这个竞态并不要求有多个在途的 graph 执行。
同样，异步错误最终在后面的某个 native 算子处报出，也不能证明是那个 native 算子导致了错误。

## 两阶段退出协议

正常退出沿用现有的所有 AICPU 线程完成汇合机制：最后到达的参与者负责让整组 worker 退出。
退出处理分为以下几个独立阶段：

1. 先向所有已经初始化的 core 发送 EXIT，再开始等待任何 ACK。
2. 使用一个共享的截止时间，收集整组 core 的 EXITED。
3. 对已经确认退出的 core，将任务分发状态重置为 IDLE，并关闭对应窗口。
   读回 MMIO 窗口并确保该次读取完成，然后才能通过 GM 发布返回许可。
4. 以 release 语义，向每个已关闭窗口的 core 的控制字写入 `AICORE_POST_CLOSE_RELEASE`。
5. AICore 观察到该控制字的放行值之后，才允许返回。

A2/A3 实机上的 worker 使用 `ld_dev` 执行非缓存／绕过缓存的读取，
并在返回前执行 `dsb(DSB_DDR)`。仿真版本使用具有 acquire 语义的原子读取。
这里没有把 `ld_dev` 当作通用的原子读－改－写（RMW）操作，也没有把它当作 C++ acquire 操作；
这里实现的是单写者、逐 core 的交接。
平台对 MMIO 操作完成顺序的保证，与 GM 放行信号的可见性，是协议中两个不同的部分。
仅靠 release store 并不能确保已经发出但尚未完成的 MMIO 写入真正完成。

紧急退出使用同一个退出处理函数，并通过退出处理权标记（ownership latch）避免重复处理，
防止后续正常收尾再次写入已经放行的 worker 的寄存器。
如果某个 core 未能在截止时间前确认退出，就不会关闭它的窗口，也不会向它发出放行信号；
其他能够响应的 core 仍然可以完成退出。
无响应的 core，以及窗口从未成功打开的 core，仍由现有的主机恢复路径负责处理。
本补丁不引入新的 reset 操作或恢复策略。

## 缓存行隔离与跨次启动的状态归属

`Handshake` 包含两条相互独立、各自按 64 字节对齐的缓存行：

| 缓存行 | 写入方／访问方式 | 用途 |
| ------ | ---------------- | ---- |
| 第一条 | 现有的缓存式状态报告／任务发布协议 | 启动身份信息、就绪报告、任务分发载荷指针 |
| 第二条 | AICPU 原子写入；AICore 绕过缓存读取 | 窗口关闭后的返回许可 |

返回许可控制字不能与 AICore 启动报告或退出报告所刷新的缓存行共用一行。
否则，旧的缓存报告在回写时，可能覆盖 AICPU 刚刚写入的放行值。
在该控制字处于使用状态期间，不能对它所在的缓存行进行缓存式写入或缓存维护操作。
编译期的布局断言和可平凡复制性断言，用于保护主机／AICPU／AICore 共享的数据布局。

负责启动的 leader 会先以原子方式重置返回许可控制字，然后才发布握手初始化信息，
并且这一重置发生在任何寄存器窗口打开之前。
因此，前一次启动留下的值 1 不会错误地放行新一次启动。
这种复用依赖现有 runtime 生命周期的前提：重新初始化同一块存储之前，前一次启动已经完成。
这并不意味着同一份 runtime 数据支持多个相互独立的运行并发执行。

HBG 与 A5 共用 `Handshake` 定义，所以 A5 上这一内部数据结构也从每个 worker 64 字节增至 128 字节；
但第二条缓存行在 A5 上仍不使用。
主机侧和设备侧二进制必须配套重新编译。
公开的 Python API 和 A5 的执行协议保持不变。

## 验证证据与复现边界

执行器回归测试将两种 runtime 的 **实际生产版本** A2/A3 AICore 执行循环，
分别与模拟的寄存器存储链接起来。
在刻意不执行 CLOSE 的情况下，未经修改的 main 实现
（提交 `fab1a41e2fd5bbefb9eb18e59609876e63297e98`）中的四项 AIC/AIV 检查全部失败：
worker 会立即返回。
加入交接协议之后，这些检查通过。
配套测试还覆盖最终能够返回、连续 32 次启动复用同一返回许可控制字、
整组 ACK 的顺序约束、部分 core 超时，以及拒绝无效目标。

```bash
cmake -S tests/ut/cpp -B tests/ut/cpp/build
cmake --build tests/ut/cpp/build --parallel 2 \
  --target test_a2a3_tensormap_and_ringbuffer_retirement test_a2a3_host_build_graph_retirement
ctest --test-dir tests/ut/cpp/build -R '^test_a2a3_.*_retirement$' --output-on-failure
```

这些测试验证的是软件层面的顺序约束，并不模拟真实芯片在寄存器窗口退出阶段的硬件风险。
促成本次修复的下游 L1 ACLGraph 排查还提供了另一组独立证据：
仅移除窗口关闭后的等待，原本能够通过 1280 次 replay 的运行，就在第 2 次 replay 出现了同类 vector timeout。
故障 PC 位于驻留式 AIV wrapper 的末尾，而不是后续 native 算子内部。
这是下游集成场景的证据，不代表上游 main 已经支持或能够复现该 L1 graph 工作负载。
本文也不据此断言任何未公开的硬件内部状态机故障。

PR 的验证记录区分了本次新增的上游测试与上述历史下游证据，并列出了无法使用的工具链。
