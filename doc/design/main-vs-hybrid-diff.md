# main -> GDN hybrid 分支：全面对比讲解

对照对象：
- **main**：`/data/tangyang/intel-accel-for-llm-main/kvshrink/kvshrink_connector.py`（566 行，纯 attention 模型）
- **本分支**：同一文件（2798 行，支持 Qwen3.5 这类 GDN/mamba 杂交模型）

读法建议：先看第 0 节的总判断（一句话：**没有一个概念是从天上掉下来的，
每处改动都对应一个"main 的做法在递归状态面前失效"的具体场景**），
然后按兴趣跳节。每节结构统一：main 怎么做 -> 我们改成什么 ->
为什么必须改 -> 不改会怎样。

--------------------------------------------------------------------
## 0. 总判断

main 解决的问题是：**attention KV 块的外挂**。块是均匀的、可切片
的、和 vLLM 自己的前缀缓存同构的，所以 main 可以把 vLLM 的 hash 直接
拿来问店、把块的 GPU 地址直接递给引擎。

GDN 打碎了所有四个前提：

| main 隐含假设 | GDN 的现实 | 破坏后果 |
|---|---|---|
| 一层 = 一个张量 | mamba 层 = 一个存储上两个异形张量(conv+ssm) | 引擎拒绝调用 |
| 块之间独立 | 递归状态是链式的，只有边界上有完整快照 | 按"块存在与否"判命中会算错 |
| hash 表达块位置 | hash 只表达 token 前缀；哪一组、哪个快照还要另说 | 键撞车/命不上 |
| 读完了就是读完了 | CURR slot 有 preprocess_mamba 在 forward 前改写它 | 时序错了读到旧状态 |

于是每一个文件的差异几乎都是这张表的直接推论。566 行到 2798 行不是
功能膨胀——多数新增行数是上面四行注释掉了下的交换成本。
反过来说也一样：**凡是能保持 main 形状的地方全部保持了**
（单文件单类、方法名、三步流程、异步生命周期形态、构造期绑定、
CPU/加速卡绑定逐行相同），下面会反复看到这种"形状不动、内核换血"
的模式。

另外一份账要先摆清：有六个 hook main 根本没有实现，是 v0.23 给的
免费能力，我们接了：

```
update_state_after_alloc   （main 里压根没有这个方法）
request_finished_all_groups（HMA 分配器下 v0.23 实际调用的那个）
SupportsHMA 接口            （声明我们懂 hybrid 内存分配器）
get_num_new_matched_tokens 返回值在无命中时的退化处理等
on_cached_request          （新请求之外的 running 请求通道）
build_resumed_load_meta    （预占恢复）
```

为什么 main 能白嫖不接？因为纯 attention 模型上这些回调里没有数据
时 vLLM 也走得通——而 resume 是杂交模型的日常事件（GDN 请求更重、
更容易被抢占），不接它的代价就是恢复后输出乱码，没有降级可言。

====================================================================
# 第一部分：寻址与命中 —— "什么算缓存过"

这是全部分支改动最根本的部分，其余一切从这里长出来。

## 1.1 store 键从一维加一维：group label

**main 怎么做**：键是 `("kv", chunk_hash)` 层面的事——KVStore 默认
label + 内容 hash 就够了。同一个前缀 hash 全 engine 只有一份数据，
`store.has(hash)` 就是问"这个前缀存过没"。

**我们怎么改**：label 变成 `g{组号}`，键 = (组, hash, 层名)。模型
basename 进目录的方式与 main 逐字相同（裸 basename），没有 namespace。

**为什么必须改**：唯一硬需求是"分组"。一本 store 账里现在住着四个
组——一个 attention 加三个 mamba，同一个前缀 hash 在每个组各有一份
实体；而存在性台账（Record）按 `(label, chunk)` 记账、**不带层名**，
共享 label 时一组落账会把别的组的存在性也标成"有"，命中判定直接失
真。层之间不用分：数据键自带全模型唯一的层名。跨 rank 同样不用分：
每 rank 持久化到自己的 `{model}_rank{r}` 目录、controller 只开
rank0 那本，同标签天然读写一致。

**为什么没有 namespace**：曾有，删了。sha256(model/dtype/tp/schema)
拌进每个键的 compute_namespace 是随同名 HybridStore 机制引入的，机
制后来整个删除，namespace 成了孤儿。main 对 dtype/tp 串味一直是裸
奔姿态且被接受为既有取舍——单独给 hybrid 上锁不在 GDN 支持范围内；
这是"最小增量"纪律的直接体现。中途还走过两个弯路也都拆掉了：
`_worker_key` rank 重映射（两边目录不同，各自相等即可，函数纯多余）
和 label 里的 r{rank} 段（同上，自娱自乐）。PP != 1 仍在创建期显式
拒绝（PP 把层切开到 rank，页字节只剩半个模型的），speculative 同理。

**不改会怎样**：不分组，一本账里住进两种状态实体，存在性互串，命
中判定失真——这是仅有的必须改的一处；其余维度与 main 完全对齐。

## 1.2 存在性查询从批量 bool 变成"任意异常 = MISS"

**main**：`store.has(block_hashes)` 拿回一批布尔列表，塞进
`ReqState.existence_cache`，之后保存过滤也用它。异常？没想过——炸
就炸了。

**我们**：`lookup_boundary()` 单发查询 + 统一裁决：**错的 hit 污染
输出，错的 miss 赔一次重算**，所以任何异常折成 MISS（fail-closed）。
全文件唯一一处故意吞异常，方向是故意的。

TP 锁步保存使"rank0 有 = 大家都有"；个别 diverge 的 rank 会在 load
load 时被 native 断言爆出来。main 的 has() 同样是 rank0 视角，但它
不需要解释这件事，因为它没有多 rank 一致性可以失去。

## 1.3 命中检测从"顺数第一个 miss"变成定点迭代

这一节是最能说明"为什么不能偷懒"的案例。

**main 怎么做**：

```python
matched_blocks = next(
    (i for i, exists in enumerate(existence_cache) if not exists),
    len(existence_cache))
matched_tokens = matched_blocks * self.block_size
```

一行线性扫描：从头数到哪里第一个不存在为止，前面全是命中。
这在纯 attention 世界是正确的——块独立，前面的不存在不影响后面的
存在意义（反正连续前缀截断到 miss 处）。

**我们的问题**：两套规则要同时满足，还得相互牵制。

- attention 组：还是 main 那套——下行闭包的前缀扫描；
- mamba 组：完全相反——**右往左找最近的完整快照**。中间某个快照不
  在没关系，只要更晚的那个在，就从那里恢复。两种方向相反的扫描；
- 还要双方互相让步：attention 说"我能恢复到这里"、mamba 说"我只能
  到这里"，取交集后各自重新扫，直到不再变化（fixed point）;
- 最后减一：最后一个 prompt token 必须重算（logprobs 和递归状态都
  需要），并对齐到 align_size。

**我们怎么改**：不是手写这些规则——而是造了个假块池
（`_StoreAsBlockPool`）把"这 hash 缓存了吗"指向外部 store，然后**直
接调 vLLM 自己的 `find_longest_cache_hit`**，每种组一次。命中判定规
则零拷贝，上游演进自动跟上；我们替换的只有数据源。

同时返回类型收紧：基类允许 `int|None`（异步延迟应答），我们恒同步
（Record-gated 查找永不 None），签名收窄为 `tuple[int, bool]`。

**不改会怎样**：把 main 的线性扫描搬过来，attention 组勉强能跑，
mamba 组要么永不命中（快照不在前缀线上）、要么错位恢复（从错误的
非边界点上切）——后者又是静默垃圾。

## 1.4 新增一道全局约束：块尺寸必须归一

main 从不考虑这个（它的 block_size 是 config 全局唯一的）。杂交模
型允许多个块尺寸共存，但我们的地址方案要求"一个 block hash 同时给
所有组指第 i 个块"。这不是我们自己发明的约定——是 vLLM resolve 出
GCD 尺寸的推论。解析期发现违反直接拒绝并说明原因，绝不猜。

## 1.5 操作员校验比 main 多四刀，每一刀对应一种"无声死法"

main 的校验面：FlashInfer 拒绝（register_kv_caches 里）、lossy codec
根本没设防。我们的：
- FlashInfer（保留 main 的那一段原样）
- speculative decoding 拒绝——spec 扩宽 GDN gather 读多列，外部快照
  只恢复第一列，剩余列未恢复但 core 已记账不用算；
- 非 align mamba 模式拒绝——顺带点出 vLLM 关 prefix caching 时会静
  默改写的坑，报错信息里给逃生参数;
- PP != 1 拒绝——PP 切层到 rank，rank 手里的页只有半个模型的字节却
  用完整的键；
- lossy truncation 启动即拒——对 int8 opaque 页它会打爆 bf16 指数
  位，attention 近似还能活，GDN 状态污染=吐错 token 无处报错。

共性：**每一种都能跑起来不报错地出错**，所以全在门口 fail-stop。

====================================================================
# 第二部分：数据的形状 —— 让引擎能吃下去

## 2.1 Canonicalizer：整个分支最重要的一个新类

**main 怎么做**：拿 kv_caches 里每个张量直接注册给 store，只需要算
一下 `block_dim`（MLA 或 [2,...] 布局取 0 否则取 1——一根 if 判断完事）。

```python
block_dim = 0 if self.use_mla or first_kv_cache.shape[1] == 2 else 1
self.kvstore = KVStore(..., block_dim=block_dim, kv_caches=kv_caches, ...)
```

**为什么必须改**：不是引擎拒收异形张量——put 对 tensors 字典逐条目
循环、每层独立入账，全端上去也合法。真正的问题是可寻址性：GDN 层天
生是一份存储上的 conv_state 和 ssm_state 两个异形张量，谁也没法当
"块表的行"按块号切片搬运。

**我们怎么改**：Canonicalizer 给每层构建统一的
`(num_blocks, page_bytes)` int8 视图——把"一个逻辑页"重新定义为字
节平面，conv 在前 ssm 在后拼起来正好一页。dtype 全部坍缩成 int8，
chunk_dim 全部坍缩为 0，整个引擎接口被拉平。（这也与后来砍掉引擎
chunk_dim/block_dim 参数的方向一致——视图归一化了那些参数就没有存
在的意义了。）

副产品：字节平面视图让 mamba 快照也获得了"按块读写"的能力，这就是
为什么 store 层面对的仍然是熟悉的 put/get 而无需知道底下是什么模型。

顺带处理了 split-K/V：某些布局 K/V 不相邻，平铺 stride 会跳到错误
字节——探测到就保留 k/v 两个半页视图，读侧拼接。main 对此的处理是
那个 `shape[1]==2` 的 if 加信赖于 iaxl 内部处理，够不够取决于布局永
不变化；我们遇到了实际变化的布局，就显式化了。

## 2.2 存储尺寸预检（storage_size_bytes）

view 视图建出来前先核对 descriptor 要求的字节数 <= 底下 storage 的实
际大小。as_strided 越界不会报错——会静默把邻近块的垃圾当数据读。
main 没有这道检查是因为 main 的视图来自 vLLM 自己的 allocate 结果，
字节预算天然吻合；我们从 descriptor 反推视图，需要一个来源外的校验。
原则 2 类检查的典型样本。

====================================================================
# 第三部分：调度侧生命周期 —— "计划怎么排"

三步骨架完全一致（查询 -> 分配时记录 -> 步末打包下行 → teardown），
但由于恢复了 vLLM 提供的三个免费 hook，骨架长出了肉。

## 3.1 每组一块表：ReqGroupState 是新的最小账本

**main**：请求级只有 `block_ids: list[int]`——一张全局表，vLLM 块池
全局唯一。

**我们**：每组一张（`groups: tuple[ReqGroupState, ...]`）。因为 HMA
下注意力组和 GDN 组有独立的分配池，同一请求在不同组的第 i 块毫无关
系。硬要一张表就得作 defer 交叉映射，比 N 张小表复杂得多。

## 3.2 三条入口路径接三路流量

main 时代只有一条路进来：新请求 get_num_new_matched_tokens ->
update_state_after_alloc。running 请求从来不通知 connector（vLLM 没
有那种 hook）。main 怎么活下来的？看它的 `_add_request_to_save`：
由 build_connector_meta **自己走 scheduler_output 抓** new_reqs 和
cached_reqs——每次 prefill 步抓新增块。它在纯 attention 世界够用的根
本原因是：块意义唯一、hash 顺序确定、prefill 长块的 save 即全量。

hybrid 下这条路裂成三股必要的需求：

1. **new 请求**：还走 get_num_new_matched_tokens（同 main），
   `_track_new_request` 登记（原名 on_new_request，撞 v0.23 基类 hook
   名被迫改名，回归教训）。
2. **resume 请求**：preemption 恢复的 GDN 请求丢状态的话损失巨大且输
   出乱码；vLLM 把它们放在 cached_reqs.resumed_req_ids 里、不在
   scheduled_new_reqs 里。专用 build_resumed_load_meta + fail-closed
   检查（accepted tokens > 0 而无可恢复页 => raise，宁可 EngineCore
   fatal 不进 forward）。
3. **running 请求**：通过 on_cached_request 同步新增块 id 并回滚游标
   ——这是本分支新增的核心簿记之一。

## 3.3 增量保存游标及其回滚：main 不需要的概念

**main 怎么做**：每个请求维护 `num_seen_blocks` 计数；保存候选 =
新见到的块中 existence_cache 显示不存在的那些。重启即失忆（进程内状
态），无游标无回滚可言。existence_cache 恰好兼做去重——已经在店的
就不重发了。

**我们**：`next_stored_chunk_idx` 每 group 一个，emit 即推进。判定上
需要的结果相同，语义升级在于回答"哪些已发的可能其实没落地"——

预占恢复时卷回 floor(N/bs)，哪怕 progress 缺失（fail-closed 到 0 重
发一切）。动机链条：写出去的数据可能因为异步还在路上而最终丢失，重
发幂等无害；漏发则那个边界永久不可再用且无任何症状。main 的幂等选
择是把"是否已在店"交给 existence_cache 实时问一遍；我们有同样的信
息（Record-gated 查询天然排除已提交项）却还是要回滚机制，是因为**存
在性判定本身可能说谎**（写在途中的内容还没入账，has 返回 false 但稍
后会 true——反之若结算顺序不同又会 false negative）。与其指望平衡
竞态，不如把"我没收到确认的一律视为未发送"做成硬规则。

**不改会怎样**：预占过一次的请求，其后续保存永久认为已经完成——从
此该请求的所有增量边界全部消失，冷静地直到该请求结束都不被发现，
后期重放命中失败率暴涨。这类 bug 以前真实存在过（略）。

## 3.4 snapshot_boundary：权威恢复点只钉一次

main 的加载范围可以从 num_external_tokens 除一下 block_size 算出来
（load_start/end 由 computed 推导——admin 的 view_world 是线性的）。

不行，理由有三（docstring 里写了三条，概括如下）：除法不一定落在合
法 mamba 边界上;每组独立计算会把组装规则复制三份;resumed 请求不在
scheduled_new_reqs 里而是同一 builder 出来的第三种入口，alloc 时刻
算计划必然漏掉它们。所以在查询那一刻定死边界数，后续一律引用。
update_state_after_alloc 从"构造加载清单"退化为"记录事实"。

## 3.5 request_finished 双入口与延迟释放

**main**：单一 `request_finished(request, block_ids)` 返回
(True, None)，把释放推迟到 get_finished。已经做了延迟释放这个聪明事。
注意 main 压根不知道自己的块有几张表——`block_ids: list[int]` 就行。

**我们**：HMA 分配器开启时（默认）v0.23 实际调用的是
`request_finished_all_groups(request, per_group_block_ids)`，base 里
的 default 会路由到 request_finished——所以我们俩都实现，all_groups
薄封装转发，保证两条门进来的行为一致。这个 free-for-all 在纯
attention 上无所谓（两者只会被调其一），但在我们这里有 actual 数据
流经 all_groups 版本时如果实现漂移就麻烦了。统一点就是全部委托同一
份 request_finished。

## 3.6 校验升级的一处细节

main 对 update_state_after_alloc 有 `state is None -> RuntimeError`。
我们删了这个防御（批次 2 清理的一部分）：v0.23 保证 alloc 前必有查询
入册；守卫真触发说明上游协议被破坏，此时 AttributeError 于运行结果
无异于 RuntimeError。main 风格优先——多一层包装少一分相似度。

====================================================================
# 第四部分：worker 执行面 —— 动手术最多的地方

## 4.1 start_load：从混合批变成分组批 + GDN 屏障

**main 怎么做**：sync 请求合并成一个大 get、async 请求逐个 get 各自
管理，到此为止。等待发生在 wait_for_layer_load 中按 layer_names 过滤
同步任务等待；异步任务的早期晋升第一 N 层归 get_finished 管。

**我们必须加的三件事**：

a) **按组合并**。不是引擎吃不下混批——是 label 契约：显式 label 的
   调用按调用整体记账结算，一次只能结算一个组的全部层，所以调用天然
   按组发起；一锅端四个组等于结算四本只写了一部分的账页。

b) **GDN host-block 屏障**。vLLM 只对 attention 层调
wait_for_layer_load，mamba/GDN 层在 forward 里没有任何 hook 经过。所
以 GDN 任务如果开头不等，永远不会有人等到它。就在 start_load 末尾把
recurient 任务一次性 pop 出来 host-block。

   代价是一次递归状态的延迟（总几个 tens of MB vs 相对一整次 forward
   微不足道）；换来的是不存在"GDN 未等待进 compute"这类事故类别，且
   无需维护任何"哪个 attn 负责哪个 gdn"的映射机构。简单压倒聪明。

c) **async 请求不在 sync 大批**——他们被 park 着、不参与本步
forward，如果混进去会被 b) 屏障误伤。单独登记到 _AsyncLoad 册子。
gate 的计算里 recurrent 全收 + 前 N attention 层（这是 _decide_async
的结果）。

## 4.2 wait_for_layer_load / wait_layer_load：排水范围扩大

main 每次只等当前层的同步任务。我们除了等当前层，还要顺手把每一个
"已 released 的 async 册子"里当前层的余款一起排水。原因是 worker 无
从知晓 step 里到底覆盖了哪些请求——**等早一步只赔一瞬间的空等，不
等则是给未恢复内存跑了 kernel**，不对称得离谱所以永远多做。

## 4.3 save 流水线：从每层一次 put 变三级接力

**main 怎么做**：save_kv_layer 里对每个待存请求做一次同步 put（其实
engine 的 put 本身就是异步 D2H+zip，所以只是 submit）。wait_for_save
是 return，是空壳。释放契约：put 任务挂在 _current_put_tasks、在
get_finished 逐个 put_wait 收割，收干净才 completed 上报，vLLM 才能
回收块。这个生命周期设计是 main 最精妙的地方，**我们完全保留了下来**。

**我们动的是 submit 端，两级流水化**：

- attention 层出口钩子（vLLM 对每个 attention 层必调 save_kv_layer）
  当场提交**该层自己的页 + 该层之前积压的 mamba 段**——段里的 GDN 层
  kernel 已执行完毕、数据已是终值，跟着搭便车。万恶之源 bug 在这里：
  执行序穿插三个 mamba 组、一段横跨三组时全用了首段的 label 提交，
  导致 g1/g2 边界从未落账，保存成功却全员 MISS——修复是段内再按组分
  拨各找各妈。
- 尾部没人管的层（最后一段 mamba、没触发钩子的 attention 层）由
  wait_for_save 兜底补交，并留一行"pages submitted"日志——这条日志
  是 gate 的锚，start_load_kv 开头的镜像日志同样如此，保证"保存开着
  但啥也没干"在日志上肉眼可见。

## 4.4 get_finished：从两大块变三段式收割

结构对比一眼看清：

```
main:
  poll async loads（两个字典、early-promote 状态机）
  drain finished reqs 的 puts

ours:
  poll_finished_loads()     <- poll loads，但状态机搬进了 _AsyncLoad 一个类
  合账 deferred finished    <- 相同
  逐 req: 先收它的 in-flight async load 余款
        再收它名下 puts    <- 相同思路 + ctx 已清则划过的 dedupe bug 修复
```

双重 drain bug 值得讲清楚，它是共享数据结构自然孕育的竞争：多个请
求贡献同一边界时（并行 decode 常态）,_SaveCandidate 把 put 挂到每个
贡献者名下。第一个请求收尾时 put_wait 已经 finalize 了任务（ctx 清
None）;第二个请求轮到时再去碰它,native 断言拍死 worker。修复 =
收账前看 ctx 是否已被清空，等于"这笔别人付过了，我销我的账"。

main 为什么没有这个 bug？main 的 put 按请求独立、从不合并同 hash 的
重复工作——代价是相同页面会被多次写入（无所谓，幂等），从而不会有
一份任务挂两个名字的情况。我们把去重当优化引入时，就要负责处理
"去重的副作用是所有权共享"。这是一条真正的架构税，我们付了。

## 4.5 异步加载早期晋升的状态机被收割简化了

main 维护四本字典（_pending_load_tasks / _pending_load_layers /
_early_promoted_tasks / _active_promoted_tasks）+ start_load 时的晋升
promote + last-layer-reset 语法. 我们把这团状态机收进单个
_AnyLoad dataclass（layer_tasks dict + gate_layers set + released
bool）：poll 时检查 gate 是否齐 -> 齐 = released 上报;剩余楼层由逐
层 hook 排水，排完即弹出。状态本质从"四个平行数组手工保持一致"变成
"一个请求生命周期的显式相位机"——**这个重构与功能无关，纯粹是复杂
度消化**；同样的信息在两套平行字典里靠纪律维系的做法，扩展到每组一
张表的规模时就会自相撕裂。

====================================================================
# 第五部分：不变的清单（评审时可直接放过的部分）

下列各处我们刻意做到与 main 逐行 / 逐词一致，审查时无需花时间：

- 单文件单类组织、Scheduler/Worker Side 两道横幅
- `_bind_cpu_affinity` / `_bind_intel_accel`（逐行相同）
- FlashInfer 检测段（register_kv_caches 开头,原样保留）
- requires_piecewise_for_cudagraph -> True
- 异步加载 env 配置层（async_load_config.py 未动，上游文件）
- 生命周期姿态：请求结束延迟释放、get_finished 上报、(True, None)
  返回约定、put 结账语义（ctx=None 视为已完成）
- 结构性方法的形态（_metadata 类型断言风格、RequestMetadata 包装）

这份不变清单和第二部分的四种变换一样重要：评审的目标是看清"哪里
变了、为什么不得不变"，而不是欣赏变形本身。

====================================================================
# 附：一句话总结每个核心差异

- 寻址：标签加组号——一本账住进四个组，存在性台账不带层名必须分账页
- 命中：一条扫描线变定点迭代——递归状态的命中方向天生和 attention 相反
- 视图：原生张量变字节平面——异形 conv/ssm 没法按块号寻址，统一 int8 行把引擎接口拉平
- 组装：alloc 时算区间变 boundary 钉死——除法落不到合法快照边界上
- 记账：一片计数变游标回滚——写可能在路上，重发幂等漏发致命
- 等待：attn 层入口等待变 attn 入口 + GDN 屏障——GDN 没人管，只能开头拦
- 写出：per-req 提交变流水线接力——attention 层出口就绪即可开搬
- 收割：双态字典变生命周期对象——同样的信息重组成本更低
