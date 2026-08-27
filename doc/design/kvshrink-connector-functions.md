# KVShrink Connector 逐函数设计说明

评审对照用文档：逐函数说明它为什么存在、为什么是这个形状、以及本分支
相对 main / 相对早期版本做过哪些删改及理由。与 `kvshrink-hybrid.md`
（算法与生命周期叙事）互补，本文只回答"代码为什么长这样"。

适用对象：`kvshrink/kvshrink_connector.py`（单文件、单类
KVShrinkConnector），对照范本是 main 的同名文件（566 行纯 attention
单连接器）。

--------------------------------------------------------------------
## 0. 总体设计：为什么整个文件只有这一个形状

**单文件单类。** main 的 kvshrink 就是一个文件一个
KVShrinkConnector 类；本分支保持完全同构：同样的模块函数区、同样的
小类区（Canonicalizer 等）、同样两道方法横幅
（`# Scheduler Side Methods` / `# Worker Side Methods`）。调度侧和
执行侧的全部方法落在同一个类上，vLLM 每个进程只实例化其中一个角色，
另一个角色的方法自然不可达——不需要两个文件、也不需要继承分层。
这样做的好处是可对照性：评审时可以拿 main 的方法逐个对着讲，多出的
每个方法都对应一条 hybrid 特有的必要性，而不是结构噪音。

**ONE path，对 groups 编程。** vLLM v0.23 已经把模型描述成 KV cache
group 列表：纯注意力模型是一组的特例，GDN/Mamba 杂交模型是多组的
一般情形。整个连接器只写针对 group 的逻辑，从不问"我是不是
hybrid"——没有 if-hybrid 分支，就没有两条路径漂移的可能。

**失败哲学。** 三条红线，全部在删改中贯彻：

1. 会自然炸的预检删掉——预检只是换报错文案；
2. 不炸但会静默产出错误数据的检查保留——这类错误没有别的防线
   （典型：as_strided 越界读、寻址不一致、curr-slot 族校验、
   speculative / FlashInfer 拒绝）；
3. 操作员环境校验、main parity、profile-run 可达的 None 守卫保留。

推论：文件里唯一的宽容异常处理是 `lookup_boundary` 的
except->MISS（方向正确的降级），其余一律 raise 到 EngineCore fatal。

**契约挂钩点。** 所有回调签名逐一核对过 vLLM v0.23 基类
KVConnectorBase_V1 / SupportsHMA：
get_num_new_matched_tokens（基类返回 `tuple[int|None, bool]`，我们收窄为
`tuple[int, bool]`，因为 Record-gated 同步查找永不返回 None）、
update_state_after_alloc、build_connector_meta、request_finished /
request_finished_all_groups（HMA 默认走后者）、start_load_kv、
wait_for_layer_load、save_kv_layer、wait_for_save、get_finished、
register_kv_caches、requires_piecewise_for_cudagraph。
内部新请求登记方法因此改名 `_track_new_request`：`on_new_request`
撞基类 hook 名，第一轮回归曾因此 5 gate 全挂。

--------------------------------------------------------------------
## 1. 模块级常量与 dataclass

### SCHEMA_VERSION = 4
页面布局版本的输入进 namespace：布局不兼容变更时 bump 它，namespace
随之改名，旧页自动不可达——靠"改地址"失效旧数据，而不是在新布局下
读旧数据。

### ReqMeta
worker 迭代的单位：一步内一个请求的全部搬运指令。
- `group_ops`: 每 group 一个 GroupTransferMeta。杂交模型的 attention
  页与 mamba 快照独立搬运，所以天然按组拆开。
- `external_hit_tokens`: 核心接受的 external token 数。用途有二：
  日志证据；fail-closed 叙事的一环（resumed 请求带着 accepted tokens
  却零 ops 必须报错而不是静默重算）。
- `is_async` / `async_load_layers`: 异步加载的生命周期说明，见
  `_decide_async`。

### ReqGroupState
每请求每组的可变状态：block_ids（组 GPU 块表的本地副本，两条同步通道，
见 `on_cached_request`）与 next_stored_chunk_idx（增量保存游标，
rollback 语义见 `on_cached_request`）。

### ReqState
请求级状态束。最关键的字段是 `live_source`：vLLM 只在 waiting 路径调
update_state_after_alloc，running 请求 decode 出的新块没有任何 hook
通知我们，所以保存路径直接引用 Request 身上的活列表
（block_hashes 或 all_token_ids，取决于 hash source；两者上游只 append
不重绑，同 LMCache 的 ConstantList 手法）。保存边界的权威值
snapshot_boundary 在查询时一次钉死，plan 构建永不再查（见
`_build_load_meta_from_state`）。

### RequestMetadata
req_id -> ReqMeta 的薄容器，load/save 各一份随元数据 pickle 下行。
曾有 `add_request` 便利方法：已删除。理由——生产路径全部走
`requests[req_id] = req_meta` 直接赋值，该方法只被单元测试调用；
为测试在生产类上开 API 踩红线，测试改为直接构造 ReqMeta。

### KVShrinkConnectorMetadata
scheduler -> worker 的唯一载荷：reqs_to_load + reqs_to_save。
worker 只看这个对象，所以每个 plan 自描述（组号、键、GPU 块俱全），
不含任何调度侧簿记。

--------------------------------------------------------------------
## 2. 模块级函数

### group_label(namespace, group_idx, rank)
store 键的命名空间：`{ns}_g{i}_r{k}`。三段各自挡住一类互踩：
- namespace：挡跨模型/跨 dtype/跨 tp/跨 schema 读；
- g{idx}：同一个前缀 hash 在每个组都存在， durability 台账按
  (label, chunk_id) 记，两组共 label 会被当成一个寿命单元；
- r{rank}：store 的 rank 参数只管管理端口和日志，不管键，TP 两 rank
  共 label 会互相覆盖分片。
label 组件被 store 校验且禁止分隔符字符——这也是 namespace 用 hex
hash 形式的原因之一。

### lookup_boundary(store, key) -> bool
存在性查询。核心裁决："错的 hit 静默污染输出；错的 miss 只赔一次
重算"——所以任何异常都折算成 MISS（fail closed），这是全文件唯一
故意吞异常的地方。查询只用 key 自己 rank 的 label：controller 进程只
与 rank-0 worker 共享台账，peer ledger 本来就不可查；TP rank 锁步
保存使 rank0 存在即全体存在，个别 diverge 的 rank 会在 load 时被
native 层硬断言爆出来。

### validate_codec_env()
启动期拒绝 IAXL_KV_LOSSY_TRUNC：lossy 截断对 element_size==1 有专门
分支，而杂交页就是 opaque int8——截断掩码会打在 bf16 的指数位上，
attention 近似还能解码，GDN 递归状态被污染则直接吐错 token 且无处报
错。IAXL_KV_DATA_SHUFFLE 故意不拒：字节重排完全可逆；多拦一道只会把
操作员推向关掉整个 connector。

### compute_namespace(...)
稳定命名空间：sha256(model id, revision, tokenizer revision, cache
dtype, SCHEMA_VERSION, tp_size)[:16]。为什么由这些而不是别的东西构成：
它们全部改变页面的字节含义或分片方式。近期改动：
- 删掉恒为 1 的 pp_size 死参。它从没表达过任何信息，反而暗示了并不
  存在的支持。
- 配套加了 PP!=1 的显式拒绝（在 `_init_kv_stack`）：PP 按 layer 切分
  rank，单个 rank 只握半个模型的页，它的页面 key 名义上是一个 block
  实际上只有一半字节——不炸但静默错数据的典型场景。既然不支持且无法
  安全降级，就在创建时 RuntimeError（与 speculative 拒绝同构）。

### _spec_kind(spec)
Mamba -> "mamba"，AttentionSpec 子类 -> "attention"。sliding window
刻意不分流：它的块布局就是 attention 的，命中规则的差异住在 vLLM 的
spec registry 里，直接查那张表（见 HybridHitPolicy._lookup）。未知
spec 抛 KVShrinkParseError，绝不猜。

### _iter_layer_specs(group_spec)
展开 UniformTypeKVCacheSpecs 成 (layer_name, spec) 对。纯展开器，
让 parse 主循环不用双层 if。

### parse_kv_cache_config(kv_cache_config)
把 vLLM 的 KVCacheConfig 解析成 (groups, layer_infos, num_blocks)。
fail-closed 规则链，每条都有明确的"为什么不猜"：
- 组内混两种 spec kind -> 拒绝：一组是一次引擎调用，要求视图同形；
- 组内层间块尺寸不一致 -> 拒绝：hash i 定址组内第 i 块，粒度不一致
  无法一一对应；
- 非 align 的 mamba_cache_mode -> 拒绝并给出启动参数指引：非 align
  模式下一个请求占一个 max_model_len 大块，根本没有可按边界寻址的
  快照槽位；vLLM 关 prefix caching 时悄悄改写 mode 为 none，而杂交
  模型默认关 prefix caching，所以这是最常见的误配；
- 全局块尺寸必须归一 -> 拒绝混合尺寸：一个 block hash 要同时给所有
  组定址第 i 块，这是 vLLM resolve 出 GCD 尺寸的推论，模型一旦违反，
  对应关系只在其中一个组成立——静默错数据。

### save_enabled()
KVSHRINK_SAVE=0 关、KVSHRINK_DEBUG_AUTOSAVE=1 强开。默认开：保存就
是产品的写入面。

### storage_size_bytes(t)
descriptor 与底层 storage 的字节数核对用。服务 Canonicalizer.register
里的越界预检：as_strided 读越界不会炸，会静默读到邻近块的垃圾字节
——划线原则里明确保留的"不炸但错数据"类。list 入口取首 tensor 的
storage（mamba 双 tensor 共一存储）。

### make_boundary_key(...) / CacheKey
CacheKey 是页面或边界的逻辑地址，layer_name=="" 表示边界本身。
近期改动：删掉 tp_size 字段。理由——namespace 的 sha256 输入已经含
tp，不同 tp 的部署在 namespace 层就已隔离；key 里再带一份 tp 是恒
常量字段，对地址唯一性贡献为零，却让每个构造点多传一个参数、
boundary_key 元组多一位。身份收敛为 (namespace, rank, hash, group)，
四个真正区分状态的维度。

### KVShrinkParseError(ValueError)
解析期失败的类型标签。"解析永不猜测"的可定位锚点。

--------------------------------------------------------------------
## 3. 小类

### Canonicalizer
存在的根本原因一句话：传输引擎要求一次调用的所有张量同形同 dtype，
而 GDN 层天然是一份存储上两个不同形的张量（conv/ssm）。统一成每层一
个 (num_blocks, page_bytes) int8 视图后才可能发起合法调用，同时
chunk_dim 对一切布局坍缩为 0——这也是 main 分支后来删掉引擎
chunk_dim/block_dim 参数的方向一致之处。
- `register`：为每层构建视图。mamba 取首 tensor 的 untyped_storage
  （vLLM 用 as_strided 从偏移 0 起铺，首 tensor 的存储就是整个状态
  池）；attention 直接包 storage。split-K/V 探测命中时保留
  (k_view, v_view) 半页对——num_blocks 活在非首物理维时，平铺
  stride=page 的视图会 stride 过相邻 K 块而非本块的 K+V，是不炸但
  静默搬错字节的场景，必须分流。（早期注释声称 FA2 布局是 [2,N,...]，
  v0.23 实际是 N-first，注释已修正；探测算法本身与布局顺序无关。）
- `_is_split_kv_layout`：与 vLLM offloading worker 同款
  stride 排序判定。静态方法+纯谓词，好测。
- `page_view_parts`：对引擎 put/get 的唯一接口。"layer#part" 平铺键
  让 split 层贡献两行。生产里 load/save 全走它；曾经与之并列的
  `get_page`/`_page_parts` 已删——生产唯一调用者是 env 门控的
  debug_dump_state，dump 改为直取 page_view_parts 的视图行，省掉每页
  torch.cat 的拷贝，也让调试看到的数据与引擎真正搬动的字节同源。
  单测相应改为对 page_view_parts 断言。

### LayerPageInfo
冻结快照：(num_blocks, page_size_bytes)。KVCacheTensor 只有
(size, shared_by)，所以页必连续（stride==page、offset 0）——这个事实
写死在这里。

### GroupInfo
组的冻结快照。`kind` 是行为分派点；`spec` 保存 vLLM 原始 spec，让命中
策略能把问题原样交还 vLLM 自己的匹配代码。frozen dataclass：组在注册
后是事实，不是状态。

### GroupTransferMeta
一组的一次搬运指令：keys[i] <-> gpu_block_ids[i] 平行配对。`kind`
刻意不带：worker 从自己注册的 GroupInfo 派生 kind，保持单一事实源，
序列化体积也小。GDN load 没有 slot 字段：v0.23 的执行时序决定只写
CURR，没有选择可言（详见 _build_load_meta_from_state 注释）。

### _StoreAsBlockPool
适配器的教科书案例：vLLM 的 find_longest_cache_hit 问的是块池
"这个 hash 缓存了吗"，把这个问题指向外部 store 就是全部适配工作；
匹配规则一行不抄上游。hit 返回 null_block 占位符——调用者只数数和
定位，真块随后由 vLLM 自己分配。

### HybridHitPolicy
定点迭代的多组合命中检测。
- 每组独立查询：A 组命中不代表 B 组有任何东西，它们的 label 不同、
  寿命不同。
- `_lookup` 直接调 KVCacheSpecRegistry 出的管理器的
  find_longest_cache_hit：attention 组是下行闭包前缀扫描，递归组是
  右往左找最近对齐快照，还有 EAGLE 与异粒度的各家处理——复刻它会随
  上游演进而悄然漂移，唯一替换是"缓存与否"从 GPU 池换成外部店。
- 构造时把 attention 组排前面（初始界更紧）、取全局最小 mamba 对齐。
- 收尾减一（最后一个 prompt token 恒重算，logprobs+state 需要）。

### _AsyncLoad
一个异步请求的跨步载入。`gate_layers` 必含全部 recurrent 层：递归状
态在 forward 一开始就被整体消费，不存在"放行一个状态还在路上的请
求"；attention 层可以按前缀放行，因为每层在自己的 kernel 前有专属
hook 等待。released 分两级退出：先经 get_finished 上报释放，余下楼
层交给 forward 中的逐层 hook。

### _SaveCandidate
步内跨请求的去重容器：同一 boundary 多请求贡献同一批页（decode 并行
常见），按 boundary_key 合并；req_ids 记录每个贡献者——put 正在读他
们的 GPU 块，这些人块的释放都得推迟到写落地（延迟释放契约的记账侧）。

--------------------------------------------------------------------
## 4. Connector 构造

### requires_piecewise_for_cudagraph -> True
vLLM 对每个 attention 层入口调 wait_for_layer_load、出口调
save_kv_layer，这套逐层 hook 依赖 piecewise cudagraph。开关就是为此
而生的诚实回答。

### __init__
近期大改：删除全部"占位再覆盖"的双写字段。原版把 scheduler 态
（_namespace/_block_hash_source/...）和 worker 态（_canon/_labels/
...）都先塞零值占位再由 _init_kv_stack 覆盖；问题是那套占位服务的唯
一场景是"构造时没拿到 kv_cache_config"的退化实例，而那种实例失败得
越晚越难查——占位让它在第一次用到某字段时才 AttributeError，根因离
现场隔了一个栈。现在每个字段只有一个赋值点：规划侧由 _init_kv_stack
一次写齐，worker 的序敏感集合在 register 写齐，每步重置的在
start_load 开头写齐。缺 config 构造立刻在最短路径上自然炸。
保留下来的初始化是另一回事：
- 跨步增量容器（_req_states/_async_load_pending/_async_loads/
  _current_put_tasks/_deferred_finished_req_ids）：语义就是"从空开始
  积累"，empty init 不是占位而是定义；
- kvstore=None：角色守卫的判据（main 风格），不是字段初始化习惯。

### _init_kv_stack(vllm_config, role, kv_cache_config)
全部分支共享的前置链，顺序即依赖序：
1. PP!=1 拒绝（新增，见 compute_namespace 条目）;
2. rank 选择：worker 用 parallel_config.rank——connector 构造早于
   distributed init，world group 在所有 TP rank 上都报 0，用它会让
   两个 rank 都自称 rank0 互踩管理端口与分片；scheduler 侧恒 0；
3. hash_block_size = 组块尺寸 GCD（对齐 v0.23 resolve 的推论）;
4. namespace 计算 + codec 校验;
5. 解析 config、speculative 拒绝（speculative 扩宽 GDN gather 读到
   未恢复槽位，无法安全降级，fail-stop）;
6. role 分派：scheduler 建只读 store（presence only）；worker 建
   Canonicalizer 与 labels，真正的 writer store 延迟到
   register_kv_caches——store 需要 kv_caches，而那时才有。

### _choose_block_hash_source(recurrent)
数据兼容开关而非行为开关：vllm/legacy 两方案产出不同的键值，切换=
全部既有条目变冷，而不是变脏。auto 语义按既成事实各守其源：boundary
布局一直用 vLLM hash 发布，block 布局一直用 token 重导——保留各自
写入时的方案意味着升级不清空旧缓存。操作员可用 env 强行统一。

### _bind_cpu_affinity / _bind_intel_accel
main parity：CPU 部署的亲和绑定与 QAT/DSA 设备按 rank 切分。保持与
main 逐行一致的实现，评审时无需对照差异。

--------------------------------------------------------------------
## 5. Scheduler Side Methods

横幅下的总注释给出了四步触发图（lookup -> alloc 记录 ->
build_connector_meta 四件事 -> teardown）；下面逐个说 WHY。

### _track_new_request(req_id, hashes, computed, request=None)
内部登记入口，三个钩子（查询/记录/plan 构建）首次见到请求都会调它。
live_source 在这里一次性挂上 Request 的活列表引用。

### take_async_load_plans(already_emitted)
异步请求的 plan 唯一下行口，drained exactly once。双重提交=对已在途
的请求二次搬运、对已运行的请求重复上报完成，vLLM 会 assert。发出的
同时把 is_async 降回 False：请求被释放后将走普通 new-request 路径再
来一遍，不复位会在那一步再次开异步（观察到的 5-gate 全挂事故根源）。

### _request_block_hashes(request)
hash source 二选一的读侧：vllm 直接 adopt 引擎前缀 hash；legacy 从
token 重导（generate_block_hashs）。两者都掐尾（最后 token 未完成，
vLLM 同款约定）。

### on_request_finished(req_id)
摘除 ReqState。内容寻址的边界不受影响——那是它们的生存意义。

### on_cached_request(req_id, new_block_ids, resumed, nct)
running 请求的两条同步通道之一（另一条是 update_state_after_alloc）：
new_block_ids 追加（resume 时整体替换，upstream CachedRequestData 语
义，含空列表清 stale）。顺带吸收 live_source 新增的 hash——decode
完成的块要靠这条路径进入保存视野。
游标回滚是本方法的灵魂：next_stored_chunk_idx 的含义是"本生命周期内
已被证明无需重发"，resume 或任何进度倒退都将其回卷到 floor(N/bs)。
宁可重发（幂等）不可漏发（漏发的边界永久丢失，且无人发现）。

### get_num_new_matched_tokens(request, computed) -> (int, bool)
返回值收窄自基类的 int|None 到 int：我们的查找是 Record-gated 同步查
找，从不延迟，None 语义不可能出现——诚实的签名比宽容的签名更容易
review。流程：登记请求 -> 定点命中检测 -> 把权威恢复点写进
snapshot_boundary -> 决定是否异步。

### _decide_async(req_id, external)
并发近似=存活请求状态数（与 block 路径共用一把尺，一个旋钮一个含
义）。clamp 逻辑防两个死锁：选层数超过存在层数 => 永不可满足 =>
请求卡 WAITING_FOR_REMOTE_KVS 到天荒地老，所以超界折叠成 -1（等全
部）。早期版本的 except Exception 静默降级已删——配置错了就让它炸。

### update_state_after_alloc(request, blocks, num_external_tokens)
只记录、不建 plan 的三条设计理由（写在 docstring 里）：
1. 恢复点是查询时钉死的边界，事后除法不一定落回合法边界；
2. 多组块表各自组装规则不同（attention 逐块、mamba 最后非空槽），
   在此组装会把规则复制一份；
3. 新请求与 resume 请求共用同一个 plan builder（都是
   _build_load_meta_from_state），在此建 plan 会裂出第二条路径。
近期删掉了循环里的越界 continue：all_block_ids 与 _groups 同源于同
一份 kv_cache_config，长度恒等；而且该分支不是 raise 是 skip，真发生
时会静默采纳半张块表——比崩溃糟得多的失败模式。

### request_finished / request_finished_all_groups
都返回 (True, None)：块释放推迟到 get_finished。理由链：异步 put 正
在读刚算完的块；请求死亡->立即回收->块被新请求覆写->put 把垃圾写给
store，全链路无声。HMA 默认走 all_groups 版本，两个入口共享同一契
约（main 同款形态）。

### build_load_meta(new_req, scheduled_tokens)
薄入口，转 _build_load_meta_from_state。

### build_resumed_load_meta(req_id, scheduled_tokens)
resume 请求独走此门：v0.23 把他们放在 scheduled_cached_reqs.
resumed_req_ids 而非 scheduled_new_reqs，漏掉=预占后垃圾输出。
fail-closed 断言：accepted external tokens > 0 而 plan 空 => RuntimeError。
宁可 EngineCore fatal，不进 forward 读未恢复页。

### _build_load_meta_from_state(req_id, state, scheduled_tokens)
三条不变量都在注释里反复强调：
1. snapshot_boundary 是唯一权威，绝不重新查找（update_state 之后本地
   计数已含 external，重新查会被污染）;
2. attention 组：连续前缀下探到第一个 MISS 为止;每层一层页键全展开;
3. mamba 组：单槽写入 CURR。为何恰好这一个槽：align 模式内核不扫表，
   mamba_get_block_table_tensor 起点减一后 gather 单列、attn 取列 0，
   所以 curr_idx=(computed+scheduled-1)//bs 是 forward 唯一会读的地
   址；上一代曾双写 prev/curr，是 v0.21 时代对 preprocess_mamba 时序
   不明的心虚备份，v0.23 源码坐实后 prev 写属 dead work 被砍。
   curr 槽合法性校验保留（不炸但读到未恢复状态 => 错 token），
   scheduled_tokens<=0 且同步 => fail-stop（那条路径上 start_load_kv
   不运行，槽永远补不上，而 core 已记功 token）。异步例外有长论证：
   park 期间 sched=0 给出的 idx 恰好等于最终被 preprocess_mamba 读作
   prev 的下标，自身拷贝链路会用它。

### build_save_meta(req_id, scheduled_tokens)
预测式 plan（forward 前 build，描述 forward 后的状态）：progress =
computed + scheduled。
- attention：[cursor, progress//gran) 区间内每个已完成块 x 每层，
  发完推进 cursor；
- mamba：progress 恰落边界才发，取表中最后一个非空槽（有整段注释讨
  论 null 前缀表型），idx 落窗才推进 cursor。
游标在 emit 时即推进——worker 保存是 fail-stop，索引不会分歧；
代价是 worker 静默丢保存不可能存在，所以绝不宽容。

### build_connector_meta(scheduler_output)
顺序敏感处两条：cached 表同步必须先于 save plan（本 pass 新分配的块
当步即可保存）；三路 load 合流（new / async-pending / resumed）。
debug 日志集中在这一处（KVSHRINK_DEBUG_LOG 门控）。

### _boundary_key / _present / _page_key
sched 侧寻址三角：组+hash -> 边界键；边界键喂给 presence 谓词供命中
策略用；边界键 replace 出每层的页键。全部排在 rank=_rank 上
（sched 恒 0），worker 收到后 _worker_key 重映射到自己——一把钥匙的
两侧刻痕。

--------------------------------------------------------------------
## 6. Worker Side Methods

横幅注释给出三大机制：载入流水线（GDN 无 hook 所以开头阻塞一段,
attention 靠逐层 hook 天然流水）、保存流水线（submit ASAP &
get_finished 收割）、CURR 槽安全性论证。

### register_kv_caches(kv_caches)
入口三件事：FlashInfer 拒绝（gather 语义差异，无法安全支持）、
register() 两层委托、writer store 建立。
- 曾经的 `if self._canon is not None:` 守卫已删。事实链：_canon 只在
  worker 分支赋值；此 hook 只在 worker 被调；worker 构造必带
  kv_cache_config 因此 _canon 必非 None。守卫两侧永不分叉，它保护的
  失败模式却是"静默 return 假装注册成功"——比 AttributeError 更糟
  的失败。main 风格的 kvstore-is-None 角色守卫在这不适用：这正是创
  建 kvstore 的地方。
- 视图展开放手给 _flat_views（原来手抄了一份同款字典推导，已收敛）。

### register(kv_caches, execution_order)
v0.23 的 kv_caches 字典不携带执行序（按组分批灌入），从层名的 index
恢复排序（vLLM bind_kv_cache 同款手法）。在此一并派生三件序敏感的东
西：_mamba_layers（载入屏障的名单）、_attn_order（异步释放门的"前 N
层"坐标轴）、_mamba_save_segments（保存流水线的挂点表：attention i
提交其前导 mamba 段——那些层的 kernel 必然已跑完，trailing 段留给
wait_for_save）。段内按组再分是回归揪出的真 bug：执行序穿插三个
mamba 组，一段横跨三组时若用首段的组提交，g1/g2 的边界从未落账，
全量 MISS。

### _metadata()
取元数据 + 类型断言。main 同款形态（main 版两处一模一样的
isinstance-raise）。

### _worker_key(key)
sched 以 rank0 建键，worker 重映射到自己 rank：各 TP rank 持久化自己
的分片，不重映射则 TP>1 时互相覆盖。

### _flat_views(layer_views)
"layer#part" 平铺。唯一的真理来源，register_kv_caches 与载入/保存提
交共用。

### _wait_load(tasks)
host-block 等读。传输不完整 => RuntimeError：forward 即将读这些块。

### start_load_kv / start_load(metadata)
提交全部载入后仅在 GDN 上 host-block：GDN 层没有 vLLM hook，不在开
头等到就没人在 forward 里等它们。换来的是零"哪层负责哪个 GDN 层"的
决策机构、零未等待风险；代价只是几块几十 MB 的重叠机会，尺寸上与一
次 forward 差一个数量级。每步开头三连重置（_load_tasks/_saved_layers/
_step_save_pages）就发生在这里，注释说明了各字段的用途。
async 请求的任务单独成册（不能被 GDN 屏障误伤——它们不进本次
forward），sync 请求按组合并成每组一次引擎调用（同组同形，一次调用
原子性也顺带保证：两组重复 hash 没问题，引擎按 (label,index) 对独立
搬运）。

### _register_async_load(req_id, meta, tasks)
release gate 计算处：recurrent 全收 + attention 前 N。N<0 或超界 => 全
层（见 _decide_async 的 clamp 讨论）。

### poll_finished_loads()
非阻塞轮询门层任务，齐了才上报 finished——这是请求解除 park 的唯一
途径。运行在别人请求的 step 里，blocking 会把 async 要消灭的停顿原样
复活。

### wait_for_layer_load / wait_layer_load(name)
vLLM 钩子与本体的薄分离保留了 main 命名习惯（wait_for_layer_load 是
基类钩子名）。本体排水两路：sync 字典里该层的任务；以及每一个已
released 的 async 册子中该层的余款。刻意"排干所有 released 而非仅本
batch 的"：worker 不知道 step 覆盖哪些请求，等早一步只是白等一瞬，
不等人则是未恢复内存。

### _op_entries(op)
(keys,gpu_ids) 平行组 -> [(gpu,label)] 去重折叠：键按层展开是 scheduler
不变量，执行时按 label 折叠正好还原引擎视角。

### _submit_group_load(_entries)
一次 get 提交一组的全部块。_load 薄转发保留是有意的命名封装（它与
save 侧的 _submit_group_layers_save 对仗），收益微小但对称性好。

### _gather_save_candidates(metadata)
跨请求去重的归并：boundary_key -> candidate(pages + req_ids)。保存侧
所有提交者的共同前置。

### _submit_group_layers_save(g_idx, layers, entries)
一次异步 put：D2H+zip 走引擎 put 流，自带 compute-stream 门控（读取
时数据必然已 final——GDN boundary slot 由所有者写过即不再触碰，另见
wait_for_save 注释里的三段安全论证）。

### _track_put(tasks, req_ids)
把 put 挂到每个贡献请求名下：get_finished 据此推迟他们的块释放。
延迟释放契约的记账动作。

### _submit_layers_save(g_idx, layers, metadata)
pipelined 路径的组内提交者。partial boundary（层集不满）跳过：保存
原子性=一组一次写齐，半截边界不算 commit，其内容反正会作为 MISS 老
化出局（cursor 已前进，绝不会错误读用——这是全文件唯一不 fail-stop
的保存侧异常路径，且有无条件 warning）。

### save_kv_layer(layer_name, kv, attn_metadata, **kw)
vLLM 在每个 attention 层出口调用的钩子。三道快速短路（无元数据/流水
关/保存关）后：提交前导 mamba 段（按组拆）+ 本层自身。只提交不等
待——等待发生在 get_finished，这就是"保存搭计算便车"的全部实现。

### wait_for_save()
步骤尾部兜底：未被 pipelined 收走的层（trailing mamba 段、没挂上钩
子的 attention 层）从这里补发。保存开着而本步什么都没存时也有一行
counterpart 日志——健康与"啥都没干"在日志上可区分（这条与
start_load_kv 的日志互为镜像，是 Gate 断言的锚点）。

### submit_saves(metadata)
上面说的补发本体：候选按组聚拢，满边界入列，剩余未保存层一次 put。
返回 (pages, boundaries) 供日志/测试。"写即 commit"在这里有一段专
注注释：没有第二发布阶段，所以没有能比数据活得久的悬垂物。

### get_finished(finished_req_ids)
三段式收割器（与 main 的生命周期同形）：
1. poll loads —— 解除 async 停泊;
2. deferred finished 合账 —— 外面通报的死请求进来排队;
3. 逐请求 drain：有余款的 async 载入先等完（它的层钩子不会再响
   了——请求死了）;然后 drain 名下 puts。共享 put 的双重 drain 修复
   在此：跨请求 dedupe 使同一 tasks dict 挂多个名字，第一个 drain 完
   成后 put_wait 已把 ctx 清 None，第二个撞上来会触发 native 断言杀
   死 worker——所以先查 ctx 全 None 则直接弹出（那笔账已被别人结清）。
完成的上报名字：completed -> finished_sending，poll 结果 ->
finished_recving。

### debug_dump_state()
KVSHRINK_DEBUG_DUMP=1 时把每个 mamba 组 gpu block 0..9 首层页的
sha256 打出来，供冷热 GPU 状态逐字节对比。env 门控、零开销短路，
属运营面工具所以保留；改成直取 page_view_parts 视图行也是本轮清理的
一部分（原来依赖已删除的 get_page，每页还要 cat 拷贝一次）。

--------------------------------------------------------------------
## 7. 近两轮删改对照（apple-to-apple 讲解锚点）

| 改动 | 类型 | 理由 |
|---|---|---|
| `if self._canon is not None:` | 删 | 恒真的死守卫；它防的不是崩溃是"静默假装成功" |
| `_now()` 包装 | 删 | 局部延迟 import 绕路；main 无此函数；两调用点直呼 time.monotonic |
| RequestMetadata.add_request | 删 | 生产零调用、测试专用 API |
| Canonicalizer.get_page/_page_parts | 删 | 生产唯一读者是 debug dump；统一走 page_view_parts |
| update_state_after_alloc 越界 continue | 删 | 不可达且是 skip 型（半张块表静默采纳） |
| register_kv_caches 手抄视图推导 | 改 | 复用 _flat_views，消灭平行实现 |
| register_kv_caches->_register_layer_caches->register | 并 | 中间层纯转发，两层足够 |
| split-K/V 布局注释 [2,N,...] | 改 | v0.23 实际 N-first；探测算法本身不受影响 |
| compute_namespace 的 pp_size | 删 | 恒 1 死参；配套加 PP!=1 显式拒绝（错数据防线） |
| CacheKey.tp_size 字段 | 删 | tp 已编入 namespace hash；恒常量字段无身份贡献 |
| __init__ 占位双写 | 删 | 单一赋值点；退化实例第一时间自然炸 |

每项共同的评价标准回到第 0 节的三条红线：不减少任何一道"不炸但错
数据"的防线，不多留一行只为图省事的代码。
