# miyako-life

**离线角色包产线**：galgame 解包原料 → 结构化角色包 `dist/{package_id}.char/`，
拷入桌面陪伴应用（miyako-pet）的 `data/characters/` + 一行 settings 即可运行。
当前实现：**《9-nine-》EP1 · 九條都**（`v2.0.0`，voice_tier=2 含音色权重；第二游戏泛化推迟）。

```text
游戏《9-nine-》EP1 ──解包──▶ 原料区 ──miyako-life──▶ miyako.char
                                                          │
                                  拷入桌宠 data/characters/ + 一行 settings
                                                          ▼
                                  桌面陪伴：聊天 / 9 种情绪立绘 / GPT-SoVITS 音色
```

四个 CLI 入口构成完整生命周期：
`build`（DAG 全跑）→ `report`（人工决策页）→ `pack`（出厂打包）→ `accept`（深路径验收，规划中）。

## 特性

- **hash 冻结链**——voice_map.json 单一事实源 sha256 上链，上游一变下游全部失效重算，绝不静默沿用旧结论；
- **统计归脚本、归纳归 LLM**——所有频次/比例由确定性脚本产出并过黄金值断言；LLM 只做带证据引用的模式归纳；
- **四工序模块隔离**——upstream / persona / voice / pack 互不 import，只经 `contracts/` 通信（CI 执法）；
- **样本组缓存与断点续跑**——LLM 归纳按样本组缓存（内容指纹），`--resume` 只补缺口，不重复烧 API 钱；
- **P3 逐字引用门**——语气示例必须逐字对回语料（脚本校验 + refine 强制重抄），杜绝 LLM 漂移引用；
- **13 项出厂检查（fail-closed）**——schema 全件 / 情绪闭集互锁 / 弧线两端保底 / prompt 预算 / 决策账本无 stale……任一不过不出包；
- **Y5 决策账本**——每条人工决策绑定所审产物 sha256，产物重跑即 stale 重问；
- **全链路日志**——`work/logs/pipeline.log` 单文件，11 工序全埋点，排障一个文件搞定；
- **两层测试**——CI 层断结构（任何环境全绿）+ 本机层断黄金值（原料在位时跑）。

## 快速开始

环境要求：Windows + Python 3.10（建议 conda 环境）+ 智谱 GLM API Key（BYOK，放 `.env`）+ 合法获得的《9-nine-》EP1 解包原料。

```bat
:: 环境依赖（纯 wheel）：pip install -r requirements-life.txt

:: 看工序计划（不执行）
python -s build.py --dry-run

:: 跑全工厂（断点续跑；--profile 指定立绘生产源）
python -s build.py --raw "E:\game\gal\tool\miyako" --profile work\profile_e2_override.yaml --resume

:: 人工决策页（浏览器打开 work\report.html 完成词表勾选 / 参考音试听 / 盲测）

:: 出厂打包（13 项检查全绿才出包）
python -s pack\packager.py --raw "E:\game\gal\tool\miyako" --profile work\profile_e2_override.yaml
```

## 架构一览

```text
upstream/    E0 输入验证 → E1.5 翻译（补缺）→ E1 语义提取（冻结链）→ E2 立绘验收
persona/     E3 表达 DNA 统计（零 LLM）→ E4 五层提炼（LLM 归纳）→ E6 自动验证（护栏）
voice/       V1 音频工程（dataset 1466 + 九桶预录）→ V2 参考音频初筛
pack/        report_builder（报告页 + 决策账本）→ packager + factory_check（13 项出厂闸门）
contracts/   唯一共享层：情绪闭集 / JSON Schema / 写权限 / 成本计量 / LLM 客户端 / 日志
```

数据漏斗实测（EP1）：台本 2103 → 去重 2098 → P 层池 2049 → **dataset 1466** → audiofile 159 → 参考候选 10；立绘 166 PNG（闭集三元组 156 + extras 6 + accessories 6）；现役卡指纹 `dde7cd7f`。

## 文档

- [docs/产品介绍.md](docs/产品介绍.md) —— **各模块实现细节**（设计原则、架构、四工序逐模块、出厂检查清单、成本结构、路线图）
- 仓库不含测试与运行时数据；本仓为源码快照，与代码不一致时以代码为准。

## 免责声明

- 本项目是**游戏衍生粉丝工具**，仅供学习研究；不含、不分发任何游戏资产（台本/语音/立绘均需用户自备合法获得的原料，且全部目录不入 git）。
- 生成的角色包**禁止二次分发**；游戏原作版权归原权利方所有。
- API Key（`.env`）为 BYOK，不入库、不外传。
