# miyako_life

离线角色包产线：galgame 解包原料 → `dist/{package_id}.char/`，拷入 miyako-pet 的
`data/characters/` + 一行 settings 即可运行。当前只做 EP1/九條都（第二游戏泛化推迟）。

四个 CLI 入口：`build`（DAG 全跑）→ `report`（人工决策页）→ `pack`（出厂打包）→ `accept`（深路径 V3 验收）。

> 支持游戏列表、适配指南、数据外发透明度声明：随里程碑推进补全（plan §0.5）。
