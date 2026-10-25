"""9 元情感闭集 —— life 侧唯一事实源。

sync 自 miyako-pet `server/emotion_defs.py`（唯一语义权威，契约 L7：闭集语义不自造）：
来源 commit 1610ea5（2026-09-06，feat: M0.1 全局常量与 9 元情感闭集唯一事实源）。
仓库：E:/my way my life my heart/miyako（pet 侧文件头声明运行期不得修改该列表，本侧同）。

互锁纪律：face_map.yaml / char_schema / E2/E3 的全部情绪取值只允许引用本模块的
SENTIMENT_CLOSE_SET，禁止任何文件出现第二份定义。pet 侧闭集若变更，本文件必须
重新 sync 并在文件头更新来源 commit。
"""

SENTIMENT_CLOSE_SET = [
    "normal",
    "happy",
    "shy",
    "pout",
    "angry",
    "sad",
    "surprise",
    "thinking",
    "sleepy",
]

DEFAULT_EMOTION = "normal"
