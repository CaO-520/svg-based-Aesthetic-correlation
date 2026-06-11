# AGENTS.md

## 云端同步

- “从云端下载”：以 `origin/main` 为准；若本地干净，执行 `git fetch origin` 和 `git pull --ff-only origin main`；若本地有改动，先提醒用户。
- “上传到云端”：以本地为准；只看 Git 状态和文件列表，不读文件内容；提交并推送到 `origin/main`。
- 除 `configs/API_KEY.txt` 外，所有代码、配置、数据、日志、输出、文档、脚本，以及删除/移动/重命名都要同步。
- 永远不提交、不上传 `configs/API_KEY.txt`；推送前确认它未暂存。
- 强制覆盖，不检查更新，除非用户明确要求检查
