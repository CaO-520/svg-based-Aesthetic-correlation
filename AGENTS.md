# AGENTS.md

## 云端同步

- “从云端下载”：若本地干净，执行 `git fetch origin` 和 `git pull --ff-only origin main`；若本地有改动，先提醒用户。
- “上传到云端”：以本地为准；只看 Git 状态和文件列表，不读文件内容。
- 除 `configs/API_KEY.txt` 和`configs/system_path.yaml` 外，所有代码、配置、数据、日志、输出、文档、脚本，以及删除/移动/重命名都要同步。
- 永远不提交、不上传 `configs/API_KEY.txt`，`configs/system_path.yaml` ；推送前确认它未暂存。
- 除了上述不上传的文件外，强制覆盖，不检查更新，除非用户明确要求检查



## 代码路径

- 新建或是修改代码时，里面的路径，在本项目内的文件，一定要使用相对路径，本项目外的文件（例如模型），将系统路径更新到`configs/system_path.yaml` 里面，然后通过导入配置文件的方式获取系统路径
