# Hermes Agent 详细文档

> 本文介绍 **Hermes Agent** 本身——由 Nous Research 开发的自进化自主 AI Agent，而非 ms-swift 中的训练模板。ms-swift 中的 `HermesAgentTemplate` 仅是借鉴了 Hermes 模型的 function calling 格式。

---

## 目录

1. [Hermes Agent 是什么](#1-hermes-agent-是什么)
2. [与相关项目的关系](#2-与相关项目的关系)
3. [核心特性](#3-核心特性)
4. [安装](#4-安装)
5. [快速上手](#5-快速上手)
6. [CLI 使用](#6-cli-使用)
7. [消息网关（多平台接入）](#7-消息网关多平台接入)
8. [工具系统](#8-工具系统)
9. [技能系统（Skills）](#9-技能系统skills)
10. [记忆系统（Memory）](#10-记忆系统memory)
11. [定时任务（Cron）](#11-定时任务cron)
12. [MCP 集成](#12-mcp-集成)
13. [模型配置与切换](#13-模型配置与切换)
14. [部署方式](#14-部署方式)
15. [安全机制](#15-安全机制)
16. [参考链接](#16-参考链接)

---

## 1. Hermes Agent 是什么

**Hermes Agent** 是由 [Nous Research](https://nousresearch.com/) 开发的开源自主 AI Agent，于 2026 年 2 月在 GitHub 开源。

它不是 IDE 里的代码助手，也不是单一 API 的聊天包装器，而是一个**真正的自主智能体**：

- **自进化（Self-Improving）**：能从任务执行中自动创建、优化和复用技能（Skill）
- **持久记忆**：跨会话保留记忆，越用越懂你
- **多平台接入**：同一个 Agent 可通过 Telegram、Discord、Slack、WhatsApp 等 15+ 平台交互
- **随处运行**：本地笔记本、5 美元/月的 VPS、GPU 集群，或 Daytona/Modal 等 Serverless 环境

> **口号**：*"The self-improving AI agent"* —— 与你共同成长的智能体。

**项目地址**：https://github.com/NousResearch/hermes-agent  
**官方文档**：https://hermes-agent.nousresearch.com/docs  
**License**：MIT

---

## 2. 与相关项目的关系

容易混淆的三个概念，务必分清：

| 项目 | 性质 | 关系 |
|------|------|------|
| **Hermes Agent** | 自主 AI Agent 应用/框架 | 本文主角，由 Nous Research 开发 |
| **Hermes 模型系列** | 微调大语言模型 | Nous Research 基于 Llama/Mistral 微调的模型（Hermes 2 Pro、Hermes 3 等），擅长 function calling 和结构化输出 |
| **Hermes-Function-Calling** | 推理代码库 | Nous Research 的配套仓库，提供 Hermes Pro 模型进行 function calling 推理的示例代码 |
| **ms-swift 的 `HermesAgentTemplate`** | 训练模板 | ms-swift 框架借鉴了 Hermes 模型的 `<tool_call>` XML 格式，用于模型训练时的数据格式化 |

**简单说**：Hermes Agent 是"用 Agent 的人"，Hermes 模型是"Agent 的大脑"，Hermes-Function-Calling 是"教大脑怎么调用工具的示例代码"。

---

## 3. 核心特性

| 特性 | 说明 |
|------|------|
| **闭环学习（Closed Learning Loop）** | Agent 自动整理记忆、周期性自我提醒、自主创建技能、技能在使用过程中自我改进 |
| **持久化记忆** | 基于 FTS5 全文搜索 + LLM 摘要，支持跨会话精准回溯；包含用户画像建模（Honcho dialectic） |
| **技能系统（Skills）** | 复杂任务执行后自动抽象为可复用 Skill（Markdown 格式，兼容 agentskills.io 开放标准） |
| **多平台消息网关** | 单一 gateway 进程支持 CLI、Telegram、Discord、Slack、WhatsApp、Signal、飞书、钉钉、Teams 等 15+ 平台 |
| **定时任务（Cron）** | 内置 cron 调度器，可用自然语言设置定时任务，结果投递到任意平台 |
| **子代理并行（Delegation）** | 可派生隔离的子代理并行处理不同工作流 |
| **代码执行** | 通过 `execute_code` 运行 Python 脚本，脚本可直接通过 RPC 调用 Hermes 工具 |
| **MCP 支持** | 兼容 Model Context Protocol，可连接任意 MCP 服务器扩展工具能力 |
| **浏览器自动化** | 网页搜索、内容提取、浏览、视觉分析 |
| **语音模式** | CLI、Telegram、Discord 支持实时语音交互 |
| **RL 训练支持** | 内置批量轨迹生成、Atropos RL 环境，可用于训练下一代 tool-calling 模型 |
| **模型无锁定** | 支持 Nous Portal、OpenRouter（200+ 模型）、OpenAI、Anthropic、GLM、Kimi、MiniMax、Ollama 等 |

---

## 4. 安装

### 4.1 一键安装（推荐）

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

支持平台：Linux、macOS、WSL2、Android（Termux）。Windows 原生不支持，需通过 WSL2 运行。

安装后：
```bash
source ~/.bashrc    # 或 source ~/.zshrc
hermes doctor       # 检查环境
hermes setup        # 交互式初始化向导
```

### 4.2 开发者安装（源码）

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh   # 自动安装 uv、创建 venv、安装依赖、创建符号链接
./hermes            # 直接使用，无需手动 source
```

或手动：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

### 4.3 最低运行配置

- **CPU**：1 核
- **内存**：1 GB RAM
- **系统**：Ubuntu 22.04 / Debian 12 / macOS
- **网络**：能访问模型 API（OpenRouter、OpenAI 等）

---

## 5. 快速上手

### 5.1 首次启动

```bash
hermes              # 启动交互式 CLI，开始对话
```

首次运行会进入设置向导（`hermes setup`），引导你：
1. 选择模型提供商（推荐 OpenRouter，有免费模型可用）
2. 配置 API Key
3. 选择是否注册为系统服务（开机自启、后台常驻）
4. 配置消息平台（可选跳过）

### 5.2 常用命令速查

```bash
hermes              # 启动交互式 TUI
hermes model        # 切换 LLM 提供商和模型
hermes tools        # 配置启用的工具
hermes config set   # 设置单个配置项
hermes gateway      # 启动消息网关（Telegram/Discord 等）
hermes setup        # 完整设置向导
hermes update       # 更新到最新版本
hermes doctor       # 诊断问题
hermes claw migrate # 从 OpenClaw 迁移数据
```

### 5.3 CLI 中的常用斜杠命令

| 命令 | 作用 |
|------|------|
| `/new` / `/reset` | 开启新会话 |
| `/model [provider:model]` | 切换模型 |
| `/personality [name]` | 设置人格 |
| `/retry` / `/undo` | 重试或撤销上一轮 |
| `/compress` | 压缩上下文 |
| `/usage` | 查看用量 |
| `/skills` | 浏览技能 |
| `/<skill-name>` | 调用指定技能 |
| `/stop` | 中断当前工作 |
| `/cron` | 管理定时任务 |

---

## 6. CLI 使用

Hermes 提供一个完整的终端用户界面（TUI）：

- **多行编辑**：支持多行输入
- **斜杠命令自动补全**：输入 `/` 后自动提示可用命令
- **对话历史**：可回溯历史会话
- **中断与重定向**：Ctrl+C 中断当前任务，或发送新消息覆盖
- **流式工具输出**：工具执行结果实时流式显示

### 6.1 配置文件

配置文件位于 `~/.hermes/config.yaml`，可手动编辑或用命令行修改：

```bash
hermes config set model.provider openrouter
hermes config set model.model google/gemma-4-31b-it:free
```

### 6.2 人格与 SOUL.md

Hermes 的默认人格由 `~/.hermes/SOUL.md` 定义，这是一个全局系统提示词文件。你可以直接编辑它来定制 Agent 的说话风格、角色定位和行为准则。

---

## 7. 消息网关（多平台接入）

Hermes 的核心设计之一是"单一 gateway，多平台接入"。你可以同时在多个平台上与同一个 Agent 对话，且对话可跨平台连续。

### 7.1 支持的平台

CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、Email、SMS、钉钉（DingTalk）、飞书（Feishu）、企业微信（WeCom）、BlueBubbles、Home Assistant、Microsoft Teams 等。

### 7.2 启动网关

```bash
hermes gateway setup    # 配置各平台 bot
hermes gateway start    # 启动网关进程
```

配置完成后，你可以在手机上通过 Telegram 发送消息，Agent 在云端 VPS 上处理后回复你。

### 7.3 跨平台对话连续性

Hermes 的会话系统支持跨平台连续性。例如，你在 CLI 上开始的任务，后续可以通过 Telegram 继续跟进，Agent 的记忆和上下文保持一致。

---

## 8. 工具系统

Hermes 内置 **68+ 工具**，覆盖以下类别：

| 类别 | 示例工具 |
|------|----------|
| **文件操作** | 读/写文件、内容搜索、格式转换 |
| **终端执行** | 运行 shell 命令、管理后台进程 |
| **网页搜索** | 搜索引擎、内容提取 |
| **浏览器自动化** | 网页导航、页面交互、视觉分析、表单填写 |
| **图像生成** | 文生图 |
| **语音** | TTS、语音转文字 |
| **代码执行** | 运行 Python 脚本 |

工具配置：
```bash
hermes tools    # 交互式 TUI，启用/禁用各工具
```

---

## 9. 技能系统（Skills）

Skills 是 Hermes 最具特色的功能之一，相当于 Agent 的"程序性记忆"。

### 9.1 什么是 Skill

Skill 是一份 Markdown 文档，描述如何完成特定任务。例如：
- 如何部署 Python 应用到 Fly.io
- 如何进行代码审查
- 如何生成 ASCII 艺术

**Skill vs Memory 的区别**：

| | Skill | Memory |
|--|-------|--------|
| **内容** | 程序性知识——"怎么做" | 事实性知识——"是什么" |
| **加载时机** | 按需加载，仅在相关时引入 | 每轮会话自动注入 |
| **大小** | 可以很大（数百行） | 保持精简（约 2200 字符上限） |
| **创建者** | 用户、Agent 自己、或从 Hub 安装 | Agent 基于对话自动生成 |
| **成本** | 不加载时零 token 消耗 | 每轮有固定的小 token 消耗 |

### 9.2 Skill 的三种来源

1. **内置 Skills**：随仓库自带
2. **Hub Skills**：从 [agentskills.io](https://agentskills.io) 社区安装
3. **Agent 自创建**：完成复杂任务后，Agent 会自动提议"将刚才的流程保存为 Skill"

### 9.3 管理 Skills

```bash
# 浏览所有技能
/skills

# 调用技能
/<skill-name>

# 按平台管理技能（某些平台禁用特定技能）
hermes skills --platform
```

### 9.4 Skill 的自动维护（Curator）

Hermes 内置一个 **Curator（馆长）** 机制，定期审查 Agent 创建的技能：

- **30 天未使用** → 标记为 stale（过期）
- **90 天未使用** → 归档到 `~/.hermes/skills/.archive/`
- **LLM 审查**：Agent 会阅读自己的技能库，决定合并重复技能、打补丁或归档
- **Pinned Skills**：被钉住的技能不会被 curator 处理

你可以随时预览 curator 的行为：
```bash
hermes curator run --dry-run
```

---

## 10. 记忆系统（Memory）

### 10.1 记忆类型

Hermes 的记忆系统分为多个层次：

| 文件 | 用途 | 大小上限 |
|------|------|----------|
| `MEMORY.md` | 环境、偏好、项目位置等事实 | ~2,200 字符 |
| `USER.md` | 用户画像（Honcho dialectic 建模） | ~1,375 字符 |

当记忆达到上限时，Agent 会自动整合和压缩旧条目。

### 10.2 跨会话回溯

Hermes 使用 **FTS5 全文搜索** 索引所有历史对话，配合 LLM 摘要技术，能在新会话中精准回溯相关内容。

### 10.3 主动记忆

你可以主动要求 Agent 记住某事：

```
记住我们项目的 CI 使用 GitHub Actions，工作流文件是 deploy.yml。
```

Agent 会立即写入磁盘，但注意：**记忆在当前会话的 system prompt 中不会实时更新**，需开启新会话后才生效。

---

## 11. 定时任务（Cron）

Hermes 内置 cron 调度器，可用**自然语言**或标准 cron 表达式设置定时任务。

### 11.1 创建任务

**在对话中直接说：**
```
每天早上 9 点，检查 Hacker News 的 AI 新闻，把摘要发到 Telegram。
```

**使用斜杠命令：**
```
/cron add "0 9 * * *" "Summarize AI news from Hacker News"
/cron add "every 2h" "Check server status"
/cron add "30m" "Remind me to check the build"
```

**使用 CLI：**
```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "0 9 * * *" "Summarize AI news" --skill blogwatcher
```

### 11.2 带 Skill 的定时任务

定时任务可以附加一个或多个 Skill：
```
/cron add "0 8 * * *" "Search arXiv for reasoning papers" \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

### 11.3 管理任务

```
/cron list              # 列出所有任务
/cron run <job_id>      # 立即执行一次（测试用）
/cron pause <job_id>    # 暂停
/cron edit <job_id> --schedule "every 4h"
/cron remove <job_id>   # 删除
```

### 11.4 无 Agent 模式

Cron 任务可以配置为 **no-agent mode**——纯脚本执行，零 LLM 参与，仅将脚本的 stdout 投递到指定位置。适合监控、备份等纯自动化场景。

---

## 12. MCP 集成

Hermes 支持 **Model Context Protocol（MCP）**，可安全地连接外部 MCP 服务器扩展工具集。

```bash
# 配置 MCP 服务器
hermes config set mcp.servers.my-server.command "npx -y @modelcontextprotocol/server-filesystem"
hermes config set mcp.servers.my-server.args '["/path/to/allowed/dir"]'
```

连接后，MCP 服务器的工具会出现在 Hermes 的工具列表中，Agent 可以像调用内置工具一样调用它们。

---

## 13. 模型配置与切换

Hermes **不锁定任何模型**，支持 400+ 主流模型，切换只需一条命令。

### 13.1 支持的提供商

- Nous Portal
- OpenRouter（200+ 模型，含免费 tier）
- OpenAI
- Anthropic
- NVIDIA NIM（Nemotron）
- GLM / z.ai
- Kimi / Moonshot
- MiniMax
- Hugging Face
- Ollama（本地模型）
- 任意兼容 OpenAI API 格式的自定义端点

### 13.2 切换模型

```bash
hermes model                    # 交互式选择
hermes model openrouter:google/gemma-4-31b-it:free
```

或在对话中：
```
/model openrouter:anthropic/claude-sonnet-4
```

### 13.3 故障转移

Hermes 支持配置 fallback provider，当主提供商限流或故障时自动切换：

```yaml
# ~/.hermes/config.yaml
fallback_providers:
  - provider: openai
    model: gpt-4o-mini
  - provider: anthropic
    model: claude-3-haiku
```

---

## 14. 部署方式

Hermes 支持 **7 种运行后端**：

| 后端 | 特点 |
|------|------|
| **local** | 本地直接运行 |
| **Docker** | 容器化隔离 |
| **SSH** | 远程 SSH 会话 |
| **Daytona** | Serverless，空闲时休眠，按需唤醒 |
| **Modal** | Serverless，按量计费 |
| **Singularity** | HPC 环境 |
| **Vercel Sandbox** | 云端沙箱 |

### 14.1 VPS 部署（推荐长期运行）

```bash
# 在 $5/月的 VPS 上
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes setup
hermes gateway start    # 后台启动网关
```

### 14.2 Docker 部署

```bash
docker run -d \
  -v ~/.hermes:/root/.hermes \
  -e OPENROUTER_API_KEY=sk-xxx \
  nousresearch/hermes-agent
```

### 14.3 Serverless（低成本）

Daytona 和 Modal 后端支持 **Serverless 持久化**：环境在空闲时休眠，收到消息时唤醒，空闲期间几乎不花钱。

---

## 15. 安全机制

| 机制 | 说明 |
|------|------|
| **命令审批** | 危险操作（如 `rm -rf`）需要用户确认 |
| **DM 配对** | 消息平台 Bot 仅响应已授权用户 |
| **容器隔离** | 工具执行可在 Docker/Singularity 容器中进行 |
| **Credential Pool** | 支持同一提供商多 Key 轮询，应对限流 |
| **Secret 白名单** | API Key 等敏感信息通过安全渠道管理 |

---

## 16. 参考链接

| 资源 | 链接 |
|------|------|
| **GitHub 仓库** | https://github.com/NousResearch/hermes-agent |
| **官方文档** | https://hermes-agent.nousresearch.com/docs |
| **Skills Hub** | https://agentskills.io |
| **Nous Research** | https://nousresearch.com |
| **Hermes 模型系列** | https://huggingface.co/NousResearch |
| **Hermes-Function-Calling** | https://github.com/NousResearch/Hermes-Function-Calling |
| **Discord 社区** | 见 GitHub README |

---

## 附录：与 ms-swift 的关系澄清

ms-swift（即本项目所在目录 `/home/ws/ws/projects/github/ms-swift`）中的 `HermesAgentTemplate` **不是** Hermes Agent 本身，而是一个**训练数据格式化模板**。

它的作用是：当使用 ms-swift 训练模型时，如果指定 `--agent_template hermes`，框架会按照 Nous Research Hermes 模型的 `<tool_call>` XML 格式来组织工具调用数据。这样训练出的模型就能学会生成 Hermes 风格的工具调用输出。

如果你只是想**使用** Hermes Agent，直接去安装 `github.com/NousResearch/hermes-agent`；如果你想**训练**一个支持 Hermes 格式 tool calling 的模型，才需要在 ms-swift 中使用 `--agent_template hermes`。
