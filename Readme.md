# AI视频翻译服务 v3.0

基于 Whisper + NLLB + DeepSeek 的高质量视频翻译解决方案，支持批量处理、上下文润色和断点续传。

## ✨ 主要特性

### 🔥 v3.0 重大更新

- **上下文翻译润色** - 结合前后2句话的上下文，翻译更准确、自然、连贯
- **断点续传** - 自动保存进度，中断后继续，不需要重新开始
- **失败重试** - 自动重试3次，超时/失败自动恢复
- **长视频支持** - 支持3小时长视频，语音识别1小时超时
- **进度管理** - 查看和重置翻译进度

### ⚡ 核心功能

- **高速翻译** - 30分钟视频仅需2分钟（不含润色）
- **高质量** - 翻译质量接近/超越 Google Translate
- **批量处理** - 支持目录批量翻译，自动断点续传
- **GPU加速** - 支持 CUDA GPU 加速
- **人声分离（可选）** - 使用 Demucs 提取人声，改善背景音乐/嘈杂场景识别效果
- **可选润色** - 集成 DeepSeek API 进行专业级翻译润色
- **灵活配置** - 支持多种模型和参数配置

## 📊 性能数据

| 视频时长 | 不含润色 | 含DeepSeek润色 | 质量 |
|---------|---------|---------------|------|
| 30分钟 | 115秒 | 140秒 | ⭐⭐⭐⭐⭐ |
| 1小时 | 230秒 | 280秒 | ⭐⭐⭐⭐⭐ |
| 3小时 | 690秒 | 840秒 | ⭐⭐⭐⭐⭐ |

**成本**（使用DeepSeek润色）：
- 30分钟视频：约 0.001元
- 10元可处理：约 10,000 个30分钟视频

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- CUDA 12.1+ (可选，GPU加速)
- ffmpeg

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

首次运行会自动下载模型（约2-3GB）。

### 4. 配置

复制配置模板并修改：

```bash
copy config.template.ini config.ini
```

编辑 `config.ini`：

```ini
[API]
# DeepSeek API密钥（可选，用于翻译润色）
deepseek_api_key = sk-your-key-here

[Service]
host = 127.0.0.1
port = 50515
# 可选：显存紧张环境按需加载/卸载（如 Colab T4）
lazy_load_models = false
manage_models = false
unload_models_after_tasks = false

[Models]
# ASR模型大小: tiny, base, small, medium, large-v3, reazonspeech
# reazonspeech 为日语优化模型（transformers 后端）；如需自定义，可用 reazonspeech:HF模型名
asr_model_size = reazonspeech
translation_model = gguf:hf:SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF@sakura-7b-qwen2.5-v1.0-iq4xs.gguf
# GGUF 用法：translation_model = gguf:/path/to/model.gguf
use_gpu = true
beam_size = 3

[GPU]
# 可选：在 GPU 重任务前清理 CUDA 缓存（降低 OOM 概率，略慢）
clear_cuda_cache_before_tasks = true

[ASR]
# 可选：指定音频语言（auto=自动检测；填错会变差）
# 备选示例：auto / en / zh / ja / ko / de / fr / es / ru / ar
language = auto

# 可选：分块识别（显示进度条/降低长音频500/OOM；0=禁用）
# 建议从 900（15分钟）起步，不稳定再降到 600/300
chunk_sec = 900
chunk_overlap_sec = 1.0

# 可选：transformers ASR 内部分块（仅对 reazonspeech/transformers 后端生效）
# 默认 30s/5s 可显著降低长音频 OOM/掉线风险；如稳定可调大或设 0 禁用
transformers_chunk_sec = 30
transformers_stride_sec = 5.0

[Translation]
default_target_language = zh
use_deepseek_polish = true  # 启用DeepSeek润色

[Subtitles]
# 字幕格式：srt / ass
format = srt

[Audio]
# 可选：人声分离（Demucs）提升背景音乐/嘈杂场景识别
# 需要额外安装：pip install demucs
enable_vocal_separation = false
vocal_separation_model = htdemucs
vocal_separation_device = cuda  # 如遇 OOM 可改为 cpu
# Demucs 分段秒数（仅对超长/超大音频触发；默认 1800=30分钟）
vocal_separation_chunk_sec = 1800
# 说明：超长音频会自动分段运行 Demucs，避免一次性加载导致 CPU 内存 OOM（exit code=-9）
```

### 5. 启动服务

```bash
# Windows
start_service.bat

# 或手动启动
python server_optimized.py
```

### 6. 翻译视频

```bash
# 翻译单个视频
python batch_translate.py video.mp4 -t zh

# 批量翻译目录（自动断点续传）
python batch_translate.py videos/ -t zh --recursive

# 查看进度
python batch_translate.py videos/ --show-progress
```

## 📖 使用指南

### 基本翻译

```bash
# 翻译单个视频
python batch_translate.py movie.mp4 -t zh

# 批量翻译目录
python batch_translate.py D:\Videos\ -t zh

# 递归处理子目录
python batch_translate.py D:\Videos\ -t zh --recursive

# 指定输出目录
python batch_translate.py videos/ -t zh -o subtitles/
```

### 字幕选项

```bash
# 生成双语字幕（默认）
python batch_translate.py video.mp4 -t zh

# 仅生成译文字幕
python batch_translate.py video.mp4 -t zh --translation-only
```

### 其他语言

```bash
# 中译英
python batch_translate.py chinese_video.mp4 -t en -s zh

# 英译日
python batch_translate.py video.mp4 -t ja

# 支持的语言
# zh(中文), en(英语), ja(日语), ko(韩语), de(德语), 
# fr(法语), es(西班牙语), ru(俄语), ar(阿拉伯语) 等200+语言
```

### 断点续传

```bash
# 批量翻译（自动保存进度）
python batch_translate.py D:\Videos\ -t zh

# 中断后（Ctrl+C），再次运行继续
python batch_translate.py D:\Videos\ -t zh
# ✓ 自动跳过已完成的视频
# ✓ 重试失败的视频（最多3次）

# 查看进度
python batch_translate.py D:\Videos\ --show-progress

# 清除进度，从头开始
python batch_translate.py D:\Videos\ -t zh --reset-progress
```

## 🎨 上下文翻译润色

### 工作原理

v3.0 的润色功能会考虑前后2句话的上下文，使翻译更准确、自然、连贯。

### 效果对比

**传统方式（不带上下文）：**
```
[1] 原文: I can't wait
[1] 机器翻译: 我迫不及待地等着
[1] 润色: 我等不及了
```

**v3.0（带上下文）：**
```
上文: 我们明天去海边。
[1] 原文: I can't wait
[1] 机器翻译: 我迫不及待地等着
[1] 润色: 我真是等不及了
下文: It will be amazing.
```

## 🛠️ 配置说明

### config.ini 参数详解

```ini
[API]
# DeepSeek API密钥（可选）
# 访问 https://platform.deepseek.com 获取
deepseek_api_key = 

[Service]
host = 127.0.0.1
port = 50515

[Models]
# ASR模型大小
# 选项: tiny, base, small, medium, large-v3, reazonspeech
# 推荐: reazonspeech（日语更好）；其它语言可用 medium/large-v3
asr_model_size = reazonspeech

# 翻译模型
# 推荐: GGUF 量化（节省显存，T4 友好）
# 备选: SakuraLLM/Sakura-4B-Qwen3-Base-v2（日->中更好）
# 备选: facebook/nllb-200-distilled-1.3B（高质量）
# GGUF 用法：gguf:/path/to/model.gguf 或 gguf:hf:repo_id@filename.gguf
translation_model = gguf:hf:SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF@sakura-7b-qwen2.5-v1.0-iq4xs.gguf

# 使用GPU
use_gpu = true

# beam_size (1-5)
# 推荐: 3（优化值）
beam_size = 3

[Translation]
# 默认目标语言
default_target_language = zh

# 是否默认使用DeepSeek润色
use_deepseek_polish = true

[Subtitles]
# 字幕格式：srt / ass
format = srt
```

### 模型选择

#### ASR模型（Whisper）

| 模型 | 大小 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|---------|
| tiny | 39M | 最快 | 一般 | 测试 |
| base | 74M | 很快 | 可用 | 快速处理 |
| small | 244M | 快 | 良好 | 一般使用 |
| **medium** | 769M | 中等 | 优秀 | 其它语言推荐 |
| large-v3 | 1.5B | 慢 | 最好 | 高质量需求 |
| reazonspeech | - | 中等 | **日语更好** | 日语视频推荐 |

注：`reazonspeech` 使用 transformers 后端（默认模型：`japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large`），
可自定义为 `reazonspeech:HF模型名`；其它语言建议用 `medium/large-v3`。

#### 翻译模型

| 模型 | 大小 | 速度 | 质量 | 推荐 |
|------|------|------|------|------|
| Sakura-7B-Qwen2.5 (GGUF iq4xs) | 7B | 中等 | 高 | T4 友好 |
| nllb-200-distilled-600M | 600M | 快 | 良好 | 快速 |
| nllb-200-distilled-1.3B | 1.3B | 中等 | 高 | 通用 |
| m2m100_418M | 418M | 很快 | 可用 | 极速 |

注：Sakura LLM 为日->中翻译模型（CausalLM），显存占用较高；如显存不足可改用 GGUF 量化或 NLLB 系列。

#### GGUF（llama.cpp）翻译

- 适合使用量化 GGUF 模型节省显存
- 需要额外安装：`pip install llama-cpp-python`（GPU 可用 cuBLAS 版本，例如：`CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`）
- 设置示例：
  ```
  translation_model = gguf:/path/to/model.gguf
  # 或从 HF 下载：
  # translation_model = gguf:hf:repo_id@filename.gguf
  ```
  如果模型是 gated，请设置 `HF_TOKEN` 环境变量

## 💡 DeepSeek 润色

### 为什么使用 DeepSeek

1. **质量提升显著** - 从机器翻译到专业级翻译
2. **成本极低** - 30分钟视频仅 0.001元
3. **时间增加少** - 仅增加20-30%时间
4. **上下文感知** - v3.0 支持结合上下文润色

### 如何启用

1. 访问 https://platform.deepseek.com 注册
2. 充值（建议10元）
3. 获取API密钥
4. 配置 `config.ini`：
   ```ini
   [API]
   deepseek_api_key = sk-your-key-here
   
   [Translation]
   use_deepseek_polish = true
   ```
5. 直接运行（自动启用润色）：
   ```bash
   python batch_translate.py video.mp4 -t zh
   ```

### 命令行控制

```bash
# 使用配置文件默认设置
python batch_translate.py video.mp4 -t zh

# 强制启用润色（覆盖配置文件）
python batch_translate.py video.mp4 -t zh --polish

# 指定API密钥
python batch_translate.py video.mp4 -t zh --polish --deepseek-key sk-xxx
```

## 📋 命令行参数

### 完整参数列表

```bash
python batch_translate.py <输入> [选项]

必需参数:
  输入                      视频文件或目录路径

可选参数:
  -t, --target LANG        目标语言（默认: zh）
  -s, --source LANG        源语言（默认: auto自动检测）
  -o, --output DIR         输出目录（默认: 与视频同目录）
  -r, --recursive          递归处理子目录
  --translation-only       仅生成译文字幕（不含原文）
  --polish                 强制使用DeepSeek润色
  --deepseek-key KEY       指定DeepSeek API密钥
  --service-url URL        翻译服务地址
  --vocal-separation       启用人声分离（Demucs，需要 pip install demucs）
  --vocal-model NAME       Demucs 模型名（如: htdemucs / mdx_extra）
  --vocal-device DEV       人声分离设备：auto/cpu/cuda
  --cuda-clear             在GPU重任务前清理CUDA缓存（降低OOM概率，略慢）
  --no-cuda-clear          禁用GPU任务前清理CUDA缓存
  --wait-ready             等待翻译服务就绪（首次加载模型可能较久）
  --wait-timeout SEC       等待服务就绪超时秒数（默认: 3600）
  --show-progress          显示当前进度
  --reset-progress         清除进度记录，从头开始
  -h, --help               显示帮助信息
```

### 使用示例

```bash
# 示例1: 基本翻译
python batch_translate.py video.mp4 -t zh

# 示例2: 批量翻译（递归）
python batch_translate.py D:\Videos\ -t zh --recursive -o D:\Subtitles\

# 示例3: 查看进度
python batch_translate.py D:\Videos\ --show-progress

# 示例4: 清除进度重新开始
python batch_translate.py D:\Videos\ -t zh --reset-progress

# 示例5: 仅译文字幕
python batch_translate.py video.mp4 -t zh --translation-only

# 示例6: 中译英
python batch_translate.py chinese_video.mp4 -t en -s zh --polish

# 示例7: 嘈杂/背景音乐视频（启用人声分离）
python batch_translate.py noisy.mp4 -t zh --vocal-separation --vocal-device cuda --cuda-clear
```

## 📂 项目结构

```
video-translation-service/
├── README.md                   # 项目文档（本文件）
├── requirements.txt            # Python依赖列表
├── config.template.ini         # 配置文件模板
├── config.ini                  # 配置文件（不提交）
├── config_manager.py           # 配置管理模块
├── server_optimized.py         # 翻译服务
├── batch_translate.py          # 批量翻译工具（v3.0）
├── translation_polisher.py     # DeepSeek润色模块
├── check_config.py             # 配置检查工具
├── start_service.bat           # Windows启动脚本
├── .gitignore                  # Git忽略规则
├── log/                        # 日志目录
│   └── translation_*.log       # 翻译日志
├── .progress/                  # 进度目录
│   └── *.json                  # 进度文件
└── models/                     # 模型目录（自动下载）
    ├── whisper/                # ASR模型
    ├── nllb/                   # 翻译模型
    ├── sakura/                 # Sakura LLM 模型
    └── gguf/                   # GGUF 模型
```

## 🔧 进度管理

### 进度文件

进度自动保存在 `.progress/` 目录：

```json
{
  "video1.mp4": {
    "status": "completed",
    "srt_file": "video1_zh.srt",
    "timestamp": "2025-11-01 20:30:45",
    "duration": 125.3
  },
  "video2.mp4": {
    "status": "failed",
    "error": "识别超时",
    "retry_count": 2,
    "timestamp": "2025-11-01 20:35:22"
  }
}
```

### 状态说明

- **completed** - 已完成（自动跳过）
- **failed** - 已失败（自动重试，最多3次）
- **processing** - 处理中（上次中断，重新处理）

### 进度命令

```bash
# 查看进度
python batch_translate.py videos/ --show-progress

# 输出示例:
# ======================================================================
# 进度报告: videos
# ======================================================================
# 已完成: 45
# 已失败: 3
# 处理中: 0
# 总计: 48
# ======================================================================

# 清除进度
python batch_translate.py videos/ --reset-progress
```

## 📝 日志系统

### 日志位置

所有日志保存在 `log/` 目录：
- 文件名格式: `translation_YYYYMMDD_HHMMSS.log`
- 每次运行创建新日志文件
- 同时输出到终端和文件

## 🐛 常见问题

### Q: 如何确认DeepSeek润色正在工作？

A: 查看日志输出，应该显示：
```
✓ DeepSeek上下文润色已启用
[4/5] DeepSeek润色（结合上下文）...
翻译+润色: XX秒 (含DeepSeek上下文润色)
```

### Q: 翻译很慢怎么办？

A: 检查以下几点：
1. GPU是否启用：`config.ini` 中 `use_gpu = true`
2. 服务是否正常：访问 http://127.0.0.1:50515/health
3. 考虑使用更快的模型：`nllb-200-distilled-600M`

### Q: 视频识别超时怎么办？

A: v3.0 已优化：
- 语音识别超时：3600秒（1小时）
- 支持最长3小时视频
- 自动重试3次

### Q: DeepSeek API错误怎么办？

A: 检查以下几点：
1. API密钥是否正确：运行 `python check_config.py`
2. 账户余额是否充足
3. 网络连接是否正常

### Q: 中断后如何继续？

A: v3.0 自动断点续传：
```bash
# 第一次运行
python batch_translate.py videos/ -t zh
# Ctrl+C 中断

# 再次运行，自动继续
python batch_translate.py videos/ -t zh
# ✓ 自动跳过已完成
# ✓ 重试失败视频
```

### Q: 如何查看翻译质量对比？

A: 日志中会显示前3个润色示例：
```
[1] 原译: 我迫不及待地等着你看到美丽的场景
[1] 润色: 我迫不及待想让你看到那些美丽的景色
```

### Q: 可以批量处理多个任务吗？

A: 可以，但不推荐同时运行。建议：
1. 串行处理多个目录
2. 或分别使用不同的服务端口

### Q: 进度文件可以手动修改吗？

A: 可以。进度文件是 JSON 格式，可以手动编辑：
- 修改 `status` 强制重新处理
- 修改 `retry_count` 控制重试次数

## 🔄 更新日志

### v3.0 (2025-11-01)

**重大更新：**
- ✨ 新增上下文翻译润色（结合前后2句话）
- ✨ 新增断点续传功能
- ✨ 新增进度管理（查看/重置）
- ✨ 支持3小时长视频
- ✨ 自动重试机制（失败重试3次）
- 📝 完整的日志系统

**优化：**
- ⚡ 语音识别超时：600秒 → 3600秒
- ⚡ 翻译/润色超时：30秒 → 90秒
- ⚡ 所有操作支持自动重试
- 📊 更详细的进度显示

### v2.0 (2025-10-15)

- ✨ 集成DeepSeek润色
- ✨ 日志系统
- ✨ 配置管理器
- ⚡ 性能优化

### v1.0 (2025-10-01)

- 🎉 初始版本
- ✨ 基本翻译功能
- ✨ 批量处理
- ✨ GPU加速

## 📄 许可证

MIT License

## 🙏 致谢

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - ASR引擎
- [SakuraLLM](https://huggingface.co/SakuraLLM) - 翻译模型
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - 翻译模型
- [DeepSeek](https://www.deepseek.com/) - AI润色服务

## 📞 支持

遇到问题？

1. 查看 [常见问题](#-常见问题) 部分
2. 运行配置检查：`python check_config.py`
3. 查看日志文件：`log/translation_*.log`
4. 查看完整文档：`BATCH_TRANSLATE_V3_GUIDE.txt`

## 🎯 最佳实践

1. **始终启用DeepSeek润色**
   - 质量提升显著
   - 成本极低（0.001元/视频）
   - v3.0 支持上下文，效果更好

2. **使用批量处理**
   ```bash
   python batch_translate.py videos/ -t zh --recursive
   ```
   - 自动断点续传
   - 失败自动重试
   - 可以随时中断

3. **定期检查配置**
   ```bash
   python check_config.py
   ```

4. **查看翻译日志**
   ```bash
   type log\translation_*.log
   ```

5. **合理使用进度管理**
   - 大批量任务前先查看进度
   - 失败较多时考虑清除进度重新开始

---

**现在开始翻译你的视频吧！** 🎉

```bash
# 一键开始
python batch_translate.py videos/ -t zh
```
