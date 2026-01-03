# CTC Forced Aligner - SRT/JSON Alignment Tool

一个基于 CTC (Connectionist Temporal Classification) 强制对齐的字幕时间戳校正工具。支持将 SRT 字幕或 JSON 格式的文本段落与音频进行对齐，生成精确的时间戳。

## 功能特性

- ✅ **多格式支持**: SRT 字幕文件和 JSON 格式输入/输出
- ✅ **自动音频转换**: 自动将任意音频格式转换为 16kHz WAV (需要 ffmpeg)
- ✅ **多语言支持**: 支持 1100+ 种语言 (ISO 639-3 语言代码)
- ✅ **GPU 加速**: 自动检测并使用 CUDA GPU 加速
- ✅ **管道模式**: 支持 stdin/stdout 用于与其他程序集成
- ✅ **调试模式**: 保存中间结果用于问题排查

## 系统要求

- Python 3.10+
- FFmpeg (用于音频转换)
- PyTorch (CPU 或 CUDA 版本)
- 本地对齐模型 (如 `mms-300m-1130-forced-aligner`)

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/corvo007/forced-aligner.git
cd forced-aligner

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 ctc-forced-aligner (重要!)
# 该库不在 PyPI 上，需要从 GitHub Releases 下载 wheel 文件手动安装：
# https://github.com/corvo007/ctc-forced-aligner/releases
pip install ctc_forced_aligner-x.x.x-py3-none-any.whl

# 4. 安装 FFmpeg
# Windows: 下载 ffmpeg.exe 并放在项目目录或添加到 PATH
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# 5. 下载对齐模型
# 从 HuggingFace 下载模型到本地目录
# 例如: https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner
```

> ⚠️ **重要**: `ctc-forced-aligner` 库不在 PyPI 上发布，必须从 [GitHub Releases](https://github.com/corvo007/ctc-forced-aligner/releases) 下载 wheel 文件手动安装。

## 快速开始

### 基本用法 (SRT 模式)

```bash
python align.py \
    --audio "audio.mp3" \
    --srt "subtitles.srt" \
    --model "./models/mms-300m-1130-forced-aligner" \
    --language "eng"
```

### JSON 模式

```bash
python align.py \
    --audio "audio.mp3" \
    --json-input "input.json" \
    --json-output "output.json" \
    --model "./models/mms-300m-1130-forced-aligner" \
    --language "cmn"
```

---

## 命令行参数

### 必需参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--audio` | `-a` | 音频文件路径 (支持 ffmpeg 支持的任意格式) |
| `--model` | `-m` | 本地模型目录路径 (必须包含 `config.json`) |

### 输入参数 (二选一)

| 参数 | 简写 | 说明 |
|------|------|------|
| `--srt` | `-s` | SRT 字幕文件路径 |
| `--json-input` | `-ji` | JSON 输入文件路径 (使用 `-` 表示 stdin) |

### 输出参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | `<input>_aligned.srt` | SRT 输出文件路径 |
| `--json-output` | `-jo` | stdout | JSON 输出文件路径 (使用 `-` 表示 stdout) |

### 对齐选项

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--language` | `-l` | `eng` | 语言代码 (ISO 639-3) |
| `--romanize` | `-r` | false | 对非拉丁文字进行罗马化处理 |
| `--batch-size` | `-b` | 4 | 推理批次大小 |

### 其他选项

| 参数 | 简写 | 说明 |
|------|------|------|
| `--keep-wav` | - | 保留转换后的 WAV 文件 |
| `--debug` | `-d` | 启用调试模式，保存中间结果 |
| `--debug-dir` | - | 调试文件输出目录 |

---

## 详细使用示例

### 1. SRT 字幕对齐

**输入文件 (subtitles.srt):**
```srt
1
00:00:00,000 --> 00:00:02,000
Hello, how are you?

2
00:00:02,000 --> 00:00:05,000
I'm doing great, thanks!
```

**命令:**
```bash
python align.py \
    --audio "podcast.mp3" \
    --srt "subtitles.srt" \
    --output "aligned.srt" \
    --model "./models/mms-300m" \
    --language "eng"
```

**输出文件 (aligned.srt):**
```srt
1
00:00:00,120 --> 00:00:01,850
Hello, how are you?

2
00:00:02,100 --> 00:00:04,780
I'm doing great, thanks!
```

---

### 2. JSON 输入/输出

**输入文件 (input.json):**
```json
{
  "segments": [
    {"index": 1, "text": "你好，世界"},
    {"index": 2, "text": "今天天气很好"},
    {"index": 3, "text": "我们一起去散步吧"}
  ]
}
```

> 💡 **提示**: `index`、`start`、`end` 字段都是可选的，只有 `text` 是必需的。

**命令:**
```bash
python align.py \
    --audio "chinese_audio.mp3" \
    --json-input "input.json" \
    --json-output "output.json" \
    --model "./models/mms-300m" \
    --language "cmn" \
    --romanize
```

**输出文件 (output.json):**
```json
{
  "segments": [
    {"index": 1, "start": 0.12, "end": 1.45, "text": "你好，世界"},
    {"index": 2, "start": 1.68, "end": 3.92, "text": "今天天气很好"},
    {"index": 3, "start": 4.15, "end": 6.78, "text": "我们一起去散步吧"}
  ],
  "metadata": {
    "count": 3,
    "processing_time": 8.45
  }
}
```

---

### 3. 管道模式 (Pipeline)

适用于与其他程序集成，通过 stdin/stdout 传递数据：

```bash
# 从 stdin 读取 JSON，输出到 stdout
echo '{"segments": [{"text": "Hello world"}]}' | \
python align.py \
    --audio "audio.wav" \
    --json-input - \
    --json-output - \
    --model "./models/mms-300m" \
    --language "eng" \
    2>/dev/null  # 隐藏日志
```

**与其他程序集成示例:**
```bash
# 从 ASR 输出 -> 对齐 -> 后处理
cat asr_output.json | \
python align.py -a audio.mp3 -ji - -jo - -m ./models/mms-300m -l eng 2>/dev/null | \
python postprocess.py
```

---

### 4. 日语对齐 (字符级)

对于 CJK 语言，工具自动使用字符级对齐：

```bash
python align.py \
    --audio "japanese_audio.mp3" \
    --srt "japanese_subs.srt" \
    --model "./models/mms-300m" \
    --language "jpn" \
    --romanize
```

---

### 5. 调试模式

启用调试模式可以保存所有中间结果，便于问题排查：

```bash
python align.py \
    --audio "audio.mp3" \
    --srt "subtitles.srt" \
    --model "./models/mms-300m" \
    --language "eng" \
    --debug \
    --debug-dir "./debug_output"
```

**调试输出目录结构:**
```
debug_output/
├── 00_summary.json          # 处理摘要信息
├── 01_original_segments.json # 原始输入段落
├── 02_full_text.txt         # 拼接后的完整文本
├── 03_tokens_starred.json   # 分词结果
├── 04_text_starred.json     # 文本分割结果
├── 05_word_timestamps.json  # 词级时间戳
├── 06_aligned_segments.json # 最终对齐结果
└── alignment.log            # 详细日志
```

---

## JSON 格式规范

### 输入格式

支持两种输入格式：

**格式 1: 带包装对象**
```json
{
  "segments": [
    {"index": 1, "start": 0.0, "end": 1.5, "text": "Hello"},
    {"index": 2, "start": 1.5, "end": 3.0, "text": "World"}
  ]
}
```

**格式 2: 直接数组**
```json
[
  {"text": "Hello"},
  {"text": "World"}
]
```

**字段说明:**

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `index` | int | ❌ | 段落序号 (自动生成) |
| `start` | float | ❌ | 原始开始时间 (秒) |
| `end` | float | ❌ | 原始结束时间 (秒) |
| `text` | string | ✅ | 文本内容 |

### 输出格式

```json
{
  "segments": [
    {"index": 1, "start": 0.12, "end": 1.45, "text": "Hello"},
    {"index": 2, "start": 1.50, "end": 2.98, "text": "World"}
  ],
  "metadata": {
    "count": 2,
    "processing_time": 5.67
  }
}
```

**输出字段说明:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `segments[].index` | int | 段落序号 |
| `segments[].start` | float | 对齐后的开始时间 (秒) |
| `segments[].end` | float | 对齐后的结束时间 (秒) |
| `segments[].text` | string | 原始文本 (不变) |
| `metadata.count` | int | 段落总数 |
| `metadata.processing_time` | float | 处理耗时 (秒) |

---

## 常用语言代码

| 语言 | 代码 | 是否需要 `--romanize` |
|------|------|------------------------|
| 英语 | `eng` | ❌ |
| 中文 (普通话) | `cmn` | ✅ |
| 日语 | `jpn` | ✅ |
| 韩语 | `kor` | ✅ |
| 德语 | `deu` | ❌ |
| 法语 | `fra` | ❌ |
| 西班牙语 | `spa` | ❌ |
| 俄语 | `rus` | ✅ |
| 阿拉伯语 | `ara` | ✅ |

> 💡 对于非拉丁字母的语言，通常需要使用 `--romanize` 参数。

---

## 常见问题

### 1. ffmpeg 未找到

```
RuntimeError: ffmpeg not found. Please install ffmpeg and add it to your PATH.
```

**解决方案:**
- Windows: 下载 [ffmpeg](https://ffmpeg.org/download.html) 并添加到 PATH
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### 2. 模型目录无效

```
Invalid model directory (config.json not found)
```

**解决方案:**
确保模型目录包含完整的模型文件：
```
model_directory/
├── config.json
├── model.safetensors (或 pytorch_model.bin)
├── preprocessor_config.json
├── tokenizer_config.json
└── vocab.json
```

### 3. CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案:**
- 减小 `--batch-size` 参数值
- 或者使用 CPU 模式 (设置 `CUDA_VISIBLE_DEVICES=""`)

### 4. 对齐结果不准确

**可能的原因和解决方案:**
- 检查语言代码是否正确
- 对于 CJK 语言，确保使用 `--romanize`
- 确保音频质量清晰，背景噪音较少
- 尝试使用更大的模型

---

## 许可证

MIT License

## 致谢

- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) - 核心对齐库
- [MMS (Massively Multilingual Speech)](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) - 多语言语音模型
