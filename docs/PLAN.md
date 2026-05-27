# 视频内容提取与总结系统规划文件

## 1. 项目目标

构建一个本地优先的视频分析与总结工具，输入视频文件后，自动提取以下信息：

1. 视频语音内容：转录为字幕、文本和时间戳片段。
2. 视频画面内容：抽取关键帧，识别画面中的文字、UI、PPT、代码、人物动作或场景变化。
3. 视频字幕信息：提取内嵌字幕、外挂字幕，或通过语音识别生成字幕。
4. 场景结构：根据镜头变化、画面变化或固定时间窗口切分视频。
5. 统一总结：生成时间线摘要、章节摘要、最终摘要和可检索结构化数据。

核心原则：

- 工具链负责提取事实。
- GPT 负责整合、描述、压缩和总结。
- Codex 或其他 coding agent 负责编写、维护、修复和扩展工程代码。
- 第一版先做确定性 pipeline，不急着做复杂 autonomous agent。

---

## 2. 非目标

第一阶段不做以下内容：

1. 实时视频流分析。
2. 多用户 Web 平台。
3. 视频编辑器。
4. 完整视频搜索引擎。
5. 复杂多 agent 自主规划系统。
6. 对每一帧进行视觉模型分析。
7. 高精度人物身份识别。
8. 自动生成剪辑成片。

原因很简单：这些会显著增加复杂度，但对 MVP 的核心价值帮助有限。

---

## 3. 典型使用场景

### 3.1 课程视频总结

输入：录屏课程、讲座、网课视频。

输出：

- 完整字幕
- 每一章讲了什么
- PPT / 屏幕文字 OCR
- 知识点摘要
- 时间戳目录

### 3.2 游戏或软件录屏分析

输入：游戏录屏、软件 bug 复现视频、产品演示视频。

输出：

- 关键画面说明
- UI 文字提取
- 操作流程总结
- 异常节点定位
- 可复现步骤

### 3.3 会议视频整理

输入：会议录屏。

输出：

- 发言转录
- 议题摘要
- 决策点
- 待办事项
- 时间线

### 3.4 短视频内容理解

输入：短视频、广告、教程片段。

输出：

- 画面内容摘要
- 字幕 / 旁白摘要
- 节奏和结构分析
- 可复用文案或脚本拆解

---

## 4. 总体架构

```text
input.mp4
   │
   ├── Preprocess Layer
   │     ├── ffprobe: metadata.json
   │     ├── ffmpeg: audio.wav
   │     ├── ffmpeg: sampled_frames/
   │     └── ffmpeg: embedded_subtitles.srt
   │
   ├── Analysis Layer
   │     ├── faster-whisper: transcript.json / transcript.srt
   │     ├── PaddleOCR: ocr_by_frame.json
   │     ├── PySceneDetect: scenes.json
   │     └── GPT Vision: frame_descriptions.json
   │
   ├── Alignment Layer
   │     ├── align transcript with scenes
   │     ├── align OCR with timestamps
   │     ├── deduplicate repeated frames / repeated OCR
   │     └── build timeline_events.json
   │
   ├── Summarization Layer
   │     ├── segment summaries
   │     ├── chapter summaries
   │     ├── final summary
   │     └── action items / key points
   │
   └── Output Layer
         ├── final_summary.md
         ├── timeline_summary.md
         ├── transcript.srt
         ├── transcript.json
         ├── scenes.json
         ├── ocr.json
         ├── frame_descriptions.json
         └── report.html / report.md
```

---

## 5. 推荐技术栈

### 5.1 本地处理

| 模块 | 工具 | 用途 |
|---|---|---|
| 视频元数据 | ffprobe | 获取时长、编码、分辨率、帧率、音轨、字幕轨 |
| 音频提取 | ffmpeg | 从视频中提取 wav / mp3 |
| 抽帧 | ffmpeg / OpenCV | 固定间隔抽帧或根据场景抽帧 |
| 场景切分 | PySceneDetect | 检测镜头变化 |
| 语音转文字 | faster-whisper | 生成带时间戳的转录文本 |
| OCR | PaddleOCR | 识别画面中文字 |
| 图片去重 | imagehash / OpenCV | 去除重复帧 |
| 数据模型 | Pydantic | 规范中间数据结构 |
| CLI | Typer / Click | 命令行入口 |
| 日志 | loguru / logging | 过程追踪 |

### 5.2 模型能力

| 模型能力 | 用途 |
|---|---|
| GPT 文本模型 | 总结字幕、OCR、时间线、章节 |
| GPT 视觉模型 | 描述关键帧、分析 UI、画面、场景 |
| Whisper / faster-whisper | 本地语音转录 |

### 5.3 Agent / Coding Agent

| 工具 | 角色 |
|---|---|
| Codex CLI | 写代码、改代码、运行测试、修 pipeline bug |
| OpenAI Agents SDK | 第二阶段把处理函数包装成 agent tools |
| LangGraph / CrewAI | 可选，不建议第一版使用 |

---

## 6. 项目目录结构

```text
video_summarizer/
  README.md
  pyproject.toml
  .env.example
  config.yaml
  Makefile

  src/
    video_summarizer/
      __init__.py
      main.py
      pipeline.py
      config.py
      logging_config.py

      core/
        metadata.py
        paths.py
        timestamps.py
        hashing.py
        validation.py

      tools/
        ffmpeg_tools.py
        whisper_tools.py
        ocr_tools.py
        scene_tools.py
        vision_tools.py
        summarize_tools.py

      models/
        metadata.py
        transcript.py
        scene.py
        frame.py
        ocr.py
        timeline.py
        report.py

      processors/
        frame_selector.py
        ocr_deduplicator.py
        transcript_cleaner.py
        timeline_builder.py
        chunker.py

      reports/
        markdown_writer.py
        html_writer.py
        srt_writer.py
        json_writer.py

      agents/
        tools.py
        video_agent.py
        prompts.py

  tests/
    test_timestamps.py
    test_chunker.py
    test_timeline_builder.py
    test_ocr_deduplicator.py

  examples/
    sample_config.yaml
    sample_outputs/

  outputs/
    .gitkeep
```

---

## 7. MVP 范围

第一版只实现这些：

1. 输入一个本地视频文件。
2. 用 ffprobe 获取视频元数据。
3. 用 ffmpeg 提取音频。
4. 用 faster-whisper 生成转录文本和 SRT。
5. 用 ffmpeg 每隔 N 秒抽一帧。
6. 用 PaddleOCR 对抽帧结果做 OCR。
7. 对 OCR 结果做简单去重。
8. 把 transcript + OCR + selected frame metadata 交给 GPT 生成总结。
9. 输出 Markdown 报告。

第一版不强制使用 PySceneDetect。场景切分留到第二版。

### MVP 命令示例

```bash
video-summary input.mp4 \
  --output outputs/input \
  --frame-interval 5 \
  --language zh \
  --whisper-model medium \
  --summary-mode balanced
```

### MVP 输出

```text
outputs/input/
  metadata.json
  audio.wav
  transcript.json
  transcript.srt
  frames/
    frame_000001.jpg
    frame_000002.jpg
  ocr.json
  timeline_events.json
  final_summary.md
```

---

## 8. 第二阶段范围

第二版加入：

1. PySceneDetect 场景切分。
2. 每个场景选择 1-3 张关键帧。
3. 图片相似度去重。
4. GPT 视觉模型分析关键帧。
5. 字幕、OCR、画面描述按时间戳对齐。
6. 分段总结，再总总结。
7. 生成时间线报告。
8. 支持批量处理目录。

第二版输出增加：

```text
scenes.json
scene_keyframes.json
frame_descriptions.json
chapter_summaries.md
timeline_summary.md
```

---

## 9. 第三阶段范围

第三版再考虑 agent 化和产品化：

1. 把每个处理模块包装成 agent tool。
2. Agent 根据视频类型自动选择策略。
3. 支持课程模式、会议模式、游戏录屏模式、短视频模式。
4. 支持失败重试、自动降级、断点续跑。
5. 支持生成 HTML 报告。
6. 支持把结果写入向量数据库，做视频内容搜索。
7. 支持 Web UI。

---

## 10. 数据结构设计

### 10.1 metadata.json

```json
{
  "video_path": "input.mp4",
  "duration_sec": 1832.4,
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "video_codec": "h264",
  "audio_codec": "aac",
  "has_audio": true,
  "has_subtitle": false,
  "created_at": "2026-05-27T00:00:00Z"
}
```

### 10.2 transcript.json

```json
{
  "language": "zh",
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 6.2,
      "text": "今天我们来讲视频分析工具的架构。"
    }
  ]
}
```

### 10.3 ocr.json

```json
{
  "frames": [
    {
      "frame_id": "frame_000001",
      "timestamp": 5.0,
      "image_path": "frames/frame_000001.jpg",
      "texts": [
        {
          "text": "系统架构",
          "confidence": 0.98,
          "bbox": [[10, 20], [300, 20], [300, 80], [10, 80]]
        }
      ]
    }
  ]
}
```

### 10.4 scenes.json

```json
{
  "scenes": [
    {
      "scene_id": 1,
      "start": 0.0,
      "end": 32.5,
      "keyframes": ["frames/frame_000001.jpg", "frames/frame_000005.jpg"]
    }
  ]
}
```

### 10.5 frame_descriptions.json

```json
{
  "frames": [
    {
      "frame_id": "frame_000001",
      "timestamp": 5.0,
      "description": "画面展示一页标题为系统架构的幻灯片，中央是视频处理流程图。",
      "visible_text_summary": "系统架构、输入视频、抽帧、OCR、总结"
    }
  ]
}
```

### 10.6 timeline_events.json

```json
{
  "events": [
    {
      "start": 0.0,
      "end": 32.5,
      "transcript": "今天我们来讲视频分析工具的架构。",
      "ocr_text": "系统架构 / 输入视频 / OCR / 总结",
      "visual_summary": "画面展示视频分析系统流程图。",
      "event_summary": "开头介绍系统目标和总体处理流程。"
    }
  ]
}
```

---

## 11. Pipeline 设计

### 11.1 主流程

```python
def run_pipeline(video_path: str, config: Config) -> PipelineResult:
    paths = prepare_output_paths(video_path, config.output_dir)

    metadata = probe_video(video_path)
    audio_path = extract_audio(video_path, paths.audio_path)

    transcript = transcribe_audio(audio_path, config.whisper)
    write_transcript_outputs(transcript, paths)

    frame_paths = extract_frames(
        video_path=video_path,
        output_dir=paths.frames_dir,
        interval_sec=config.frame_interval,
    )

    ocr_results = run_ocr(frame_paths)
    deduped_ocr = deduplicate_ocr(ocr_results)

    timeline = build_timeline(
        metadata=metadata,
        transcript=transcript,
        ocr=deduped_ocr,
        frame_paths=frame_paths,
    )

    report = summarize_timeline(timeline, config.summary)
    write_report(report, paths.final_summary_path)

    return PipelineResult(
        metadata=metadata,
        transcript=transcript,
        ocr=deduped_ocr,
        timeline=timeline,
        report=report,
    )
```

### 11.2 第二版流程

```python
def run_pipeline_v2(video_path: str, config: Config) -> PipelineResult:
    metadata = probe_video(video_path)
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)

    scenes = detect_scenes(video_path)
    keyframes = extract_keyframes_by_scene(video_path, scenes)
    keyframes = deduplicate_frames(keyframes)

    ocr_results = run_ocr(keyframes)
    frame_descriptions = describe_keyframes_with_gpt(keyframes)

    timeline = align_all_sources(
        transcript=transcript,
        scenes=scenes,
        ocr=ocr_results,
        frame_descriptions=frame_descriptions,
    )

    segment_summaries = summarize_segments(timeline)
    final_summary = summarize_over_segments(segment_summaries)

    write_all_outputs(...)
    return result
```

---

## 12. 抽帧策略

### 12.1 MVP 固定间隔抽帧

推荐默认值：

| 视频长度 | 抽帧间隔 |
|---|---|
| 小于 5 分钟 | 每 2 秒 |
| 5-30 分钟 | 每 5 秒 |
| 30-120 分钟 | 每 10 秒 |
| 超过 120 分钟 | 每 20 秒 |

### 12.2 第二版场景抽帧

策略：

1. 先用 PySceneDetect 切分场景。
2. 每个场景取开始、中间、结束各一帧。
3. 对相似帧做 hash 去重。
4. 对包含大量 OCR 文字的帧提高优先级。
5. 对视觉变化明显的帧提高优先级。

### 12.3 不要全量逐帧分析

逐帧分析会带来三个问题：

1. 成本高。
2. 速度慢。
3. 重复信息极多。

正确方式是先筛选，再交给模型。

---

## 13. OCR 去重策略

视频中 OCR 最大的问题不是识别，而是重复。

### 13.1 简单去重

对每一帧 OCR 结果合并为字符串，然后计算相邻帧文本相似度。

如果相似度高于阈值，例如 0.9，则只保留第一帧。

### 13.2 推荐字段

```json
{
  "timestamp": 35.0,
  "raw_text": "系统架构 输入视频 OCR 总结",
  "normalized_text": "系统架构 输入视频 OCR 总结",
  "is_duplicate": false,
  "duplicate_of": null
}
```

### 13.3 文字归一化

需要处理：

1. 空格差异。
2. 换行差异。
3. 标点差异。
4. OCR 误识别。
5. 重复页脚、页码、水印。

---

## 14. 时间线对齐策略

最终总结质量很大程度取决于时间戳对齐。

### 14.1 时间线事件生成

以以下单位之一作为时间线事件：

1. 固定时间窗口，例如每 30 秒。
2. Whisper segment 聚合。
3. PySceneDetect scene。
4. PPT / OCR 内容变化点。

第一版推荐：每 30-60 秒聚合一次。

第二版推荐：scene + transcript 混合切分。

### 14.2 合并规则

每个事件包含：

- start
- end
- transcript segments
- OCR frames
- visual descriptions
- scene id
- summary

---

## 15. 总结策略

### 15.1 不推荐一次性总结全视频

长视频一次性塞给模型，会出现：

1. 超上下文。
2. 细节丢失。
3. 时间线混乱。
4. 幻觉增加。

### 15.2 推荐分层总结

```text
transcript + OCR + visual notes
        │
        ▼
30-90 秒片段总结
        │
        ▼
章节总结
        │
        ▼
最终总结
```

### 15.3 报告结构

```markdown
# 视频总结

## 一句话概括

## 核心内容

## 时间线

| 时间 | 内容 | 画面信息 | 文字信息 |
|---|---|---|---|

## 关键知识点 / 事件

## 画面中出现的重要文字

## 可执行事项

## 可能遗漏或不确定的信息
```

---

## 16. Codex 使用方式

### 16.1 Codex 的正确角色

Codex 应该用来：

1. 创建项目结构。
2. 编写 Python CLI。
3. 封装 ffmpeg / whisper / OCR / scene detection。
4. 编写测试。
5. 修复报错。
6. 优化模块边界。
7. 生成 README 和示例命令。

Codex 不应该被当成直接视频理解模型。

### 16.2 可以交给 Codex 的初始任务

```text
Create a Python CLI project named video_summarizer.

Goal:
Build a local-first video summarization pipeline.

Requirements:
1. Use ffprobe to extract video metadata.
2. Use ffmpeg to extract audio.
3. Use faster-whisper to transcribe audio into transcript.json and transcript.srt.
4. Use ffmpeg to sample frames every N seconds.
5. Use PaddleOCR to extract text from sampled frames.
6. Deduplicate OCR results.
7. Build timeline_events.json by aligning transcript segments and OCR results by timestamp.
8. Use OpenAI API to generate final_summary.md.
9. Provide Typer CLI interface.
10. Add config.yaml support.
11. Add logging and error handling.
12. Add unit tests for timestamp formatting, timeline chunking, and OCR deduplication.

Do not build a web app yet.
Do not build a multi-agent system yet.
Focus on a reliable deterministic pipeline.
```

### 16.3 Codex 后续任务

```text
Add PySceneDetect support.
Use scene detection to select keyframes.
Keep the old fixed-interval frame extraction mode as fallback.
```

```text
Add GPT vision support.
For each selected keyframe, send the image to the vision model and save frame_descriptions.json.
Add cost control: max_frames, max_resolution, and skip_duplicate_frames.
```

```text
Add report.html generation.
The HTML report should include timeline, transcript excerpts, OCR text, keyframes, and summaries.
```

---

## 17. Agent 化设计

不要第一天就多 agent。等 pipeline 稳定后再加。

### 17.1 Tool 函数

```python
@function_tool
def probe_video(video_path: str) -> str:
    """Return video metadata as JSON."""

@function_tool
def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio from video and return audio path."""

@function_tool
def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio and return transcript JSON path."""

@function_tool
def extract_frames(video_path: str, output_dir: str, interval_sec: int) -> str:
    """Extract sampled frames and return frames directory."""

@function_tool
def run_ocr(frames_dir: str) -> str:
    """Run OCR on frames and return OCR JSON path."""

@function_tool
def summarize_artifacts(output_dir: str) -> str:
    """Summarize generated artifacts into Markdown report."""
```

### 17.2 Agent 角色

第一阶段不需要多 agent。

第二阶段最多拆成：

1. Pipeline Agent：决定执行哪些工具。
2. Vision Summary Agent：总结关键帧。
3. Report Agent：生成最终报告。

但默认仍然应该由确定性 pipeline 控制执行顺序。

### 17.3 Agent 不该做什么

Agent 不应该：

1. 随机决定是否跳过必要步骤。
2. 每个视频都重新发明流程。
3. 在没有日志的情况下静默失败。
4. 直接处理全部原始帧。
5. 生成无法复现的结果。

---

## 18. 配置文件设计

### config.yaml

```yaml
input:
  video_path: null

output:
  base_dir: outputs
  overwrite: false
  keep_intermediate_files: true

ffmpeg:
  frame_interval_sec: 5
  audio_sample_rate: 16000
  max_frame_width: 1280

whisper:
  model: medium
  device: auto
  compute_type: auto
  language: zh
  vad_filter: true

ocr:
  engine: paddleocr
  language: ch
  confidence_threshold: 0.5
  deduplicate: true
  duplicate_similarity_threshold: 0.9

scene_detection:
  enabled: false
  threshold: 27.0
  min_scene_len_sec: 2.0

vision:
  enabled: false
  model: gpt-4.1-mini
  max_frames: 80
  max_image_width: 1280

summary:
  model: gpt-4.1-mini
  mode: balanced
  segment_window_sec: 60
  output_language: zh
  include_uncertainties: true
```

---

## 19. CLI 设计

```bash
video-summary input.mp4
```

```bash
video-summary input.mp4 \
  --output outputs/demo \
  --frame-interval 5 \
  --language zh \
  --whisper-model medium
```

```bash
video-summary input.mp4 \
  --scene-detect \
  --vision \
  --max-frames 100
```

```bash
video-summary batch ./videos \
  --output outputs/batch \
  --language zh
```

---

## 20. 错误处理

必须处理以下错误：

1. ffmpeg 未安装。
2. 输入文件不存在。
3. 视频无音轨。
4. 视频无字幕轨。
5. Whisper 模型下载失败。
6. PaddleOCR 初始化失败。
7. OCR 结果为空。
8. GPT API 调用失败。
9. 输出目录已存在。
10. 视频过长导致抽帧过多。

失败策略：

- 非关键步骤失败时降级继续。
- 关键步骤失败时给出明确错误。
- 所有步骤写日志。
- 中间产物存在时支持断点续跑。

---

## 21. 性能与成本控制

### 21.1 限制帧数量

必须设置：

```yaml
vision:
  max_frames: 80
```

否则长视频会失控。

### 21.2 限制图片尺寸

建议将长边压缩到 1280 或以下。

### 21.3 优先本地处理

优先用 OCR、hash、scene detection 筛选关键帧，再调用 GPT 视觉模型。

### 21.4 缓存

缓存以下结果：

1. metadata.json
2. audio.wav
3. transcript.json
4. ocr.json
5. scenes.json
6. frame_descriptions.json

重复运行时不要重新计算已有产物，除非用户指定 `--force`。

---

## 22. 测试计划

### 22.1 单元测试

测试：

1. 时间戳格式转换。
2. Whisper segment 合并。
3. OCR 文本去重。
4. timeline chunking。
5. 输出路径生成。
6. config 加载。

### 22.2 集成测试

准备 3 个小视频：

1. 有语音无字幕。
2. 有 PPT 文字。
3. 有明显场景切换。

测试完整 pipeline 是否能产出所有文件。

### 22.3 回归测试

每次修改后，跑同一个 sample 视频，对比输出结构是否稳定。

---

## 23. 里程碑

### Milestone 1: CLI MVP

目标：能处理单个视频，输出 transcript、OCR 和 final_summary。

交付：

- CLI 可运行
- ffmpeg 抽音频和抽帧
- faster-whisper 转录
- PaddleOCR OCR
- Markdown 总结

### Milestone 2: 时间线增强

目标：字幕和 OCR 按时间线合并。

交付：

- timeline_events.json
- timeline_summary.md
- OCR 去重
- 分段总结

### Milestone 3: 场景切分与视觉分析

目标：增加关键帧视觉理解。

交付：

- PySceneDetect
- keyframe selection
- GPT vision descriptions
- scene summaries

### Milestone 4: Agent 封装

目标：把稳定 pipeline 包装成 agent tools。

交付：

- tools.py
- video_agent.py
- agent-driven report generation
- 自动降级和重试

### Milestone 5: 批量处理与搜索

目标：处理大量视频。

交付：

- batch mode
- HTML report
- optional vector search

---

## 24. 最小实现优先级

按这个顺序做，不要乱：

1. 项目结构。
2. ffprobe metadata。
3. ffmpeg extract audio。
4. faster-whisper transcript。
5. ffmpeg sample frames。
6. PaddleOCR OCR。
7. OCR 去重。
8. timeline builder。
9. GPT final summary。
10. CLI 参数和 config。
11. 基础测试。
12. PySceneDetect。
13. GPT vision。
14. HTML report。
15. Agent 封装。

---

## 25. 关键判断

最危险的错误不是技术选型，而是过早复杂化。

不要一开始就做：

- 多 agent
- Web UI
- 向量数据库
- 复杂任务队列
- 每帧视觉分析
- 全自动视频理解平台

先做一个可以稳定跑完 20 个真实视频的 CLI。能稳定跑完，再谈 agent 化和产品化。

---

## 26. 推荐第一周任务

### Day 1

- 初始化项目。
- 写 CLI。
- 接入 ffprobe。
- 生成 metadata.json。

### Day 2

- 接入 ffmpeg 音频提取。
- 接入 faster-whisper。
- 输出 transcript.json 和 transcript.srt。

### Day 3

- 接入 ffmpeg 抽帧。
- 建立 frame metadata。

### Day 4

- 接入 PaddleOCR。
- 输出 ocr.json。
- 做 OCR 去重。

### Day 5

- 实现 timeline_events.json。
- 把 transcript 和 OCR 对齐。

### Day 6

- 接入 GPT 文本总结。
- 输出 final_summary.md。

### Day 7

- 整理 README。
- 加测试。
- 用 3 个真实视频跑通。

---

## 27. 最终判断

这套系统的核心不是“让 agent 分析视频”，而是：

1. 用确定性工具提取可靠证据。
2. 用时间戳把证据组织起来。
3. 用模型做压缩、解释和总结。
4. 用 Codex / agent 降低工程实现成本。

只要你守住这个边界，项目就可控。反过来，如果你一开始就幻想一个 agent 自动看视频、自动判断、自动总结、自动改流程，最后大概率会得到一个慢、贵、不稳定、难 debug 的系统。

