# douyin-liked-analyzer

本地优先的视频证据提取与总结 CLI。当前目标是稳定处理本地抖音素材，生成可复现的证据产物：元数据、转录、抽帧、OCR、场景切分、关键帧视觉描述、时间线、章节摘要和最终总结。

暂不做 Web UI、Agent 封装、向量搜索或自动剪辑。

## 当前能力

- `doctor`：检查本机依赖、OCR/场景检测环境，以及当前配置的 LLM provider。
- `ocr-prepare`：准备并 smoke-test 仓库外的 PP-OCRv5 mobile 本地 OCR 模型。
- `inspect`：用 `ffprobe` 生成 `metadata.json`，支持 audio-only、video-only、audio+video 文件。
- `scan`：扫描素材目录，输出 `media_scan.json` 和 `candidate_pairs.json`，辅助匹配 audio-only / video-only 文件。
- `merge`：把一个 video-only 文件和一个 audio-only 文件合并成 MP4。
- `run`：串联完整 pipeline，生成转录、抽帧、OCR、场景、关键帧、视觉描述、时间线和 Markdown 总结。

## 安装

基础安装：

```powershell
python -m pip install -e .
```

完整安装：

```powershell
python -m pip install -e ".[all]"
```

按能力安装：

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e ".[transcribe]"
python -m pip install -e ".[ocr]"
python -m pip install -e ".[scene]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[ai]"
python -m pip install -e ".[gemini]"
```

PP-OCRv5 mobile CPU 本地 OCR 还需要 PaddlePaddle：

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

本机还必须能在 `PATH` 中找到 `ffmpeg` 和 `ffprobe`。

## 依赖检查

```powershell
video-summary doctor
```

`doctor` 会检查：

- `ffmpeg` / `ffprobe`：基础媒体处理，缺失时不能完成稳定 CLI 基线。
- `faster-whisper`：生成 `transcript.json` 和 `transcript.srt`。
- `PaddlePaddle` / `PaddleOCR` / `PP-OCRv5 model_root`：生成真实 PP-OCRv5 mobile OCR。
- `PySceneDetect` / `OpenCV`：启用 `--scene-detect` 后生成真实场景切分。
- `summary provider (...)`：最终 LLM 总结所需 SDK、模型名和 API key。
- `vision provider (...)`：只在 `vision.enabled` 或 `--vision` 启用时检查视觉理解能力。
- `Pillow`：视觉输入压缩和缩放。

默认配置使用 OpenAI 做 summary provider。如果没有安装 `openai` 包或没有设置 `OPENAI_API_KEY`，本地证据链仍可运行，但最终总结会标记为失败，不能算完整 LLM 验收通过。

## 多模型 LLM Provider

项目不再把 AI 能力写死到 `openai` / `OPENAI_API_KEY`。第一版支持：

- `openai`：OpenAI-compatible，默认 key 为 `OPENAI_API_KEY`，支持文本总结和 Vision。
- `deepseek`：OpenAI-compatible，默认 `https://api.deepseek.com`，key 为 `DEEPSEEK_API_KEY`，第一版只做文本总结。
- `qwen`：阿里云百炼 OpenAI-compatible，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`，key 为 `DASHSCOPE_API_KEY`，第一版只做文本总结。
- `gemini`：使用 `google-genai`，key 为 `GEMINI_API_KEY`，第一版做文本总结，Vision 位置预留。
- `local`：本地 OpenAI-compatible 服务，默认 `http://localhost:11434/v1`，不要求 key。

多个服务商的 API key 可以集中放在项目根目录 `.env`。`.env` 已被 Git 忽略，不要提交真实密钥。可以从示例复制：

```powershell
Copy-Item .env.example .env
```

`.env` 示例：

```dotenv
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
GEMINI_API_KEY=...
```

`config.yaml` 默认会读取 `.env`：

```yaml
llm:
  env_file: .env
  override_env: false
```

`override_env: false` 表示系统环境变量优先；如果系统里已经设置了同名变量，`.env` 不会覆盖它。

典型配置：

```yaml
summary:
  provider: deepseek
  model: deepseek-v4-pro

vision:
  enabled: false
  provider: openai
  model: gpt-4.1-mini

llm:
  providers:
    local:
      base_url: http://localhost:11434/v1
      api_key_required: false
```

DeepSeek V4 总结默认使用思考模式以提高最终总结质量：

```yaml
llm:
  providers:
    deepseek:
      reasoning_effort: max
      extra_body:
        thinking:
          type: enabled
```

这是 DeepSeek 官方 OpenAI SDK 写法对应的持久配置：`reasoning_effort="high/max"` 控制思考强度，`extra_body={"thinking":{"type":"enabled"}}` 打开思考模式。思考模式下不要依赖 `temperature`、`top_p` 等采样参数。

也可以在命令行临时覆盖：

```powershell
video-summary run outputs\merged-demo\merged.mp4 --output outputs\llm-local --force --scene-detect --summary-provider local --summary-model "<local-model-name>"
video-summary run outputs\merged-demo\merged.mp4 --output outputs\llm-deepseek --force --scene-detect --summary-provider deepseek --summary-model deepseek-v4-pro
video-summary run outputs\merged-demo\merged.mp4 --output outputs\llm-openai-vision --force --scene-detect --vision --summary-provider openai --summary-model gpt-4.1-mini --vision-provider openai --vision-model gpt-4.1-mini --max-frames 20
```

`--vision` 是可选能力。若选择的 provider 不支持图片理解，`frame_descriptions.json` 会记录 `provider_vision_unsupported`，但不会阻断最终文本总结。

最终总结默认是必需能力：未使用 `--skip-summary` 时，缺 provider、缺 SDK、缺 key 或模型调用失败都会写出 fallback `final_summary.md`，但 `run_status.json` 中 `summary.status` 为 `failed`，CLI 退出码为 1。

## PP-OCRv5 Mobile 本地 OCR

默认配置：

```yaml
ocr:
  engine: paddleocr
  version: PP-OCRv5
  device: cpu
  model_root: D:\someElse\video_summarizer-models\paddleocr
  text_detection_model_name: PP-OCRv5_mobile_det
  text_recognition_model_name: PP-OCRv5_mobile_rec
```

首次准备模型：

```powershell
video-summary ocr-prepare
```

模型固定在仓库外：

```text
D:\someElse\video_summarizer-models\paddleocr\official_models\
  PP-OCRv5_mobile_det\
  PP-OCRv5_mobile_rec\
```

## PySceneDetect 场景检测

默认使用 PySceneDetect `ContentDetector`。它适合抖音、录屏、讲解类视频的常规场景切分。

安装：

```powershell
python -m pip install -e ".[scene]"
```

`scene` extra 固定为：

```text
scenedetect==0.6.5.2
platformdirs>=4
opencv-contrib-python==4.10.0.84
```

不要使用 `scenedetect[opencv]`。当前 PP-OCRv5 复用 `opencv-contrib-python 4.10.0.84`，`scenedetect[opencv]` 容易额外拉入 `opencv-python`，造成同一环境双 OpenCV wheel 冲突。

默认配置：

```yaml
scene_detection:
  enabled: false
  detector: content
  threshold: 27.0
  min_scene_len_sec: 2.0
  adaptive_threshold: 3.0
  min_content_val: 15.0
  window_width: 2
```

只有当 `content` 对运动剧烈视频误切较多时，才改为：

```yaml
scene_detection:
  detector: adaptive
```

## 典型工作流

扫描素材：

```powershell
video-summary scan video --output outputs\scan-demo --force
```

查看 `outputs\scan-demo\candidate_pairs.json`，选择一组 `video_path` 和 `audio_path` 后合并：

```powershell
video-summary merge "<candidate video_path>" "<candidate audio_path>" --output outputs\merged-demo\merged.mp4 --force
```

完整运行：

```powershell
video-summary run outputs\merged-demo\merged.mp4 --output outputs\final-acceptance --force --scene-detect --vision --max-frames 20 --max-image-width 1280
```

只跑本地确定性证据链，不调用 LLM：

```powershell
video-summary run outputs\merged-demo\merged.mp4 --output outputs\local-only --force --scene-detect --skip-summary
```

## 输出文件

`run` 输出目录通常包含：

```text
outputs/<name>/
  metadata.json
  audio.wav
  transcript.json
  transcript.srt
  frames/
  frames.json
  scenes.json
  scene_keyframes.json
  frame_descriptions.json
  ocr.json
  timeline_events.json
  timeline_summary.md
  chapter_summaries.md
  final_summary.md
  run_status.json
```

## 降级和验收口径

- `PaddlePaddle` / `PaddleOCR` 缺失：`ocr.json` 写入空 OCR 结构和原因，pipeline 可继续，但 OCR 验收未通过。
- PP-OCRv5 模型目录未准备：运行 `video-summary ocr-prepare`。
- `PySceneDetect` 未安装、未启用或检测失败：`scenes.json` 写入 whole-video fallback。
- `ContentDetector` 没有检测到切点：回退为整段视频场景，长讲解视频或画面变化少的视频可能出现。
- `--vision` 未启用：`frame_descriptions.json` 记录 `vision_disabled`。
- Vision provider 不支持图片理解：记录 `provider_vision_unsupported`，不阻断 summary。
- Summary provider 不可用：`final_summary.md` 写 fallback，`summary.status = failed`，CLI 退出码为 1。
- `--skip-summary`：仍生成 `timeline_summary.md` 和 `chapter_summaries.md`，`final_summary.md` 为本地 fallback，CLI 不因 summary 失败退出。

完整验收时，`run_status.json` 中 OCR、scenes、summary 应为 `ok`；Vision 只有在启用并选择支持图片理解的 provider 时才要求 `ok` 或可解释的 `partial`。

## 回归验证

```powershell
python -m pytest
video-summary doctor
video-summary ocr-prepare
git status --short
```

本地 `video/` 目录只作为测试素材来源，不进入 Git。
