# douyin-liked-analyzer

本地优先的视频内容提取与总结实验项目。第一版优先做稳定、可验证的命令行 pipeline，不做 Web UI、多 agent、向量库或自动剪辑。

## 当前能力

- `inspect`：用 `ffprobe` 生成 `metadata.json`，支持 audio-only / video-only / audio+video 文件。
- `run`：串联 MVP pipeline，按顺序生成元数据、音频、转录、抽帧、OCR、时间线和 Markdown 总结。
- OCR 和 OpenAI 总结都是可降级步骤：缺少 PaddleOCR、`openai` 包或 `OPENAI_API_KEY` 时不会中断前面的确定性产物。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

可选能力：

```powershell
python -m pip install -e ".[ai]"
python -m pip install -e ".[ocr]"
```

## 使用

查看帮助：

```powershell
video-summary --help
```

真实 Douyin 素材建议按这个顺序处理：

```powershell
video-summary scan video --output outputs\scan-demo --force
```

查看 `outputs\scan-demo\candidate_pairs.json`，选择一组 `video_path` 和 `audio_path` 后合并：

```powershell
video-summary merge "video\<video-only文件>.mp4" "video\<audio-only文件>.mp4" --output outputs\merged-demo\merged.mp4 --force
```

再对合并后的文件运行 pipeline：

```powershell
video-summary run outputs\merged-demo\merged.mp4 --output outputs\run-merged-demo --force --skip-summary
```

只检查媒体流并输出元数据：

```powershell
video-summary inspect "video\<样例文件>.mp4" --output outputs\inspect-demo
```

运行 MVP pipeline：

```powershell
video-summary run "video\<样例文件>.mp4" --output outputs\mvp-demo
```

如果输出目录已存在，默认会拒绝覆盖；确认重跑时加：

```powershell
video-summary run "video\<样例文件>.mp4" --output outputs\mvp-demo --force
```

跳过 OpenAI 总结：

```powershell
video-summary run "video\<样例文件>.mp4" --output outputs\mvp-demo --skip-summary
```

## 输出文件

```text
outputs/<name>/
  metadata.json
  audio.wav
  transcript.json
  transcript.srt
  frames/
  frames.json
  ocr.json
  timeline_events.json
  final_summary.md
  run_status.json
```

扫描和合并会额外生成：

```text
outputs/scan-demo/
  media_scan.json
  candidate_pairs.json

outputs/merged-demo/
  merged.mp4
  merge_status.json
```

## 验证

```powershell
python -m pytest
git status --short
```

## 已知限制

- 当前不自动配对 audio-only 和 video-only 文件，只按单个输入文件处理。
- `scan` 只生成候选配对，不会自动批量合并所有文件。
- `merge` 需要手动指定一个 video-only 文件和一个 audio-only 文件。
- PaddleOCR 未安装时会输出空 OCR 结构并记录跳过原因。
- 缺少 `OPENAI_API_KEY` 时会生成带时间线证据的 fallback Markdown，而不是调用模型。
- 暂不做 PySceneDetect、GPT Vision、HTML report、batch 和 agent 封装。
