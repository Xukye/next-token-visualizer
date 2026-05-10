# Next-Token Visualizer / 下一个 Token 可视化教具

A classroom tool for humanities, social science, and interdisciplinary instructors who want to discuss AI without requiring students to code. It shows how a language model generates text one token at a time. 这是一个给人文社科及跨学科课程教师使用的课堂教具，适合在不要求学生写代码的情况下讨论 AI。它用来展示语言模型如何一步一步预测并生成下一个 token。

## What It Does / 它展示什么

The app lets you enter a short prompt, runs a local Qwen language model, and saves an interactive HTML visualization. The visualization shows generated tokens and highlights earlier tokens that had stronger attribution scores for each next-token step. 这个工具允许你输入一个简短提示词，在本地运行 Qwen 语言模型，并保存一个交互式 HTML 可视化。可视化会展示模型生成的 token，并高亮在每一步生成中影响分数较高的前文 token。

It is especially useful for law, communication, sociology, public policy, education, digital humanities, and other courses where students discuss how AI systems produce, rank, recommend, or classify information. 它尤其适合法学、新闻传播、社会学、公共政策、教育学、数字人文等课程，用来讨论 AI 系统如何生成、排序、推荐或分类信息。

This is intended as a teaching aid, not as a complete explanation of how large language models work. 它是教学辅助工具，不是对大语言模型工作机制的完整解释。

## Quick Start / 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5050
```

On macOS, you can also double-click. 在 macOS 上，也可以直接双击启动：

```text
start_server.command
```

## Notes on Installation / 安装说明

The first run downloads the model `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face, so it requires an internet connection and enough local disk space. 首次运行会从 Hugging Face 下载 `Qwen/Qwen2.5-1.5B-Instruct` 模型，因此需要联网，并需要足够的本地磁盘空间。

PyTorch installation can vary by operating system and hardware. If `pip install -r requirements.txt` does not install a working PyTorch version, follow the official PyTorch installation selector for your machine, then run the install command again for the remaining packages. 不同系统和硬件上的 PyTorch 安装方式可能不同。如果 `pip install -r requirements.txt` 没有安装到可用的 PyTorch，请先按照 PyTorch 官方安装指引安装适合自己机器的版本，再安装其他依赖。

## Saved Outputs / 保存的结果

Generated visualizations are saved in this folder. 生成出来的可视化文件会保存在这个文件夹：

```text
outputs/
```

For a public teaching repository, use a small demo set that shows the tool across disciplines instead of making it look like a law-only project. The selected demo set has one Chinese example and two English examples. 如果要公开作为教学仓库，建议使用少量跨学科示例，避免让项目看起来只服务于法学课程。当前选定的 demo 包含一个中文示例和两个英文示例。

1. `01_解释一下著作权.html`: copyright as a legal, cultural, and creative-work topic. `01_解释一下著作权.html`：用著作权作为法律、文化与创作活动相关的示例。
2. `02_How_does_big_data_recommendation_decide_what_content_to_show.html`: platform recommendation as a media, communication, and data-society topic. `02_How_does_big_data_recommendation_decide_what_content_to_show.html`：用大数据推荐作为平台、传播与数据社会相关的示例。
3. `03_What_is_data_bias.html`: data bias as a social impact and fairness topic. `03_What_is_data_bias.html`：用数据偏见作为社会影响与公平性相关的示例。

## Demo Output Policy / 示例文件选择

This repository is meant to stay small and classroom-ready. 这个仓库应该保持轻量，方便课堂直接使用。

1. Generate short classroom examples with the web interface. 用网页界面生成简短的课堂示例。
2. Keep only the selected demo `.html` files in `outputs/`. 只把精选出来的 demo `.html` 文件保留在 `outputs/` 里。
3. Use readable filenames that show the prompt topic, such as `03_What_is_data_bias.html`. 使用能看出问题主题的文件名，例如 `03_What_is_data_bias.html`。
4. Prefer examples that help humanities and social science students discuss AI in their own fields. 优先选择能帮助人文社科学生把 AI 放回自己专业语境中讨论的示例。
5. Do not include `.DS_Store`, `__pycache__/`, `.venv/`, or model cache folders. 不要包含 `.DS_Store`、`__pycache__/`、`.venv/` 或模型缓存文件夹。

## Suggested Classroom Use / 课堂使用建议

1. Ask students what they think the model is doing before pressing generate. 生成前，先让学生猜模型到底在做什么。
2. Generate a short answer. 生成一个短回答。
3. Play the visualization step by step. 逐步播放可视化。
4. Discuss the difference between "next-token prediction" and "understanding." 讨论“预测下一个 token”和“理解”之间的区别。
5. Emphasize that attribution visualizations are approximations, not direct proof of model reasoning. 强调归因可视化只是近似，并不是模型推理过程的直接证据。

## Project Structure / 项目结构

```text
.
├── app.py
├── templates/
│   └── index.html
├── outputs/
│   ├── README.md
│   ├── .gitkeep
│   └── selected demo .html files
├── start_server.command
├── requirements.txt
├── TEACHING_NOTES.md
├── LICENSE
└── README.md
```

## Model / 使用模型

Default model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Default model is `Qwen/Qwen2.5-1.5B-Instruct`. 默认模型是 `Qwen/Qwen2.5-1.5B-Instruct`。

## Limitations / 局限

- The visualization uses a gradient-based attribution signal. It is an approximation. 这个可视化使用基于梯度的归因信号，只是一种近似。
- The highlighted tokens should not be treated as the model's full reasoning process. 高亮 token 不应被理解为模型完整的推理过程。
- Longer outputs take more memory and more time. 输出越长，所需内存和时间越多。
- Results can vary across hardware and model versions. 不同硬件和模型版本可能产生不同结果。
