import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import datetime
import re
import numpy as np
from flask import Flask, request, render_template, Response, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Ensure outputs directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Global Model Initialization (Loaded once at server start)
# ---------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
print(f"Using device: {device}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
model.to(device)
model.eval()

# CRITICAL MEMORY FIX: Disable gradients for model weights so backward() only computes for embeddings
for param in model.parameters():
    param.requires_grad = False

print("Model loaded successfully! Server is ready.")

def clear_memory(device):
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# Generator Function for SSE
# ---------------------------------------------------------
def generate_stream(prompt, max_new_tokens):
    target_words = max(1, int(max_new_tokens) - 15)
    lower_bound = int(target_words * 0.7)
    upper_bound = int(target_words * 1.3)
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Please provide a plain-text answer between {lower_bound} and {upper_bound} words long. Avoid numbered lists, bullet points, and Markdown formatting."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    generated_ids = input_ids.clone()
    
    steps_data = []
    # Replace newlines for HTML display compatibility
    tokens = [tokenizer.decode([tok]).replace('\n', ' ') for tok in input_ids[0]]
    prompt_length = len(tokens)
    
    # Notify start
    yield f"data: {json.dumps({'progress': 0, 'status': 'generating'})}\n\n"
    
    for step in range(1024): # Allow natural generation up to 1024 tokens
        # Memory cleanup before each pass to prevent MPS OOM
        clear_memory(device)

        # Detach from the embedding layer and make it a leaf tensor that requires gradients
        inputs_embeds = model.get_input_embeddings()(generated_ids).detach()
        inputs_embeds.requires_grad_(True)
        
        outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = outputs.logits[0, -1, :] 
        
        next_token_id = torch.argmax(logits).item()
        next_token_str = tokenizer.decode([next_token_id]).replace('\n', ' ')

        # Do not add the terminal ChatML/EOS token as a visualization step.
        # It is not rendered as a visible token below, so keeping a step for it
        # would make the final frame point at a missing DOM element.
        if next_token_id == tokenizer.eos_token_id:
            break

        with torch.no_grad():
            display_logits = logits.detach().to(torch.float32)
            probs = torch.softmax(display_logits, dim=-1)
            candidate_count = min(12, display_logits.shape[0])
            candidate_probs, candidate_ids = torch.topk(probs, k=candidate_count)
            candidate_logits = display_logits[candidate_ids]
            step_candidates = []
            for rank, (candidate_id, candidate_logit, candidate_prob) in enumerate(
                zip(candidate_ids.tolist(), candidate_logits.tolist(), candidate_probs.tolist()),
                start=1,
            ):
                candidate_text = tokenizer.decode([candidate_id]).replace('\n', ' ')
                step_candidates.append({
                    "rank": rank,
                    "token": candidate_text,
                    "token_id": int(candidate_id),
                    "score": float(candidate_logit),
                    "prob": float(candidate_prob),
                    "chosen": int(candidate_id) == next_token_id,
                })
        
        target_logit = logits[next_token_id]
        model.zero_grad(set_to_none=True)
        target_logit.backward()
        
        grad = inputs_embeds.grad[0]
        embeds = inputs_embeds[0].detach()
        
        grad_x_input = grad * embeds
        scores = torch.norm(grad_x_input, dim=-1).cpu().to(torch.float32).numpy()
        
        # Sparsification
        top_k = 10
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Normalize top scores so the highest is 1.0 (for opacity scaling)
        max_score = float(scores[top_indices[0]]) if float(scores[top_indices[0]]) > 0 else 1.0
        sum_scores = float(np.sum(scores)) if float(np.sum(scores)) > 0 else 1.0
        
        step_attributions = []
        for idx in top_indices:
            weight = float(scores[idx]) / max_score
            pct = (float(scores[idx]) / sum_scores) * 100
            if weight > 0.05: # Threshold
                step_attributions.append({"src": int(idx), "weight": weight, "pct": pct, "score": float(scores[idx])})
                
        steps_data.append({
            "generated_token_idx": prompt_length + step,
            "generated_token": next_token_str,
            "candidates": step_candidates,
            "attributions": step_attributions
        })
        
        # Append to sequence
        tokens.append(next_token_str)
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)
        
        # Cleanup tensors
        del inputs_embeds
        del outputs
        del logits
        del display_logits
        del probs
        del candidate_probs
        del candidate_ids
        del candidate_logits
        del target_logit
        del grad
        del embeds
        del grad_x_input
        del scores
        
        # Yield progress (cap at 99% until actually done)
        progress = min(int(((step + 1) / max_new_tokens) * 100), 99)
        yield f"data: {json.dumps({'progress': progress, 'status': 'generating', 'token': next_token_str})}\n\n"

    # Generation complete, save HTML
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
    clean_prompt = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', prompt).strip()
    if ' ' in clean_prompt:
        safe_prompt = "_".join(clean_prompt.split()[:5])
    else:
        safe_prompt = clean_prompt[:5]
    if not safe_prompt:
        safe_prompt = "output"
        
    output_filename = f"{timestamp}_{safe_prompt}.html"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    generate_html(tokens, steps_data, prompt_length, output_path)
    
    # Notify complete and send the link
    yield f"data: {json.dumps({'progress': 100, 'status': 'done', 'link': f'/outputs/{output_filename}', 'filename': output_filename})}\n\n"


def generate_html(tokens, steps_data, prompt_length, output_path):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Next-Token Visualization | 下一个 Token 可视化</title>
        <style>
            body {
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Arial Unicode MS', 'Microsoft YaHei', sans-serif;
                margin: 0;
                padding: 24px clamp(24px, 5vw, 72px);
                display: flex;
                flex-direction: column;
                align-items: center;
                box-sizing: border-box;
            }
            .page-shell {
                width: min(1280px, 100%);
            }
            .controls {
                margin-bottom: 30px;
                z-index: 10;
            }
            button {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 10px 20px;
                cursor: pointer;
                border-radius: 5px;
                font-size: 16px;
                margin: 0 10px;
            }
            button:hover { background-color: #444; }
            .container {
                position: relative;
                width: 100%;
                text-align: left;
                padding: 20px;
                background: #1e1e1e;
                border-radius: 10px;
                white-space: pre-wrap;
                overflow: hidden;
                box-sizing: border-box;
            }
            .role-block {
                margin-bottom: 20px;
            }
            .role-block-system {
                line-height: 1.6;
                margin-top: 0;
            }
            .role-block-user, .role-block-assistant {
                line-height: 3.2;
            }
            .token.future-token {
                opacity: 0.2 !important;
                transition: opacity 0.3s;
            }
            .token {
                display: inline-block;
                padding: 2px 1px;
                margin: 0;
                border-radius: 3px;
                transition: background-color 0.2s, box-shadow 0.2s;
                position: relative;
                z-index: 2;
                font-size: 18px;
                opacity: 1;
            }
            .token::after {
                content: '';
                position: absolute;
                bottom: 1px;
                left: 50%;
                transform: translateX(-50%);
                width: 4px;
                height: 4px;
                background-color: #666;
                border-radius: 50%;
                opacity: 0.5;
            }
            .token.role-system { color: #666666; font-size: 14px; }
            .token.role-user { color: #a0c4ff; }
            .token.role-assistant { color: #ffadad; }
            .token.active-target { 
                background-color: rgba(255, 173, 173, 0.4); 
                box-shadow: 0 0 6px rgba(255, 173, 173, 0.5);
                z-index: 3;
            }
            .token.pinned-target {
                outline: 1px solid #ffd166;
                box-shadow: 0 0 10px rgba(255, 209, 102, 0.55);
            }
            .token.hover-target {
                outline: 2px solid #95d5b2;
                box-shadow: 0 0 12px rgba(149, 213, 178, 0.7);
            }
            .token.active-source { 
                background-color: rgba(160, 196, 255, 0.4); 
                box-shadow: 0 0 6px rgba(160, 196, 255, 0.5);
                z-index: 3;
                text-shadow: 0px 0px 4px rgba(160, 196, 255, 0.8);
            }
            .token:hover {
                background-color: rgba(149, 213, 178, 0.4);
                box-shadow: 0 0 6px rgba(149, 213, 178, 0.5);
                cursor: pointer;
                z-index: 4;
            }
            
            #reference-panel {
                background: #1a1a1a;
                padding: 20px;
                border-radius: 10px;
                margin-top: 24px;
                box-sizing: border-box;
                width: 100%;
            }
            #reference-panel h3 {
                margin-top: 0;
                color: #a0c4ff;
                font-size: 16px;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }
            .reference-grid {
                display: grid;
                grid-template-columns: minmax(520px, 1.35fr) minmax(320px, 1fr);
                gap: 22px;
                align-items: start;
            }
            .reference-card {
                min-width: 0;
                background: #202020;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 14px;
                box-sizing: border-box;
            }
            .attribution-card {
                display: grid;
                grid-template-columns: minmax(160px, 220px) minmax(300px, 1fr);
                gap: 18px;
                align-items: start;
            }
            .reference-pie {
                display: flex;
                justify-content: center;
                align-items: flex-start;
                min-height: 180px;
            }
            .reference-list {
                min-width: 0;
            }
            .reference-list-title {
                color: #888;
                font-size: 13px;
                margin-bottom: 12px;
            }
            .data-table {
                display: grid;
                grid-template-columns: var(--token-col, 8ch) 38px minmax(110px, 1fr) 54px;
                column-gap: 8px;
                row-gap: 10px;
                align-items: center;
            }
            .data-row {
                display: contents;
                cursor: help;
            }
            .data-row > * {
                cursor: help;
            }
            .data-score {
                color: #888;
                text-align: right;
                font-size: 12px;
                font-variant-numeric: tabular-nums;
            }
            .data-label {
                color: #e0e0e0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                font-size: 14px;
                min-width: 0;
            }
            .data-bar-bg {
                height: 10px;
                background: #2a2a2a;
                border-radius: 5px;
                overflow: hidden;
                min-width: 0;
            }
            .data-bar-fill {
                height: 100%;
                border-radius: 5px;
                transition: width 0.3s ease;
            }
            .data-pct {
                text-align: right;
                font-weight: bold;
                color: #fff;
                font-size: 14px;
                font-variant-numeric: tabular-nums;
                flex-shrink: 0;
            }
            .chart-row {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }
            .chart-label {
                width: 150px;
                text-align: right;
                padding-right: 15px;
                font-weight: bold;
                color: #e0e0e0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .chart-bar-bg {
                flex-grow: 1;
                background: #333;
                height: 18px;
                border-radius: 4px;
                overflow: hidden;
                position: relative;
            }
            .chart-bar-fill {
                height: 100%;
                background: linear-gradient(90deg, #4dabf7, #95d5b2);
                border-radius: 4px;
                transition: width 0.3s ease;
            }
            .chart-pct {
                position: absolute;
                right: 10px;
                top: 0;
                line-height: 18px;
                font-size: 12px;
                color: #fff;
                font-weight: bold;
                text-shadow: 1px 1px 2px #000;
            }
            
            #svg-layer {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }
            .attr-path {
                fill: none;
                stroke-dasharray: 1000;
                stroke-dashoffset: 1000;
                animation: dash 0.2s ease-out forwards;
            }
            @keyframes dash {
              to {
                stroke-dashoffset: 0;
              }
            }
            @media (max-width: 760px) {
                body { padding: 16px; }
                .reference-grid {
                    grid-template-columns: 1fr;
                }
                .attribution-card {
                    grid-template-columns: 1fr;
                }
                .reference-pie {
                    min-height: auto;
                }
            }
        </style>
    </head>
    <body>
        <div class="page-shell">
            <div class="controls" style="display: flex; align-items: center; width: 100%;">
                <button id="btn-prev">Previous / 上一步</button>
                <button id="btn-play">Play / Pause / 播放/暂停</button>
                <button id="btn-next">Next / 下一步</button>
                <input type="range" id="timeline-slider" min="-1" max="0" value="-1" style="margin-left: 20px; flex-grow: 1;">
            </div>
            <div class="container" id="text-container"><svg id="svg-layer"></svg></div>
            <section id="reference-panel">
                <h3>Data Reference / 数据参考</h3>
                <div id="reference-content">Hover over a token to inspect its attention, or click to pin it. / 将鼠标悬停在 Token 上查看注意力分布，或点击固定这个 Token 的数据。</div>
            </section>
        </div>

        <script>
            const data = DATA_PLACEHOLDER;
            const container = document.getElementById('text-container');
            const svgLayer = document.getElementById('svg-layer');
            const slider = document.getElementById('timeline-slider');
            const referenceContent = document.getElementById('reference-content');
            let currentStep = -1;
            let isPlaying = false;
            let playInterval;
            let pinnedTarget = null;
            let hoverTarget = null;
            const readyStep = data.steps.length;

            function isInteractionReady() {
                return data.steps.length > 0 && currentStep >= readyStep;
            }

            function clearInteractionState() {
                pinnedTarget = null;
                hoverTarget = null;
            }

            function stopPlayback() {
                if (playInterval) {
                    clearInterval(playInterval);
                    playInterval = null;
                }
                isPlaying = false;
            }

            function setCurrentStep(nextStep) {
                currentStep = Math.max(-1, Math.min(readyStep, nextStep));
                slider.value = currentStep;
            }

            slider.max = readyStep;
            slider.addEventListener('input', (e) => {
                stopPlayback();
                setCurrentStep(parseInt(e.target.value));
                updateView();
            });

            // Render Tokens
            let currentRole = 'system';
            let currentBlock = document.createElement('div');
            currentBlock.className = 'role-block role-block-system';
            container.appendChild(currentBlock);

            data.tokens.forEach((t, i) => {
                const span = document.createElement('span');
                span.id = 'token-' + i;
                if (i >= data.prompt_length && i < data.prompt_length + data.steps.length) {
                    span.dataset.generated = 'true';
                }
                
                // Determine Role and add breaks
                let t_trim = t.trim();
                if (i > 0 && data.tokens[i-1].includes('<|im_start|>')) {
                    if (t_trim === 'system' || t_trim === 'user' || t_trim === 'assistant') {
                        currentRole = t_trim;
                        currentBlock = document.createElement('div');
                        currentBlock.className = 'role-block role-block-' + currentRole;
                        container.appendChild(currentBlock);
                    }
                }

                span.className = 'token role-' + currentRole;
                
                // Hide special ChatML tokens completely
                let isHidden = false;
                if (t.includes('<|im_start|>') || t.includes('<|im_end|>')) {
                    isHidden = true;
                }
                if (i > 0 && data.tokens[i-1].includes('<|im_start|>') && (t_trim === 'system' || t_trim === 'user' || t_trim === 'assistant')) {
                    isHidden = true;
                }
                
                if (isHidden) {
                    span.classList.add('hidden-token');
                    span.style.display = 'none';
                }
                
                span.innerText = t;
                currentBlock.appendChild(span);
            });
            function escapeHtml(value) {
                return String(value)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }

            const chartColors = ['#ff6b6b', '#4dabf7', '#95d5b2', '#ffd166', '#a0c4ff', '#e0e0e0', '#ff9f1c', '#b5179e', '#f72585', '#4361ee', '#7209b7'];

            function polarPoint(cx, cy, radius, angleDegrees) {
                const angleRadians = (angleDegrees - 90) * Math.PI / 180.0;
                return {
                    x: cx + (radius * Math.cos(angleRadians)),
                    y: cy + (radius * Math.sin(angleRadians))
                };
            }

            function pieSlicePath(cx, cy, radius, startAngle, endAngle) {
                const safeEndAngle = Math.min(endAngle, startAngle + 359.99);
                const start = polarPoint(cx, cy, radius, safeEndAngle);
                const end = polarPoint(cx, cy, radius, startAngle);
                const largeArcFlag = safeEndAngle - startAngle <= 180 ? "0" : "1";
                return [
                    "M", cx, cy,
                    "L", start.x, start.y,
                    "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y,
                    "Z"
                ].join(" ");
            }

            function tooltipText(item, scoreLabel) {
                const pieces = [`${item.text}: ${item.pct.toFixed(2)}%`];
                if (Number.isFinite(item.score)) {
                    pieces.push(`${scoreLabel}: ${item.score.toFixed(4)}`);
                }
                return escapeHtml(pieces.join(" | "));
            }

            function textUnits(value) {
                return Array.from(String(value)).reduce((count, char) => {
                    return count + (/[\u3400-\u9fff\uff00-\uffef]/.test(char) ? 2 : 1);
                }, 0);
            }

            function tokenColumnStyle(items) {
                const maxUnits = Math.max(4, ...items.map(item => textUnits(item.text)));
                const colCh = Math.min(Math.max(maxUnits + 1, 5), 14);
                return `--token-col: ${colCh}ch;`;
            }

            function renderPieSvg(items, scoreLabel) {
                let currentAngle = 0;
                const slices = items.map((item, index) => {
                    const startAngle = currentAngle;
                    const endAngle = currentAngle + Math.max(0, Math.min(100, item.pct)) * 3.6;
                    currentAngle = endAngle;
                    if (endAngle <= startAngle) return "";
                    const color = item.color || chartColors[index % chartColors.length];
                    const title = tooltipText(item, scoreLabel);
                    return `<path d="${pieSlicePath(80, 80, 70, startAngle, endAngle)}" fill="${color}" style="cursor: help;"><title>${title}</title></path>`;
                }).join("");

                return `
                    <svg viewBox="0 0 160 160" style="width: clamp(120px, 28vw, 180px); aspect-ratio: 1; flex-shrink: 0; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));">
                        <circle cx="80" cy="80" r="70" fill="#2a2a2a"></circle>
                        ${slices}
                    </svg>
                `;
            }

            function renderCandidateList(stepData, displayCount) {
                if (!stepData || !stepData.candidates || stepData.candidates.length === 0) {
                    return `
                        <div style="color: #666; font-size: 14px; padding: 10px 0;">No candidate data. / 暂无候选数据。</div>
                    `;
                }

                const visibleCandidates = stepData.candidates.slice(0, displayCount || stepData.candidates.length);
                const maxProb = Math.max(...visibleCandidates.map(c => c.prob || 0), 0.000001);
                const candidateItems = visibleCandidates.map((candidate, index) => {
                    let tokenText = candidate.token;
                    if (tokenText === undefined || tokenText === null) tokenText = '';
                    if (String(tokenText).trim() === '') tokenText = '␣';
                    return {
                        text: tokenText,
                        pct: (candidate.prob || 0) * 100,
                        score: Number(candidate.score),
                        color: index === 0 ? '#ff6b6b' : '#4dabf7',
                        barWidth: Math.max(2, Math.min(100, ((candidate.prob || 0) / maxProb) * 100))
                    };
                });

                const rows = candidateItems.map(item => {
                    const safeToken = escapeHtml(item.text);
                    const probPct = item.pct.toFixed(2);
                    const scoreText = Number.isFinite(item.score) ? item.score.toFixed(2) : '';
                    const title = tooltipText(item, 'score');
                    return `
                        <div title="${title}" class="data-row">
                            <div class="data-label" title="${title}">${safeToken}</div>
                            <div class="data-score">${scoreText}</div>
                            <div title="${title}" class="data-bar-bg">
                                <div class="data-bar-fill" style="width: ${item.barWidth}%; background: ${item.color};"></div>
                            </div>
                            <div class="data-pct">${probPct}%</div>
                        </div>
                    `;
                }).join('');

                return `<div class="data-table" style="${tokenColumnStyle(candidateItems)}">${rows}</div>`;
            }

            function setHoverTarget(nextHoverTarget) {
                if (hoverTarget !== nextHoverTarget) {
                    hoverTarget = nextHoverTarget;
                    updateView();
                }
            }

            function activeInspectionTarget() {
                return hoverTarget !== null ? hoverTarget : pinnedTarget;
            }

            function tokenIndexFromElement(el) {
                const tokenEl = el && el.closest ? el.closest('.token') : null;
                if (!tokenEl || !container.contains(tokenEl) || tokenEl.classList.contains('hidden-token')) return null;
                const tokenIndex = parseInt(tokenEl.id.replace('token-', ''));
                return Number.isFinite(tokenIndex) ? tokenIndex : null;
            }

            document.querySelectorAll('.token').forEach((tokenEl) => {
                tokenEl.addEventListener('click', (e) => {
                    const tokenIndex = tokenIndexFromElement(e.target);
                    if (tokenIndex === null) return;
                    e.stopPropagation();
                    pinnedTarget = tokenIndex;
                    hoverTarget = tokenIndex;
                    updateView();
                });
            });

            container.addEventListener('mouseover', (e) => {
                const tokenIndex = tokenIndexFromElement(e.target);
                if (tokenIndex !== null) {
                    setHoverTarget(tokenIndex);
                }
            });

            container.addEventListener('mouseout', (e) => {
                const tokenIndex = tokenIndexFromElement(e.target);
                if (tokenIndex === null) return;
                const relatedToken = tokenIndexFromElement(e.relatedTarget);
                if (relatedToken === tokenIndex) return;
                if (hoverTarget !== null) {
                    hoverTarget = null;
                    updateView();
                }
            });

            container.addEventListener('mouseleave', () => {
                if (hoverTarget !== null) {
                    hoverTarget = null;
                    updateView();
                }
            });

            container.addEventListener('click', () => {
                if (pinnedTarget !== null || hoverTarget !== null) {
                    pinnedTarget = null;
                    hoverTarget = null;
                    updateView();
                }
            });

            function updateView() {
                slider.value = currentStep;
                container.classList.toggle('interaction-ready', isInteractionReady());
                
                // Clear SVG and redefine arrow markers
                svgLayer.innerHTML = `
                    <defs>
                        <marker id="arrow-in" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="10" markerHeight="10" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6b6b" />
                        </marker>
                        <marker id="arrow-out" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="10" markerHeight="10" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4dabf7" />
                        </marker>
                        <marker id="dot-in" markerUnits="userSpaceOnUse" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="6" markerHeight="6">
                            <circle cx="3" cy="3" r="3" fill="#ff6b6b" />
                        </marker>
                        <marker id="dot-out" markerUnits="userSpaceOnUse" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="6" markerHeight="6">
                            <circle cx="3" cy="3" r="3" fill="#4dabf7" />
                        </marker>
                        <marker id="dot-gray" markerUnits="userSpaceOnUse" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="6" markerHeight="6">
                            <circle cx="3" cy="3" r="3" fill="#666666" />
                        </marker>
                    </defs>
                `;
                
                // Remove highlights from previous steps
                for(let i=0; i<data.tokens.length; i++) {
                    const el = document.getElementById('token-' + i);
                    el.classList.remove('active-target', 'active-source', 'future-token', 'pinned-target', 'hover-target');
                    if (pinnedTarget === i) {
                        el.classList.add('pinned-target');
                    }
                    if (hoverTarget === i) {
                        el.classList.add('hover-target');
                    }
                    
                    // Faint out future tokens based on current step, unless the user is inspecting a token.
                    if (activeInspectionTarget() === null && i >= data.prompt_length + currentStep + 1) {
                        el.classList.add('future-token');
                    }
                }

                const inspectionTarget = activeInspectionTarget();
                const isHoverMode = (inspectionTarget !== null);
                let chartItems = [];
                let selectedStepData = null;

                // Draw lines for all steps up to currentStep
                const lastStepToDraw = isHoverMode ? data.steps.length - 1 : currentStep;
                for (let s = 0; s <= lastStepToDraw; s++) {
                    if (s >= data.steps.length) continue;
                    
                    const stepData = data.steps[s];
                    const isCurrent = (s === currentStep);
                    const targetIdx = stepData.generated_token_idx;
                    
                    const targetEl = document.getElementById('token-' + targetIdx);
                    if (!targetEl) continue;
                    
                    let isTargetCurrent = false;
                    if (isCurrent && !isHoverMode) {
                        isTargetCurrent = true;
                        selectedStepData = stepData;
                        targetEl.classList.add('active-target');
                        
                        // Populate chart for currently generating token
                        stepData.attributions.forEach(attr => {
                            const srcEl = document.getElementById('token-' + attr.src);
                            if (srcEl && !srcEl.classList.contains('hidden-token')) {
                                chartItems.push({ labelId: attr.src, pct: attr.pct, score: attr.score });
                            }
                        });
                    }

                    const targetRect = targetEl.getBoundingClientRect();
                    const containerRect = container.getBoundingClientRect();

                    stepData.attributions.forEach(attr => {
                        const srcIdx = attr.src;
                        const srcEl = document.getElementById('token-' + srcIdx);
                        if (!srcEl || srcEl.classList.contains('hidden-token')) return;
                        
                        const isSystemSource = srcEl.classList.contains('role-system');
                        let isThickBright = false;
                        let isThinBright = false;

                        if (isHoverMode) {
                            if (targetIdx === inspectionTarget) {
                                selectedStepData = stepData;
                                isThickBright = true;
                                targetEl.classList.add('active-target');
                                srcEl.classList.add('active-source');
                                chartItems.push({ labelId: attr.src, pct: attr.pct, score: attr.score });
                            } else if (srcIdx === inspectionTarget) {
                                isThinBright = true;
                                srcEl.classList.add('active-target');
                                // Do not push outgoing influences to the chart
                            }
                        } else if (isTargetCurrent) {
                            isThickBright = true;
                            srcEl.classList.add('active-source');
                        }
                        
                        if (isSystemSource && !isThickBright && !isThinBright) return;
                        
                        const srcRect = srcEl.getBoundingClientRect();
                        const x1 = srcRect.left + srcRect.width/2 - containerRect.left;
                        const y1 = srcRect.bottom - containerRect.top - 3;
                        const x2 = targetRect.left + targetRect.width/2 - containerRect.left;
                        const y2 = targetRect.bottom - containerRect.top - 3;

                        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        
                        const isSameLine = Math.abs(y1 - y2) < 15;
                        if (isSameLine) {
                            const midX = (x1 + x2) / 2;
                            const dist = Math.abs(x2 - x1);
                            const depth = Math.min(dist * 0.3, 40); // curve downwards
                            const midY = Math.max(y1, y2) + depth;
                            path.setAttribute('d', `M ${x1} ${y1} Q ${midX} ${midY} ${x2} ${y2}`);
                        } else {
                            path.setAttribute('d', `M ${x1} ${y1} L ${x2} ${y2}`);
                        }
                        
                        path.setAttribute('class', 'attr-path');
                        path.setAttribute('fill', 'none');
                        
                        if (isThickBright) {
                            path.setAttribute('stroke', '#ff6b6b');
                            path.setAttribute('stroke-width', 1 + attr.weight * 4);
                            path.setAttribute('stroke-opacity', 0.3 + attr.weight * 0.7);
                            path.setAttribute('marker-end', 'url(#arrow-in)');
                            path.setAttribute('marker-start', 'url(#dot-in)');
                        } else if (isThinBright) {
                            path.setAttribute('stroke', '#4dabf7');
                            path.setAttribute('stroke-width', 2);
                            path.setAttribute('stroke-opacity', 0.6);
                            path.setAttribute('marker-end', 'url(#arrow-out)');
                            path.setAttribute('marker-start', 'url(#dot-out)');
                        } else {
                            path.setAttribute('stroke', '#666666');
                            path.setAttribute('stroke-width', 1);
                            path.setAttribute('stroke-opacity', isHoverMode ? 0.02 : 0.1);
                            path.setAttribute('marker-start', 'url(#dot-gray)');
                        }
                        
                        if (isHoverMode || !isTargetCurrent) {
                            path.style.animation = 'none';
                            path.style.strokeDasharray = 'none';
                        }
                        
                        svgLayer.appendChild(path);
                    });
                }
                
                // Render data reference panel
                if (chartItems.length === 0) {
                    const emptyMessage = `Hover over a generated token, or click it to pin the view.<br><br>将鼠标悬停在已生成 Token 上，或点击固定这个 Token 的数据。`;
                    referenceContent.innerHTML = `
                        <div style="color: #666; font-size: 14px; padding: 10px 0;">
                            ${emptyMessage}
                        </div>
                    `;
                } else {
                    const groupMap = {};
                    let rawSum = 0;
                    chartItems.forEach(item => {
                        const tokenEl = document.getElementById('token-' + item.labelId);
                        if (tokenEl) {
                            let text = data.tokens[item.labelId];
                            if (text === undefined) text = tokenEl.innerText;
                            if (text.trim() === '') text = '␣';
                            else text = text.trim();
                            
                            if (!groupMap[text]) groupMap[text] = { pct: 0, score: 0 };
                            groupMap[text].pct += (item.pct || 0);
                            groupMap[text].score += (item.score || 0);
                            rawSum += (item.pct || 0);
                        }
                    });

                    if (rawSum < 100) {
                        const otherPct = 100 - rawSum;
                        if (otherPct > 0.1) {
                            groupMap['others'] = { pct: otherPct, score: NaN };
                        }
                    } else if (rawSum > 100) {
                        Object.keys(groupMap).forEach(k => {
                            groupMap[k].pct = (groupMap[k].pct / rawSum) * 100;
                        });
                    }

                    const sortedGroups = Object.keys(groupMap).map(k => ({
                        text: k,
                        pct: groupMap[k].pct,
                        score: groupMap[k].score
                    })).sort((a, b) => {
                        if (a.text === 'others') return 1;
                        if (b.text === 'others') return -1;
                        return b.pct - a.pct;
                    });
                    
                    let legendHtml = "";

                    const nonOtherGroups = sortedGroups.filter(g => g.text !== 'others');
                    const maxNonOtherPct = nonOtherGroups.length > 0 ? nonOtherGroups[0].pct : 100;

                    sortedGroups.forEach((g, index) => {
                        g.color = g.text === 'others' ? '#444444' : chartColors[index % chartColors.length];
                        const barWidthPct = g.text === 'others' ? 100 : (maxNonOtherPct > 0 ? (g.pct / maxNonOtherPct * 100) : 0);
                        const safeText = escapeHtml(g.text);
                        const scoreText = Number.isFinite(g.score) ? g.score.toFixed(2) : '';
                        const title = tooltipText(g, 'attribution score');
                        
                        legendHtml += `
                            <div title="${title}" class="data-row">
                                <div class="data-label" title="${title}">${safeText}</div>
                                <div class="data-score">${scoreText}</div>
                                <div title="${title}" class="data-bar-bg">
                                    <div class="data-bar-fill" style="width: ${Math.min(barWidthPct, 100)}%; background: ${g.color};"></div>
                                </div>
                                <div class="data-pct">${g.pct.toFixed(1)}%</div>
                            </div>
                        `;
                    });

                    const attributionHtml = `<div class="data-table" style="${tokenColumnStyle(sortedGroups)}">${legendHtml}</div>`;
                    const candidateCount = sortedGroups.length;
                    const candidateHtml = renderCandidateList(selectedStepData, candidateCount);

                    referenceContent.innerHTML = `
                        <div class="reference-grid">
                            <div class="reference-card attribution-card">
                                <div class="reference-pie">${renderPieSvg(sortedGroups, 'attribution score')}</div>
                                <div class="reference-list">
                                    <div class="reference-list-title">Gradient attribution / 逆梯度归因</div>
                                    ${attributionHtml}
                                </div>
                            </div>
                            <div class="reference-card reference-list">
                                <div class="reference-list-title">Top candidate tokens / 候选 Token 排名</div>
                                ${candidateHtml}
                            </div>
                        </div>
                    `;
                }
            }

            document.getElementById('btn-next').onclick = () => {
                if (currentStep < readyStep) {
                    stopPlayback();
                    setCurrentStep(currentStep + 1);
                    updateView();
                }
            };
            document.getElementById('btn-prev').onclick = () => {
                if (currentStep > -1) {
                    stopPlayback();
                    setCurrentStep(currentStep - 1);
                    updateView();
                }
            };
            document.getElementById('btn-play').onclick = () => {
                isPlaying = !isPlaying;
                if (isPlaying) {
                    playInterval = setInterval(() => {
                        if (currentStep < readyStep) {
                            setCurrentStep(currentStep + 1);
                            updateView();
                            if (isInteractionReady()) {
                                stopPlayback();
                            }
                        } else {
                            stopPlayback();
                        }
                    }, 200); // Match the 200ms arrow animation so each step completes before the next one.
                } else {
                    stopPlayback();
                }
            };

            // Init
            updateView();
        </script>
    </body>
    </html>
    """
    
    json_data = json.dumps({
        "tokens": tokens,
        "prompt_length": prompt_length,
        "steps": steps_data
    })
    
    html_content = html_template.replace("DATA_PLACEHOLDER", json_data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    output_files = sorted(
        [name for name in os.listdir(OUTPUT_DIR) if name.endswith(".html")],
        reverse=True,
    )
    return render_template("index.html", output_files=output_files)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    try:
        max_tokens = int(data.get("max_tokens", 60))
    except (TypeError, ValueError):
        return {"error": "Invalid input. / 输入无效。"}, 400
    
    if not prompt or max_tokens < 15:
        return {"error": "Invalid input. / 输入无效。"}, 400

    return Response(generate_stream(prompt, max_tokens), mimetype='text/event-stream')

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True)
