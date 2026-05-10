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
                step_attributions.append({"src": int(idx), "weight": weight, "pct": pct})
                
        steps_data.append({
            "generated_token_idx": prompt_length + step,
            "attributions": step_attributions
        })
        
        # Terminate generation early to prevent <|im_end|> from entering the visualization
        if next_token_id == tokenizer.eos_token_id:
            break
            
        # Append to sequence
        tokens.append(next_token_str)
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)
        
        # Cleanup tensors
        del inputs_embeds
        del outputs
        del logits
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
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
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
                width: 960px;
                flex-shrink: 0;
                text-align: left;
                padding: 20px;
                background: #1e1e1e;
                border-radius: 10px;
                white-space: pre-wrap;
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
            
            #chart-container {
                background: #1a1a1a;
                padding: 20px;
                border-radius: 10px;
                min-height: 100px;
                flex: 1;
                min-width: 300px;
                position: sticky;
                top: 20px;
            }
            #chart-container h3 {
                margin-top: 0;
                color: #a0c4ff;
                font-size: 16px;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
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
            
            svg {
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
                animation: dash 0.6s ease-out forwards;
            }
            @keyframes dash {
              to {
                stroke-dashoffset: 0;
              }
            }
        </style>
    </head>
    <body>
        <div class="controls" style="display: flex; align-items: center; width: 100%; max-width: 1000px;">
            <button id="btn-prev">Previous / 上一步</button>
            <button id="btn-play">Play / Pause / 播放/暂停</button>
            <button id="btn-next">Next / 下一步</button>
            <input type="range" id="timeline-slider" min="-1" max="0" value="-1" style="margin-left: 20px; flex-grow: 1;">
        </div>
        <div style="display: flex; gap: 30px; align-items: flex-start; max-width: 1500px; width: 100%;">
            <div class="container" id="text-container"><svg id="svg-layer"></svg></div>
            <div id="chart-container">
                <h3>Token Influence Breakdown / Token 影响分布</h3>
                <div id="chart-content">Hover over a token or play the animation to see influence scores. / 将鼠标悬停在 Token 上，或播放动画查看影响分布。</div>
            </div>
        </div>

        <script>
            const data = DATA_PLACEHOLDER;
            const container = document.getElementById('text-container');
            const svgLayer = document.getElementById('svg-layer');
            const slider = document.getElementById('timeline-slider');
            let currentStep = -1;
            let isPlaying = false;
            let playInterval;

            slider.max = data.steps.length;
            slider.addEventListener('input', (e) => {
                currentStep = parseInt(e.target.value);
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

            let hoverTarget = null;

            // Hover Events
            container.addEventListener('mouseover', (e) => {
                if (currentStep >= data.steps.length) {
                    const el = e.target.closest('.token');
                    if (el) {
                        hoverTarget = parseInt(el.id.replace('token-', ''));
                        updateView();
                    }
                }
            });

            container.addEventListener('mouseout', (e) => {
                if (currentStep >= data.steps.length) {
                    const el = e.target.closest('.token');
                    if (el) {
                        if (e.relatedTarget && el.contains(e.relatedTarget)) {
                            return;
                        }
                        hoverTarget = null;
                        updateView();
                    }
                }
            });

            function updateView() {
                slider.value = currentStep;
                
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
                    el.classList.remove('active-target', 'active-source', 'future-token');
                    
                    // Faint out future tokens based on current step
                    if (i >= data.prompt_length + currentStep + 1) {
                        el.classList.add('future-token');
                    }
                }

                const isHoverMode = (currentStep >= data.steps.length && hoverTarget !== null);
                let chartItems = [];

                // Draw lines for all steps up to currentStep
                for (let s = 0; s <= currentStep; s++) {
                    if (s >= data.steps.length) continue;
                    
                    const stepData = data.steps[s];
                    const isCurrent = (s === currentStep);
                    const targetIdx = stepData.generated_token_idx;
                    
                    const targetEl = document.getElementById('token-' + targetIdx);
                    if (!targetEl) continue;
                    
                    let isTargetCurrent = false;
                    if (isCurrent && !isHoverMode) {
                        isTargetCurrent = true;
                        targetEl.classList.add('active-target');
                        
                        // Populate chart for currently generating token
                        stepData.attributions.forEach(attr => {
                            const srcEl = document.getElementById('token-' + attr.src);
                            if (srcEl && !srcEl.classList.contains('hidden-token')) {
                                chartItems.push({ labelId: attr.src, pct: attr.pct });
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
                            if (targetIdx === hoverTarget) {
                                isThickBright = true;
                                targetEl.classList.add('active-target');
                                srcEl.classList.add('active-source');
                                chartItems.push({ labelId: attr.src, pct: attr.pct });
                            } else if (srcIdx === hoverTarget) {
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
                
                // Render Influence Bar Chart
                const chartContent = document.getElementById('chart-content');
                if (chartItems.length === 0) {
                    chartContent.innerHTML = `
                        <div style="display: flex; align-items: center; justify-content: center; margin-top: 15px; padding: 10px;">
                            <div style="width: 160px; height: 160px; border-radius: 50%; background: #2a2a2a; margin-right: 40px; box-shadow: inset 0 4px 12px rgba(0,0,0,0.2); flex-shrink: 0; display: flex; align-items: center; justify-content: center; color: #555; font-size: 14px; text-align: center;">No data<br>无数据</div>
                            <div style="flex-grow: 1; max-width: 300px; color: #666; font-size: 14px;">
                                Hover over a token or play the animation to see influence scores.<br><br>将鼠标悬停在 Token 上，或点击播放动画来查看影响分布。
                            </div>
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
                            
                            if (!groupMap[text]) groupMap[text] = 0;
                            groupMap[text] += (item.pct || 0);
                            rawSum += (item.pct || 0);
                        }
                    });

                    if (rawSum < 100) {
                        const otherPct = 100 - rawSum;
                        if (otherPct > 0.1) {
                            groupMap['Other / 其他 (long tail / 长尾影响)'] = otherPct;
                        }
                    } else if (rawSum > 100) {
                        Object.keys(groupMap).forEach(k => {
                            groupMap[k] = (groupMap[k] / rawSum) * 100;
                        });
                    }

                    const sortedGroups = Object.keys(groupMap).map(k => ({
                        text: k,
                        pct: groupMap[k]
                    })).sort((a, b) => {
                        if (a.text === 'Other / 其他 (long tail / 长尾影响)') return 1;
                        if (b.text === 'Other / 其他 (long tail / 长尾影响)') return -1;
                        return b.pct - a.pct;
                    });
                    
                    let conicParts = [];
                    let currentAcc = 0;
                    const colors = ['#ff6b6b', '#4dabf7', '#95d5b2', '#ffd166', '#a0c4ff', '#e0e0e0', '#ff9f1c', '#b5179e', '#f72585', '#4361ee', '#7209b7'];
                    let legendHtml = "";

                    const nonOtherGroups = sortedGroups.filter(g => g.text !== 'Other / 其他 (long tail / 长尾影响)');
                    const maxNonOtherPct = nonOtherGroups.length > 0 ? nonOtherGroups[0].pct : 100;

                    sortedGroups.forEach((g, index) => {
                        const isOther = g.text === 'Other / 其他 (long tail / 长尾影响)';
                        const color = isOther ? '#444444' : colors[index % colors.length];
                        const start = currentAcc;
                        currentAcc += g.pct;
                        const end = currentAcc;
                        
                        conicParts.push(`${color} ${start}% ${end}%`);
                        
                        let barWidthPct = 0;
                        if (isOther) {
                            barWidthPct = 100;
                        } else {
                            barWidthPct = maxNonOtherPct > 0 ? (g.pct / maxNonOtherPct * 100) : 0;
                        }
                        
                        legendHtml += `
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <div style="width: 80px; color: #e0e0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 14px; margin-right: 12px; flex-shrink: 0;" title="${g.text}">${g.text}</div>
                                <div style="flex-grow: 1; height: 8px; background: #2a2a2a; border-radius: 4px; overflow: hidden; margin-right: 15px;">
                                    <div style="width: ${Math.min(barWidthPct, 100)}%; height: 100%; background: ${color}; border-radius: 4px; transition: width 0.3s ease;"></div>
                                </div>
                                <div style="width: 45px; text-align: right; font-weight: bold; color: #fff; font-size: 14px; flex-shrink: 0;">${g.pct.toFixed(1)}%</div>
                            </div>
                        `;
                    });

                    const gradient = conicParts.join(', ');

                    chartContent.innerHTML = `
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 15px; padding: 10px;">
                            <div style="width: 140px; height: 140px; border-radius: 50%; background: conic-gradient(${gradient}); margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); flex-shrink: 0;"></div>
                            <div style="width: 100%; max-height: 400px; overflow-y: auto; padding-right: 5px;">
                                ${legendHtml}
                            </div>
                        </div>
                    `;
                }
            }

            document.getElementById('btn-next').onclick = () => {
                if (currentStep < data.steps.length) {
                    currentStep++;
                    updateView();
                }
            };
            document.getElementById('btn-prev').onclick = () => {
                if (currentStep > -1) {
                    currentStep--;
                    updateView();
                }
            };
            document.getElementById('btn-play').onclick = () => {
                isPlaying = !isPlaying;
                if (isPlaying) {
                    playInterval = setInterval(() => {
                        if (currentStep < data.steps.length) {
                            currentStep++;
                            updateView();
                        } else {
                            clearInterval(playInterval);
                            isPlaying = false;
                        }
                    }, 800); // 600ms animation + 200ms pause
                } else {
                    clearInterval(playInterval);
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
