# Chain of Draft
[[Paper]](https://arxiv.org/abs/2502.18600)

Large Language Models (LLMs) have demonstrated remarkable performance in solving complex reasoning tasks through mechanisms like Chain-of-Thought (CoT) prompting, which emphasizes verbose, step-by-step reasoning. 
However, humans typically employ a more efficient strategy: drafting concise intermediate thoughts that capture only essential information. 
In this work, we propose Chain of Draft (CoD), a novel paradigm inspired by human cognitive processes, where LLMs generate minimalistic yet informative intermediate reasoning outputs while solving tasks. 
By reducing verbosity and focusing on critical insights, CoD matches or surpasses CoT in accuracy while using as little as only 7.6% of the tokens, significantly reducing cost and latency across various reasoning tasks.


## Usage
To run evaluation:
```bash
python evaluate.py \
    --task gsm8k \      # task: gsm8k, date, sports, or coin_flip
    --model gpt-4o \    # model to evaluate, supports only gpt-* and claude-* models
    --prompt cod \      # prompting strategy: baseline, cod, cot 
    --fewshot 5 \       # optional, # of fewshot examples to include in prompt, include all examples by default if omitted
```
The evaluation results will be stored under `./results/`.

All prompts and fewshot examples are stored under `./configs/{task}-{prompt}.yaml`. 

## Citation
```latex
@article{xu2025cod,
  title={Chain of Draft: Thinking Faster by Writing Less},
  author={Xu, Silei and Xie, Wenhao and Zhao, Lingxiao and He, Pengcheng},
  journal={arXiv preprint arXiv:2502.18600},
  year={2025}
}
```