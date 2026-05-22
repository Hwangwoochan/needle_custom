# Needle 학술 리뷰 노트

작성일: 2026-05-22  
대상 프로젝트: Cactus Compute, Needle  
리뷰 관점: 논문이 없는 오픈소스 모델 아티팩트에 대한 코드, 데이터, 학습 파이프라인, 평가 타당성 검토

## 1. 요약

Needle은 Cactus Compute가 공개한 26M 파라미터 규모의 function-calling 전용 소형 모델이다. 일반적인 대화형 LLM이 아니라, 사용자 질의와 사용 가능한 tool schema를 입력받아 JSON 형식의 tool call을 생성하는 데 특화되어 있다. 공식 설명에 따르면 Gemini 3.1의 tool-calling 동작을 증류했고, 16 TPU v6e에서 200B token 사전학습 후 2B token 규모의 단일 function-call 데이터로 후학습했다.

학술 논문은 현재 확인되지 않는다. 따라서 본 리뷰는 논문 리뷰라기보다 공개 코드, 모델 카드, 학습 데이터 생성 코드, 가중치 배포 방식, 평가 코드에 대한 artifact review로 보는 것이 적절하다.

핵심 주장은 다음과 같다.

- Function calling은 일반 추론보다 "tool retrieval + argument extraction + JSON assembly"에 가깝다.
- 따라서 대형 decoder-only LLM 대신, 작은 encoder-decoder 모델로도 충분히 높은 성능을 낼 수 있다.
- tool schema 자체가 외부 지식원 역할을 하므로, FFN/MLP를 제거한 attention 중심 구조가 가능하다.
- 온디바이스 환경에서 낮은 지연시간과 낮은 메모리 사용량을 목표로 한다.

## 2. 문제 정의

Needle이 해결하려는 문제는 다음과 같이 정식화할 수 있다.

입력:

- 자연어 사용자 질의 `q`
- 사용 가능한 tool 목록 `T = {t_1, ..., t_n}`
- 각 tool의 이름, 설명, 파라미터 schema

출력:

- 호출할 tool 이름
- 각 tool에 전달할 argument dictionary
- JSON array 형식의 structured output

예:

```json
[
  {
    "name": "get_weather",
    "arguments": {
      "location": "Paris"
    }
  }
]
```

이 문제는 일반적인 자연어 생성보다 제약이 강하다. 출력 공간은 주어진 tool schema에 의해 제한되고, 모델은 자유로운 대화 응답이 아니라 실행 가능한 structured command를 생성해야 한다.

## 3. 모델 아키텍처

Needle의 코드상 모델은 `SimpleAttentionNetwork`로 구현되어 있다. 구조적으로는 encoder-decoder Transformer 계열이다.

주요 구성:

- Encoder: 입력 질의와 tool schema를 처리
- Decoder: `<tool_call>` 이후 JSON tool call을 생성
- Shared embedding
- Tied output projection
- RoPE positional encoding
- GQA, grouped-query attention
- ZCRMSNorm
- Gated residual connection
- 기본 설정에서 FFN/MLP 제거

코드 기준 주요 위치:

- `needle/model/architecture.py`
- `TransformerConfig`
- `SimpleAttentionNetwork`
- `Encoder`, `Decoder`, `MultiHeadAttention`

공식 README/Hugging Face 모델 카드의 대표 구성은 다음과 같다.

| 항목 | 값 |
|---|---|
| 파라미터 수 | 약 26M |
| hidden dimension | 512 |
| vocabulary | 8192 SentencePiece BPE |
| encoder layers | 12 |
| decoder layers | 8 |
| attention | 8 query heads / 4 KV heads |
| precision | bfloat16, training 중 INT4 QAT |
| FFN | 기본적으로 제거 |

### 3.1 Encoder 입력 형식

코드상 encoder 입력은 다음과 같이 구성된다.

```text
[query tokens] + [<tools>] + [tool schema tokens]
```

즉, tool schema는 prompt/context처럼 encoder에 직접 제공된다. 이는 Needle의 핵심 가정과 연결된다. 모델 내부에 모든 tool knowledge를 저장하기보다, 사용 가능한 tool definition을 입력으로 받아 해당 schema 안에서 선택과 추출을 수행한다.

### 3.2 Decoder 출력 형식

Decoder는 `[EOS]`를 prefix로 받고, 이후 `<tool_call>` 및 JSON answer token을 예측하도록 학습된다.

학습 데이터 생성 시 decoder 입력과 target은 대략 다음 구조를 따른다.

```text
decoder input:  [EOS, <tool_call>, answer tokens]
decoder target: [<tool_call>, answer tokens, EOS]
```

이 구조는 일반 대화 생성보다 structured generation에 맞춰져 있다.

## 4. Simple Attention Network 관점

Needle의 아키텍처 설명에서 가장 특이한 지점은 FFN/MLP 제거다. 표준 Transformer block은 attention sublayer와 FFN sublayer로 구성된다. Needle은 function calling에서는 FFN이 저장하는 일반적 지식보다, 입력으로 주어진 tool schema와 query 사이의 matching이 중요하다고 본다.

이 주장은 다음 조건에서 설득력을 가진다.

- 사용 가능한 tool 목록이 inference 시점에 명시적으로 주어진다.
- 출력은 tool 이름과 argument 값으로 제한된다.
- 필요한 정보 대부분은 query와 tool schema 안에 존재한다.
- 모델은 장문 reasoning보다 retrieval, copying, schema alignment를 수행한다.

반대로 다음 조건에서는 약해질 수 있다.

- tool 설명이 모호하거나 부족한 경우
- multi-step planning이 필요한 경우
- tool 실행 결과를 보고 다음 action을 정해야 하는 경우
- 사용자 발화에 암묵적 지식이나 장기 기억이 필요한 경우
- schema 밖의 domain knowledge가 필요한 경우

따라서 Needle은 "general agent model"이라기보다 "on-device tool router / function-call generator"로 이해하는 것이 적절하다.

## 5. 학습 데이터

Needle의 데이터 파이프라인은 크게 두 갈래다.

### 5.1 사전학습

코드상 `needle pretrain`은 `PleIAs/SYNTH`를 streaming dataset으로 사용한다. 입력은 `query`, `query_seed_text`, `synthetic_answer` 필드를 기반으로 구성된다.

목적은 일반적인 sequence-to-sequence mapping 능력과 작은 encoder-decoder 모델의 기본 언어/복사 능력을 확보하는 것으로 해석할 수 있다.

코드 위치:

- `needle/training/pretrain.py`
- `_HF_PRETRAIN_REPO = "PleIAs/SYNTH"`

### 5.2 Function-calling 후학습

후학습용 tool-call 데이터는 Gemini를 사용해 합성하는 방식이다. 코드상 기본 생성 모델은 다음과 같다.

```python
MODEL = "gemini-3.1-flash-lite-preview"
```

데이터 생성 코드는 다양한 tool pool, query style, language, speech error, no-tool case, multi-call case를 포함하도록 prompt를 구성한다.

생성 예시는 다음 필드를 가진 JSONL 형태다.

```json
{
  "query": "Turn off the lights",
  "tools": "[{\"name\":\"toggle_lights\", ...}]",
  "answers": "[{\"name\":\"toggle_lights\",\"arguments\":{\"state\":\"off\"}}]"
}
```

코드 위치:

- `needle/dataset/generate.py`
- `needle/dataset/dataset.py`
- `needle/dataset/tokenize.py`

### 5.3 데이터 공개성 이슈

공식 README는 weights와 dataset generation이 공개되어 있다고 설명한다. 하지만 실제 학습에 사용된 raw dataset과 tokenized dataset은 코드상 다음 Hugging Face repo를 참조한다.

- `Cactus-Compute/tool-calls`
- `Cactus-Compute/tokenized-tool-calls`

코드는 이 저장소 접근에 `token=True`를 사용한다. 리뷰 시점에는 공개 검색에서 해당 dataset이 명확히 확인되지 않았고, 접근 가능성이 모델 가중치만큼 명확하지 않다. 따라서 "데이터 생성 코드 공개"와 "실제 학습 데이터 완전 공개"는 구분해야 한다.

학술 리뷰에서는 이 부분을 재현성의 주요 한계로 표시하는 것이 타당하다.

## 6. 학습 방법

학습 루프는 JAX/Flax/Optax 기반이다.

주요 특징:

- packed sequence training
- segment id 기반 block-diagonal attention mask
- weighted token loss
- tool name, argument value, argument key에 다른 loss weight 적용
- Muon optimizer와 AdamW 혼합
- 주기적 INT4 quantization-aware training
- contrastive retrieval auxiliary loss

### 6.1 Token-level loss weighting

Tool call 생성에서 모든 token이 같은 중요도를 갖지는 않는다. JSON boilerplate보다 tool name, argument key, argument value가 중요하다. Needle은 answer JSON을 분석해 token class를 부여하고, 학습 시 class별 weight를 적용한다.

예:

- base JSON token
- tool name token
- argument key token
- argument value token

이 설계는 function calling task에 맞춘 실용적인 inductive bias로 볼 수 있다.

### 6.2 Contrastive auxiliary objective

학습 코드에는 query와 positive tool을 가까이 배치하는 CLIP-style contrastive loss도 포함되어 있다. 이는 tool retrieval 성능을 보조하려는 목적이다.

다만 학습 루프상 contrastive loss는 매 step 사용되는 것이 아니라 주기적으로 적용된다. 이 보조 목적이 최종 성능에 얼마나 기여하는지는 공개 ablation 없이는 판단하기 어렵다.

## 7. 추론과 constrained decoding

Needle은 greedy decoding을 기본으로 사용한다. 추론 시 tool name과 argument key에 대해 constrained decoding을 적용할 수 있다.

구현 방식:

- tool name trie 구성
- parameter key trie 구성
- JSON generation state machine으로 현재 위치 추적
- `"name":"..."` 구간에서는 가능한 tool name token만 허용
- `"arguments":{"..."}` 구간에서는 해당 tool의 parameter key만 허용

이는 structured output 품질을 높이는 데 유용하지만, 완전한 JSON schema validator는 아니다.

한계:

- argument value의 type correctness는 강제하지 않음
- required parameter completeness는 강제하지 않음
- nested JSON schema 전체를 지원하지 않음
- OpenAI-style `parameters.properties` 구조와 코드 구현 사이에 불일치 가능성

따라서 constrained decoding은 "부분적 grammar/lexical constraint"로 이해해야 한다.

## 8. 평가 코드와 평가 타당성

Needle은 다음 metric을 계산한다.

- JSON parse rate
- exact match
- tool name precision/recall/F1
- call precision/recall/F1
- argument accuracy
- parameter hallucination
- parameter missing rate
- value accuracy
- throughput

평가 코드는 `needle/training/eval.py`와 `needle/training/train.py`에 있다.

### 8.1 장점

- 단순 exact match만 보지 않고 tool name, argument, hallucinated parameter를 나누어 본다.
- single-call과 multi-call sample을 분리해 평가하려는 코드가 있다.
- packed validation perplexity와 structured generation metric을 함께 본다.

### 8.2 한계

- 프로젝트 자체 논문이 없어 benchmark 구성과 비교 조건이 충분히 문서화되어 있지 않다.
- best checkpoint 선택이 single-call `call_f1` 중심이다.
- training-time generation eval은 constrained decoding을 끈 상태로 수행된다.
- 공식 주장인 FunctionGemma, Qwen, Granite, LFM 대비 성능 우위의 독립 재현 자료가 부족하다.
- no-call, ambiguous-call, adversarial schema, long tool list, multi-turn tool use에 대한 공개 평가가 부족하다.

## 9. 보안 및 배포상 고려

Needle의 공개 가중치는 Hugging Face에서 pickle 형태로 배포된다. 코드의 `load_checkpoint` 역시 `pickle.load`를 직접 호출한다.

pickle은 임의 코드 실행 위험이 있으므로 다음 조치가 바람직하다.

- 신뢰 가능한 출처에서만 다운로드
- checksum 또는 commit hash 고정
- sandbox 환경에서 로드
- 가능하면 safetensors, npz 등 safer serialization으로 변환
- production runtime에는 pickle loader를 포함하지 않기

온디바이스 사용을 목표로 하는 모델이라면 이 부분은 특히 중요하다.

## 10. 관련 연구

Needle 자체 논문은 확인되지 않는다. 관련 연구는 다음 축으로 정리할 수 있다.

### 10.1 Tool use 학습

**Toolformer**  
Schick et al., 2023. "Toolformer: Language Models Can Teach Themselves to Use Tools."  
LLM이 API call의 사용 여부, API 선택, argument, 결과 활용을 학습하는 대표 연구다. Needle과의 차이는 Toolformer가 일반 LLM의 tool-use 능력을 다루는 반면, Needle은 function-calling 자체를 작은 특화 모델로 분리한다는 점이다.

### 10.2 Function-calling 평가

**Berkeley Function Calling Leaderboard (BFCL)**  
Function-calling 모델의 tool selection, parameter extraction, multi-turn setting을 평가하는 대표 벤치마크 계열이다. Needle의 성능 주장을 검증하려면 BFCL류의 독립 benchmark가 필요하다.

### 10.3 Action model / function-calling model

**xLAM**  
Salesforce의 Large Action Model 계열 연구로, agent action과 function calling을 위한 모델과 데이터 구축을 다룬다. Needle은 xLAM류보다 훨씬 작은 모델을 목표로 한다.

### 10.4 Tool-agentic dataset synthesis

**TOUCAN**  
MCP 환경에서 tool-agentic trajectory를 합성하는 데이터셋 연구다. Needle의 Gemini 합성 데이터와 비교하면, TOUCAN은 실제 MCP 환경과 multi-turn trajectory 쪽에 더 초점을 둔다.

## 11. Needle의 학술적 의의

Needle의 의의는 새로운 foundation model을 제안했다기보다, agent system에서 반복적으로 발생하는 function routing 문제를 별도 소형 모델로 분리했다는 점에 있다.

학술적으로는 다음 질문을 제기한다.

1. Tool calling은 일반 reasoning과 분리 가능한가?
2. Function selection과 argument extraction은 어느 정도까지 소형 모델로 충분한가?
3. Tool schema가 외부 지식원으로 제공될 때 FFN 없는 Transformer가 얼마나 잘 작동하는가?
4. On-device agent architecture에서 local router와 cloud reasoner를 어떻게 나눌 것인가?
5. Synthetic teacher data만으로 robust tool-calling behavior를 만들 수 있는가?

이 질문들은 Needle을 단순한 작은 모델이 아니라, agent architecture의 모듈화 관점에서 의미 있게 만든다.

## 12. 주요 한계

1. 논문 부재
   - 연구 가설, 실험 설계, ablation, benchmark 세부 조건이 충분히 문서화되어 있지 않다.

2. 데이터 재현성 부족
   - 생성 코드는 공개되어 있지만 실제 학습 데이터 접근성은 명확하지 않다.

3. 독립 평가 부족
   - README와 모델 카드의 비교 성능 주장을 독립적으로 재현할 자료가 부족하다.

4. Synthetic data bias
   - Gemini-generated data에 강하게 의존하므로 teacher model의 표현, 오류, 선호가 student에 전이될 수 있다.

5. Schema robustness 한계
   - nested schema, complex argument type, strict required field validation은 제한적이다.

6. Multi-turn agentic behavior 한계
   - Needle은 single-shot function-call 모델에 가깝고, tool result를 보고 다음 action을 계획하는 agent loop는 별도 시스템이 필요하다.

7. Pickle 가중치 배포
   - 연구용으로는 편리하지만 production/security 관점에서는 부적절하다.

## 13. 발표용 핵심 메시지

발표에서는 다음 5문장으로 요약할 수 있다.

1. Needle은 대화형 LLM이 아니라 function-calling 전용 26M 소형 encoder-decoder 모델이다.
2. 핵심 가정은 tool calling이 deep reasoning보다 schema retrieval, argument extraction, JSON assembly에 가깝다는 것이다.
3. 이 가정 때문에 Needle은 FFN 없는 attention 중심 구조를 채택하고, tool schema를 입력 context로 사용한다.
4. 다만 학술 논문이 없고, 실제 학습 데이터와 독립 평가의 공개성이 제한적이므로 artifact-level 검증이 필요하다.
5. Needle의 의의는 "모든 agent 작업을 큰 LLM에 맡기지 말고, 반복적 tool routing은 작은 on-device model로 분리할 수 있다"는 시스템 설계 방향을 제시한 데 있다.

## 14. 발표 슬라이드 구성안

1. 배경: 왜 function calling이 중요한가
2. 문제 정의: query + tools -> JSON function call
3. Needle 개요: 26M on-device tool-calling model
4. 아키텍처: encoder-decoder, shared embedding, no FFN
5. 학습 데이터: PleIAs/SYNTH pretrain + Gemini synthetic tool calls
6. 학습 전략: packed training, weighted loss, QAT, contrastive loss
7. 추론: greedy decoding + constrained decoding
8. 평가: exact match, call F1, argument accuracy, hallucination
9. 한계: 논문 없음, 데이터 재현성, pickle, benchmark 부족
10. 관련 연구: Toolformer, BFCL, xLAM, TOUCAN
11. 결론: 작은 tool router와 큰 reasoner의 역할 분리

## 15. 참고 자료

- Needle GitHub: https://github.com/cactus-compute/needle
- Needle Hugging Face model card: https://huggingface.co/Cactus-Compute/needle
- Toolformer: https://arxiv.org/abs/2302.04761
- BFCL technical report: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-184.html
- xLAM paper page: https://huggingface.co/papers/2409.03215
- TOUCAN: https://arxiv.org/abs/2510.01179

