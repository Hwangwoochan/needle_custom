# Needle Custom 변경 메모

작성일: 2026-05-22  
브랜치: `custom-korean-local-finetune`

이 문서는 원본 `cactus-compute/needle`에서 갈라진 `needle_custom` 실험 변경사항을 추적하기 위한 메모다. 코드 변경의 의도, 영향 범위, 확인 방법을 함께 남겨서 이후 실험과 발표 자료 작성에 사용할 수 있게 한다.

## 현재 변경 요약

### 1. 한국어 학술 리뷰 노트 추가

커밋:

- `a7814bf Add Korean academic review notes`

파일:

- `docs/needle_academic_review_ko.md`

내용:

- Needle의 문제 정의
- 모델 구조
- 학습 데이터와 가중치 검토
- function-calling 평가 방식
- 관련 연구
- 한계와 발표용 핵심 메시지

목적:

- Needle 자체 논문이 없는 상황에서 코드/모델/데이터 artifact 중심으로 발표와 리뷰를 준비하기 위함

### 2. Playground finetune 데이터 생성을 한국어로 변경

커밋:

- `f1867fd Generate Korean finetune data from playground`

파일:

- `needle/ui/server.py`
- `needle/dataset/generate.py`

변경 내용:

- UI finetune 데이터 생성 시 언어를 `Korean`으로 고정
- 생성된 JSONL을 한국어가 그대로 보이도록 `ensure_ascii=False`로 저장
- `Korean`을 데이터 생성 언어 목록에 추가

기존 동작:

```python
gd.LANGUAGES = ["English"]
```

변경 후:

```python
_CUSTOM_FINETUNE_LANGUAGE = "Korean"
gd.LANGUAGES = [_CUSTOM_FINETUNE_LANGUAGE]
```

목적:

- Playground에서 사용자가 tool schema를 넣고 finetune할 때, 한국어 query/answer 데이터가 생성되도록 하기 위함

### 3. 생성된 전체 JSONL 저장

커밋:

- `f1867fd Generate Korean finetune data from playground`

파일:

- `needle/ui/server.py`

변경 내용:

- Playground finetune에서 Gemini가 생성한 전체 데이터셋을 train/val/test split 전 상태로 보존
- 저장 위치:

```text
checkpoints/generated_data/
```

파일명 예:

```text
needle_custom_korean_generated_20260522_153012.jsonl
```

finetune 성공 후 zip bundle에도 다음 파일을 포함:

```text
generated.jsonl
```

목적:

- 생성 데이터 품질을 직접 확인하기 위함
- 한국어 query가 제대로 생성되는지 확인하기 위함
- 잘못된 tool call, hallucinated argument, duplicate query 등을 분석하기 위함

## 현재 실행 흐름

Playground에서 `Finetune on these tools`를 누르면 다음 순서로 동작한다.

1. 왼쪽 `Tools JSON` 읽기
2. Gemini API key 사용
3. tool당 120개 샘플 생성 시도
4. 한국어 데이터 생성 prompt 사용
5. 생성된 전체 JSONL을 `checkpoints/generated_data/`에 저장
6. 데이터 validation
7. train/val/test split
8. 기존 `needle.pkl`에서 finetune
9. base model과 finetuned model 평가
10. checkpoint와 데이터 bundle zip 생성

## 생성 데이터 확인 방법

finetune을 한 번 실행한 뒤 다음 디렉터리를 확인한다.

```bash
ls checkpoints/generated_data
```

최근 생성 파일 보기:

```bash
tail -n 5 checkpoints/generated_data/needle_custom_korean_generated_*.jsonl
```

JSONL 한 줄 예시 형태:

```json
{"query":"10분 타이머 맞춰줘","tools":"[...]","answers":"[{\"name\":\"set_timer\",\"arguments\":{\"time_human\":\"10분\"}}]","language":"Korean"}
```

## 주의할 점

- tool 이름과 parameter key는 영어 API identifier로 유지하는 것이 좋다.
- tool description과 parameter description은 한국어/영어를 병기해도 된다.
- 생성 데이터가 한국어로 나오더라도 모델의 기본 tokenizer와 pretraining 분포가 한국어에 최적화되어 있다고 보장할 수는 없다.
- 한국어 성능은 생성 데이터 품질과 tool schema 명확성에 크게 좌우된다.
- generated JSONL을 반드시 확인해서 query와 answer가 실제로 대응되는지 봐야 한다.

## 추천 tool schema 작성법

좋은 예:

```json
{
  "name": "set_timer",
  "description": "사용자가 말한 시간 또는 종료 시각으로 타이머를 설정합니다. Set a timer for a duration or target end time.",
  "parameters": {
    "time_human": {
      "type": "string",
      "description": "사용자가 말한 타이머 시간. 예: '10분', '1시간 30분', '오후 3시까지'",
      "required": true
    }
  }
}
```

요령:

- `name`은 snake_case 영어 유지
- `parameters` key도 영어 유지
- `description`에는 한국어 사용 상황과 예시 포함
- required/optional을 명확히 표시
- 비슷한 tool이 있으면 차이를 description에 명확히 적기

## 다음 작업 후보

1. UI에서 언어 선택 드롭다운 추가
   - Korean / English / Mixed 선택 가능하게 만들기

2. UI에서 Gemini model 선택 가능하게 만들기
   - 예: `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`

3. 생성 데이터 preview 기능 추가
   - finetune 시작 전에 JSONL 일부를 화면에서 확인

4. 생성만 하고 학습은 하지 않는 모드 추가
   - 데이터 품질 검수 후 별도 finetune 가능하게 만들기

5. local LLM teacher endpoint 추가
   - Ollama, vLLM, llama.cpp server 등 OpenAI-compatible endpoint 지원

6. 한국어 평가 샘플 세트 추가
   - 고정 test set을 만들어 base vs finetuned 비교

## Git 원격 정보

```text
origin   https://github.com/Hwangwoochan/needle_custom.git
upstream https://github.com/cactus-compute/needle.git
```

작업 브랜치:

```text
custom-korean-local-finetune
```

