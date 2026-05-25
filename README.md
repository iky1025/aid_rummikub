# AID Rummikub — PPO + ILP

루미큐브(조커 제외 단순화 버전)를 PPO 에이전트가 학습하고, 그리디 ILP 솔버를 상대로 대결하는 프로젝트.

- 에이전트: PPO (Proximal Policy Optimization)
- 상대 봇: greedy ILP (매 턴 손패 사용량 최대화)
- 후보 생성: ILP가 PPO에게 매 턴 가능한 수 N개를 후보로 제공
- 액션: PPO가 후보 중 하나를 고르거나 "드로우" 선택

---

## 1. 게임 정의

### 1.1 타일 구성

- 색상 4종: R, B, Y, K
- 숫자 13종: 1 ~ 13
- 같은 (색, 숫자) 타일이 2장씩
- 조커 없음 (단순화 위해 제거)
- 총 타일 수: 4 × 13 × 2 = **104장**

### 1.2 유효 세트

- **Run**: 같은 색, 연속 숫자, 길이 ≥ 3 (예: R3 R4 R5)
- **Group**: 같은 숫자, 서로 다른 색, 크기 3 또는 4 (예: R7 B7 Y7)

### 1.3 게임 진행

1. 게임 시작 시 두 플레이어에게 각각 14장 분배 (덱에서)
2. PPO 차례
   - 손패와 테이블을 보고 ILP가 가능한 수 후보를 생성
   - PPO가 후보 중 하나 선택하거나 "드로우"
   - 후보 선택 시: 손패에서 타일을 빼서 테이블 세트 구성 (테이블 전체 재배열 허용)
   - 드로우 선택 시: 덱에서 1장 가져옴
3. ILP(상대) 차례
   - greedy ILP가 자기 손패 사용량 최대화하는 수 1개 선택해서 실행
   - 낼 수 없으면 드로우
4. 다시 PPO 차례 → 반복

### 1.4 종료 조건

- 누군가의 손패가 0장이면 즉시 종료 (승/패 결정)
- 또는 100턴 도달 시 타임아웃

---

## 2. 코드 구조

```
aid_rummikub/
├── rummikub_solver.py    # 타일/세트 정의 + ILP 솔버
├── rummikub_env.py       # 단일 플레이어 환경 (덱/손패/테이블)
├── ppo_env.py            # gymnasium 호환 PPO vs ILP 환경
├── ppo_model.py          # Actor-Critic 신경망
├── train_ppo.py          # PPO + VecEnv 학습 스크립트
├── eval_ppo.py           # 학습된 모델 평가 스크립트
├── main.py               # 사람이 직접 돌리는 인터랙티브 데모
├── test_ppo_env.py       # 환경 단위 테스트
├── test_ppo_model.py     # 모델 단위 테스트
└── README.md
```

### 2.1 의존성 계층

```
train_ppo.py / eval_ppo.py
        ↓
   ppo_env.py (gym.Env)
        ↓
   rummikub_env.py
        ↓
   rummikub_solver.py
```

`ppo_model.py`는 `train_ppo.py`와 `eval_ppo.py`에서만 사용.

---

## 3. ILP 솔버 (`rummikub_solver.py`)

루미큐브의 핵심 의사결정을 정수 선형 계획 (ILP)으로 모델링.

### 3.1 결정 변수

`x_i ∈ {0, 1}` — i번째 후보 세트를 사용할지 여부.

후보 세트는 `generate_all_valid_sets()`로 모듈 로드 시 1회 생성 (Run + Group 약 285개). 매 호출에서는 `filter_available_sets()`로 현재 손패+테이블 타일로 만들 수 있는 후보만 추림.

### 3.2 제약

1. 각 일반 타일 사용량 ≤ 보유량
2. **기존 테이블 타일은 반드시 재사용** (테이블에서 타일이 사라지면 안 됨)
3. (옵션) 손패 최소 1장은 써야 함 — PPO 후보 생성 시 사용

### 3.3 목적함수

`maximize (총 사용 타일 수 - 기존 테이블 타일 수) = 새로 낸 손패 타일 수`

→ **매 턴 손패를 최대한 많이 내는 그리디 ILP**.

### 3.4 다중 해 생성 (`solve_many`)

PPO에게 후보를 N개 주기 위해 사용

1. ILP로 최적해 1개 찾음
2. "이 해를 제외" 제약 추가: `Σx_i ≤ |selected|-1` (선택된 후보 중 하나는 빠져야 함)
3. 다시 ILP 풀기 → 같은 최적값을 가지는 대안 해 또는 차선해 반환
4. 최대 N=10번 반복

### 3.5 솔버 선택

- PuLP의 `COIN_CMD(threads=1)` 사용
- conda-forge의 pulp에는 PULP_CBC_CMD가 번들되지 않아 conda-installed `cbc` 바이너리를 PATH에서 자동 인식
- threads=1로 CBC 자체 멀티스레딩 비활성 (VecEnv와 코어 경쟁 방지)

---

## 4. RL 환경 (`ppo_env.py`)

gymnasium.Env 호환. SubprocVecEnv에 들어가서 멀티프로세스로 실행됨.

### 4.1 Observation (Dict)

```python
observation_space = spaces.Dict({
    "state":      Box(0, 10, shape=(106,)),       # 내 손패/테이블/덱/상대패
    "cand_feats": Box(0, 10, shape=(10, 104)),    # 후보별 next-state
    "mask":       Box(0, 1, shape=(11,)),         # 유효 action 마스크
})
```

**`state`** (106차원) — R5 적용

| 슬라이스 | 차원 | 의미 |
|---|---|---|
| `[0:52]` | 52 | 내 손패의 타일 카운트 벡터 (값 / 2.0) |
| `[52:104]` | 52 | 테이블 전체 타일 카운트 벡터 |
| `[104:105]` | 1 | 덱 잔여 비율 (deck_count / 104) |
| `[105:106]` | 1 | 상대 손패 수 정규화 (ilp_hand / 14) |

**`cand_feats`** (10 × 104) — 각 후보를 적용한 다음 상태

| 슬라이스 | 차원 | 의미 |
|---|---|---|
| `[i, 0:52]` | 52 | 후보 i 적용 후의 내 손패 |
| `[i, 52:104]` | 52 | 후보 i 적용 후의 테이블 |

후보가 10개 미만이면 0으로 패딩 (mask로 무효 표시).

**`mask`** (11차원) — 유효한 액션은 1, 무효는 0
- `mask[0:10]`: 후보 슬롯
- `mask[10]`: "드로우" 액션 (항상 1)

### 4.2 Action Space

`Discrete(11)`. 인덱스 0~9는 후보, 10은 드로우.

### 4.3 환경 인터페이스 (gymnasium 표준)

```python
obs, info = env.reset(seed=...)
obs, reward, terminated, truncated, info = env.step(action)
```

- `terminated`: 누군가 손패 비움 (승/패)
- `truncated`: 100턴 도달 (타임아웃)
- `info["outcome"]`: `"win" / "loss" / "timeout"`

### 4.4 보상 구조 (다음 라운드 계획 반영)

| 이벤트 | 보상 | 비고 |
|---|---|---|
| PPO가 타일 n장 냄 | `+0.1n` | 매 후보 선택 시 |
| 매 턴 시간 페널티 | `-0.01` | 무한 드로우 방지 |
| 드로우 (덱 있음) | `-0.5` | 손패 늘어남 페널티 |
| 드로우 (덱 없음) | `-1.0` | 정말 할 게 없을 때 |
| 상대가 타일 n장 냄 | `-0.02n` | 상대 진행을 약간 페널티 |
| **승리** | `+5.0 + 0.3 × ilp_hand_remaining` | 상대 잔여 많으면 압승 보너스 |
| **패배** | `-5.0 - 0.3 × ppo_hand_remaining` | 내 잔여 적으면 박빙 |
| **타임아웃** | `+0.3 × (ilp_hand - ppo_hand)` | 진행 중이던 상황 반영 |

**제거된 신호** (R5에서 제거)

- ~~매 턴 손패 크기 페널티 `-0.02 × hand_size`~~
  - 이유: 매 턴 일정한 압박이 "역전 시도(큰 콤보 기다리기)"를 불가능하게 만듦
  - 종료 시점 margin 보상으로 옮김

### 4.5 보상 설계 의도

세 가지 행동 양상을 정책이 **상태에 따라 동적으로** 학습하게 함

| 상태 | 권장 행동 | 보상 신호 |
|---|---|---|
| PPO 패 적음, 상대 패 적음 (비등 종반) | 안전한 마무리, 빠르게 비움 | 승리 +5 ~ +9 |
| PPO 패 많음, 상대 패 적음 (압패 직전) | 역전 노림, 큰 콤보 기다림 | 박빙 패배 -5 vs 압패 -9 차이로 보상 |
| PPO 패 적음, 상대 패 많음 (압승 직전) | 어떻게든 빨리 끝 | 승리 +5 + 큰 보너스 |

→ obs에 상대 손패 정보가 있어야 이 동적 정책이 표현 가능. 그래서 4.1의 NEW 차원.

---

## 5. 신경망 모델 (`ppo_model.py`)

### 5.1 구조

```
state (106) ─→ state_encoder (MLP 128) ─→ state_emb (128)
                                              │
                                              ├──→ critic (Linear) ─→ V(s)
                                              │
                                              ├──→ draw_head (Linear, bias=-2.0)
                                              │       └─→ draw_logit
                                              │
                                              └──→ concat with each cand_emb
                                                          │
cand_feats (10, 104) ─→ cand_encoder (MLP 128) ─→ cand_emb (10, 128)
                                                          │
                                                  score_head (MLP) ─→ cand_logits (10)

최종 logits = concat([cand_logits, draw_logit]) → (11,)
```

### 5.2 Action 점수화

- 각 후보 i에 대해 `score_i = MLP([state_emb, cand_emb_i])` — (state, action) 쌍을 직접 점수화
- 드로우는 state만 보는 별도 헤드 `draw_logit = draw_head(state_emb)`
- **인덱스 의존 없음** — 어떤 후보가 슬롯 0에 들어가든 점수는 그 내용에 의해 결정 (permutation-invariant)

### 5.3 Draw head bias 초기화

```python
nn.init.constant_(self.draw_head.bias, -2.0)
```

학습 초기 draw 확률을 ~12%로 강제 시작. "기본값 드로우" 함정 방지.

### 5.4 Mask 적용

```python
masked_logits = logits.clone()
masked_logits[mask == 0] = -1e9
dist = Categorical(logits=masked_logits)
```

무효 액션은 사실상 확률 0. PPO의 surrogate loss는 mask된 분포에서 계산.

---

## 6. 학습 파이프라인 (`train_ppo.py`)

### 6.1 Vectorized Environment

stable-baselines3의 `SubprocVecEnv` 차용

```python
env_fns = [make_env_fn(seed + i) for i in range(n_envs)]
vec_env = SubprocVecEnv(env_fns, start_method="forkserver")
```

- 8~10개 worker process가 각자 RummikubPPOEnv 실행
- CBC subprocess 호출이 여러 코어에서 동시 진행 → 학습 속도 2-2.5x
- update당 episode 수 증가 (12 → 30+) → gradient 노이즈 √2배 감소

### 6.2 PPO 알고리즘

표준 PPO with clipping. 직접 구현 (sb3의 PPO는 사용 안 함, VecEnv만 차용).

학습 한 update의 흐름

```
1. n_steps 동안 vec_env에서 rollout 수집
   - 매 step: model.forward_actor(state, cand_feats) → logits
   - mask 적용 후 Categorical 샘플 → action
   - vec_env.step(action) → next_obs, reward, done, info
   - buffer에 (state, cand_feats, mask, action, reward, done, log_prob, value) 저장

2. 마지막 상태에서 V(s) bootstrap

3. Per-env GAE 계산 (env마다 별도 trajectory)

4. flatten (n_steps, n_envs, ...) → (n_steps * n_envs, ...)

5. ppo_epochs 반복:
   - 미니배치별로 forward → loss → backward → step
   - actor_loss: clipped surrogate (clip_range=0.1)
   - critic_loss: MSE
   - entropy_loss: 음의 entropy (탐색 보너스)
   - total = actor + 0.1 * critic - 0.01 * entropy

6. LR scheduler.step() — LinearLR로 점진 감소

7. CSV/콘솔 로깅
8. best 갱신 시 best.pt 저장, save_every마다 model.pt 저장
```

### 6.3 Hyperparameter (기본값)

| 파라미터 | 값 | 비고 |
|---|---|---|
| `n_envs` | 10 | M-series 10코어 활용 |
| `n_steps` | 128 | env당 rollout 길이 |
| update당 총 경험 | 1280 steps | n_envs × n_steps |
| `total_updates` | 100 | |
| `batch_size` | 128 | PPO 미니배치 |
| `ppo_epochs` | 4 | 한 rollout 데이터로 4 epoch |
| `gamma` | 0.99 | 할인 |
| `gae_lambda` | 0.95 | GAE |
| `clip_range` | 0.1 | PPO clip |
| `value_coef` | 0.1 | critic loss 가중치 |
| `entropy_coef` | 0.01 | entropy 보너스 가중치 |
| `lr_init` | 3e-4 | Adam 초기 lr |
| `lr_final` | 3e-5 | LinearLR 종료 시 lr |
| `hidden_dim` | 128 | MLP 은닉층 (모델 내부) |

### 6.4 출력 파일

- `rummikub_ppo_model.pt` — 매 N updates마다 + 마지막에 저장
- `rummikub_ppo_best.pt` — `avg_episode_reward`가 최대일 때 저장
- `train_log.csv` — update별 통계

`--tag <suffix>`로 파일명 끝에 접미사 붙여 충돌 방지 가능.

### 6.5 로깅 컬럼 (R5 업데이트)

```text
update, elapsed_sec, steps_total, episodes,
win_rate, loss_rate, timeout_rate,
avg_episode_reward, avg_episode_length,
draw_action_ratio, forced_draw_ratio, chosen_draw_ratio,   # R5
avg_candidate_count,
avg_win_margin, avg_loss_margin, expected_score,           # R5
actor_loss, critic_loss, entropy,
lr, best_avg_reward
```

콘솔 출력 한 줄 예시 (R5)

```text
upd= 10/100 t= 980s steps=12800 eps=31 W/L/T=14/17/0 rew= -4.20 len=41.2
              draw=0.52(f=0.50 c=0.02) wm= 3.8 lm= 1.2 es=+1.65
              a=-0.001 cl=5.8 ent=0.71 lr=2.5e-04 best= -4.20
```

핵심 신규 지표

- `draw=0.52(f=0.50 c=0.02)`: 전체 드로우 / 강제 드로우 / 선택 드로우
  - **forced**: 유효 후보 0개 → 어쩔 수 없는 드로우
  - **chosen**: 유효 후보 있지만 정책이 선택한 드로우
- `wm` (win_margin): 이긴 게임에서 상대 잔여 손패 평균
- `lm` (loss_margin): 진 게임에서 내 잔여 손패 평균
- `es` (expected_score): 게임당 평균 net margin = "이긴 만큼 - 진 만큼"

---

## 7. 평가 (`eval_ppo.py`)

### 7.1 모드

- **Deterministic (argmax)**: 매 상태에서 logit 최대 액션. 실제 배포 시 기준 성능
- **Stochastic (sample)**: 분포에서 샘플. 학습 분포 그대로 검증

```bash
python eval_ppo.py --model rummikub_ppo_best.pt --episodes 100
python eval_ppo.py --model rummikub_ppo_best.pt --episodes 100 --stochastic
```

### 7.2 지표 (R5 업데이트)

- `win_rate`, `loss_rate`, `timeout_rate`
- `avg_reward` (보상 평균)
- `avg_steps` (게임당 턴 수)
- `draw_ratio` (드로우 액션 비율) — `forced` / `chosen` 분리 표시
- **`avg_win_margin`** — 이긴 게임에서 상대의 잔여 손패 평균
- **`avg_loss_margin`** — 진 게임에서 내 잔여 손패 평균
- **`expected_score`** — (Σ win_margins - Σ loss_margins) / episodes

해석 예시

```text
episodes        : 100
wins            : 48 (48.0%)
  avg margin    : 5.3 (opponent tiles left)
losses          : 52 (52.0%)
  avg margin    : 1.8 (own tiles left)
expected_score  : +1.61
draw_ratio      : 0.53
  forced        : 0.50
  chosen        : 0.03
```

→ 승률 48%여도 expected_score = +1.61로 양수면 "잘 이기고 잘 진다" — 강한 정책.
→ forced=0.50, chosen=0.03이면 드로우의 대부분은 게임 구조상 불가피함.

### 7.3 표본 크기 권장

- 20판: ±22% 신뢰구간 (참고용)
- 50판: ±14%
- 100판: ±10% (권장)
- 200판: ±7% (논문급)

---

## 8. 실행 방법

### 8.1 환경 설정

conda 환경 사용 (pip 혼합 X)

```bash
# 환경 생성
conda create -n rummikub python=3.11 -y

# 라이브러리 설치 (pytorch, numpy, pulp, sb3)
conda install -n rummikub -c pytorch -c conda-forge \
    pytorch numpy pulp stable-baselines3 -y

# 활성화
conda activate rummikub
```

MPS 지원 확인

```python
import torch
print(torch.backends.mps.is_available())  # True여야 함
```

### 8.2 학습

기본 (10 envs, 100 updates)

```bash
python train_ppo.py
```

옵션

```bash
python train_ppo.py \
    --n-envs 10 \
    --n-steps 128 \
    --total-updates 100 \
    --batch-size 128 \
    --ppo-epochs 4 \
    --lr-init 3e-4 \
    --lr-final 3e-5 \
    --clip-range 0.1 \
    --seed 42 \
    --tag round1
```

체크포인트에서 이어서 (보수적 lr 권장)

```bash
python train_ppo.py \
    --resume rummikub_ppo_best.pt \
    --lr-init 1e-4 \
    --lr-final 1e-5 \
    --tag round2
```

백그라운드 실행

```bash
nohup python train_ppo.py > run.log 2>&1 &
tail -f run.log
```

### 8.3 평가

```bash
# Deterministic
python eval_ppo.py --model rummikub_ppo_best.pt --episodes 100

# Stochastic (학습 분포 그대로)
python eval_ppo.py --model rummikub_ppo_best.pt --episodes 100 --stochastic

# 매 에피소드 결과 출력
python eval_ppo.py --model rummikub_ppo_best.pt --episodes 50 --verbose
```

### 8.4 인터랙티브 데모 (사람용)

```bash
python main.py
```

테이블 세트를 직접 입력 → ILP가 후보 10개 제시 → 사용자가 선택해서 적용.

---

## 9. 설계 결정 / 알려진 한계

### 9.1 단순화 사항 (실제 루미큐브와 차이)

- 조커 없음 (행동 공간 단순화, 학습 신호 명료화)
- 30점 초기 등록 룰 없음 (모든 등록 자유)
- 점수제 — 실제 게임 점수 대신 자체 보상 (대신 종료 시 margin으로 근사)
- 4인 게임 아니라 1:1
- 시간 제한 없음 (대신 100턴 타임아웃)

### 9.2 ILP 후보 생성의 한계

- PPO가 고를 수 있는 행동은 **ILP의 그리디 해**에서 파생된 N개로 제한
- "후보에 없는 더 좋은 수"는 PPO가 학습으로 발견 불가
- 다만 ILP의 약점(타일 다양성 무시, 멀티턴 무계획)을 정책 선택으로 부분 보완 가능

### 9.3 학습 신호 약점

- 12-30 episodes/update의 표본 노이즈
- ILP 상대가 매 턴 최적해 → 비대칭 게임 → 승률 천장이 50% 근처
- 보상이 매 턴 dense + 종료 sparse 혼합 → tuning 민감

### 9.4 성능 최적화 적용된 것

- ILP `solve_many`에서 deepcopy 제거 (next state는 ILPResult 필드로 직접 계산)
- CBC threads=1 (멀티 envs와의 코어 경쟁 방지)
- SubprocVecEnv로 ILP subprocess 호출 병렬화
- MPS device 사용 (배치된 forward에서 효과)

### 9.5 미해결 이슈

- Mac 절전 모드 시 학습 시간 급증 가능 — 학습 중 절전 차단 권장 (caffeinate 등)
- ILP가 큰 후보군에서 풀 때 30-50ms 걸려 rollout 병목 — 추가 최적화 여지

---

## 10. 라운드별 진척 요약

| 라운드 | 주요 변경 | 결과 | 다음 단계 |
|---|---|---|---|
| 초기 | 조커 있음, 단순 PPO | - | 조커 제거, deepcopy 제거 |
| R1 | 조커 X, deepcopy X, 보상 1/10 | best -3.5 (12 ep 노이즈), 100판 평가 46% | LR schedule, clip 0.1, 손패 페널티 |
| R2 | LR schedule, clip 0.1, 손패 페널티 | best -10.4 (upd 46), 평가 46%/55% | n_steps↑ |
| R3 | n_steps=1024 | 시간 4.5시간, plateau ~47% | VecEnv |
| R4 | SubprocVecEnv (10 envs) | 학습 2배 빠름, 30 ep/update, 100판 평가 49%/52% (det/stoch) | 보상 재설계 |
| **R5 (현재)** | 상대 패 obs, draw bias=-2.0, margin 보상, per-turn 손패 페널티 제거, forced/chosen draw 분리 측정 | 진행 예정 | 결과 분석 후 결정 |

### R5에서 측정으로 검증된 발견 ⭐

Smoke test (4 envs × 64 steps × 2 updates)에서 즉시 드러난 사실

```text
upd 1: draw=0.50(f=0.48 c=0.02) wm=3.3 lm=1.0 es=+2.25
upd 2: draw=0.54(f=0.52 c=0.02) wm=4.0 lm=1.0 es=+2.12
```

#### 발견 1: draw_ratio ~0.5의 96%는 강제 드로우

- 이전 R1~R4 라운드에서 plateau로 보인 `draw=0.53`
- **정책의 문제가 아님** — 게임 본질
- 손패 + 테이블로 유효 세트를 못 만드는 상황이 평균 ~50%
- 정책이 의도적으로 드로우하는 비율(`chosen`)은 단 2-3%
- 따라서 "드로우를 더 줄이기" 방향의 개선은 효과 미미할 것

#### 발견 2: Margin 신호가 정책 차별화에 적합

- R1~R4 평가는 모두 win_rate 46~52%에서 통계적으로 구별 불가
- R5에서 `expected_score` 측정으로 "잘 이기고 잘 진다"를 정량화
- 작은 표본(4-8 게임)에서도 `es=+2.25`로 정책의 강함이 보임
- 향후 라운드 간 비교는 `es`를 1순위로 봐야 함

### R5의 알려진 우려 사항 / 검증 필요 항목

#### 1. 체크포인트 호환성 깨짐

- `STATE_DIM` 105 → 106으로 변경
- 모든 R1~R4 모델 파일 (`rummikub_ppo_best.pt` 등)을 R5 코드로 **로드 불가**
- `--resume`도 R5 이전 체크포인트엔 사용 불가
- 백업이 필요하면 학습 시작 전 미리 rename:

  ```bash
  mv rummikub_ppo_best.pt rummikub_ppo_best_r4.pt
  mv rummikub_ppo_model.pt rummikub_ppo_model_r4.pt
  mv train_log.csv train_log_r4.csv
  ```

#### 2. 보상 magnitude 변화

- 매 턴 손패 페널티 (`-0.02 × hand_size`) 제거로 reward 절대값이 작아짐
- R4: `rew ≈ -15` (큰 음수, 손패 페널티 누적)
- R5: `rew ≈ -3 ~ -6` 예상
- → 절대값 직접 비교 X. **상대 변화/expected_score 위주로 봐야 함**
- critic_loss도 초기 6-25 범위 (이전 5-10보다 살짝 큼) — 정상

#### 3. Reward shaping과 정책 안정성

- 종료 시점 큰 보상(±5~±9)이 dense 보상보다 영향이 큼
- gamma=0.99로 40턴 할인하면 종료 보상 가치가 약 0.67배 → 여전히 의미 있음
- 다만 초기 학습 시 critic이 새 보상 분포 학습하느라 actor 신호 약할 수 있음
- 5~10 updates는 critic 적응 기간으로 봐야 함

#### 4. Timeout margin 처리

- Timeout 시 PPO가 앞서 있으면 `win_margin > 0`, 뒤지면 `loss_margin > 0`
- 학습 보상: `+0.3 × (ilp_hand - ppo_hand)` (부호 있음, 합리적)
- 로그 표시: `avg_win_margin`은 wins-only, `avg_loss_margin`은 losses-only, `expected_score`만 timeout 포함
- timeout 비율이 보통 0%라 영향 미미, 다만 0% 아니어도 부정확하지 않게 분리 추적

#### 5. forced_draw 측정의 정확도

- `mask[:max_candidates].sum() == 0` 인 시점에 드로우 액션을 선택했는가로 판정
- mask는 PPO 결정 직전 상태 기반 → **정확함**
- 단 후보 0개이지만 mask[10]=1 (드로우 가능)으로 강제 드로우는 항상 가능

#### 6. Draw head bias 학습 가능성

- 초기 bias=-2.0이지만 학습이 진행되며 bias도 업데이트됨
- 만약 정책이 "draw가 항상 좋다"고 학습하면 bias가 다시 0 이상으로 올라갈 수 있음
- chosen_draw가 시간이 갈수록 늘어나면 정상적인 학습 (큰 콤보 대기 발견)
- 만약 chosen_draw가 0.3+로 증가하면 의심 — 정책이 과도하게 보수적

#### 7. 학습 곡선 해석 변경 필요

- R1~R4의 win_rate 위주 분석은 더 이상 충분치 않음
- `expected_score` 우상향이 진짜 학습 신호
- `wm` 증가 (이길 때 압승 정도 ↑)
- `lm` 감소 (질 때 박빙 정도 ↑)
- 가능하면 윈도우 평균(5 updates)으로 추세 판단

#### 8. ILP 후보 다양성 한계

- max_candidates=10에서 평균 후보 수는 2.5-3.5
- "draw 0.5의 chosen 부분 = 0.02"는 후보가 부족해서 어쩔 수 없는 측면 있음
- 만약 ILP가 더 다양한 후보를 줄 수 있다면 chosen draw가 증가할 가능성
- 현 단계에서는 max_candidates를 늘리지 않음 (다음 라운드 검토 항목)

---

## 11. 빠른 참고

### 11.1 자주 쓰는 명령

```bash
# 학습 시작
conda activate rummikub && python train_ppo.py

# 평가 (100판)
python eval_ppo.py --model rummikub_ppo_best.pt --episodes 100

# 학습 로그 빠르게 보기
tail -10 train_log.csv | column -ts,

# Best 갱신 추적
awk -F, 'NR==1 || $NF != prev {print; prev=$NF}' train_log.csv | column -ts,
```

### 11.2 Best 정책 사용 (사람과 게임)

학습된 best 모델로 사람과 대결하려면 `main.py`를 확장하거나, `eval_ppo.py --verbose`로 정책의 행동을 관찰.

### 11.3 디버깅 모드

VecEnv 대신 단일 프로세스로

```bash
python train_ppo.py --no-subproc --n-envs 1 --total-updates 2 --n-steps 32
```
