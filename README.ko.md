# AID Rummikub

[English](README.md) · **한국어**

**2인 루미큐브에서 최적의 한 턴 베이스라인을, 증류된 forward-only 정책 네트워크로 이긴다.**

히든 정보 2인 루미큐브(조커 없음; 4색 × 13숫자 × 2벌 = 104타일)를 두는 학습 에이전트로, **그리디 ILP 베이스라인을 유의하게 이깁니다** — 추론 시 탐색을 전혀 쓰지 않고요. 우승 정책은 탐색 기반 교사로부터 **DAgger**로 증류한 ~94K 파라미터 네트워크입니다.

> **핵심 결과:** forward-only 네트워크가 1,000 미러페어 게임에서 그리디 베이스라인 대비 **67.8%** 승률(50%로부터 ≈17σ), 랜덤 상대에게도 일반화(69.6%), 그리고 ablation에 의해 그 우위가 솔버의 후보 생성이 아니라 **네트워크의 수 선택**에서 옴이 규명됩니다.

---

## TL;DR

- **문제.** 탐색(몬테카를로 롤아웃 + 엔드게임 lookahead)은 이 게임을 잘 두지만 실전용으론 너무 느립니다. 탐색 교사를 순진하게 모방(오프폴리시 behavior cloning)하면 covariate shift로 **붕괴**합니다.
- **처방.** **DAgger** — 학생이 직접 두어 자기 상태 분포를 방문하고, 교사가 학생이 실제로 도달한 각 상태에서 정답 수를 재라벨합니다. 오프폴리시 증류를 침몰시키는 오차 복리 루프를 끊습니다.
- **결과.** 아주 작은 forward-only 네트워크가 교사급에 도달 — 실측상 능가 — 하여, 프로젝트의 *순수 네트워크* 에이전트 목표를 달성합니다.

| 에이전트 | 승률 | pair net | 비고 |
|---|---:|---:|---|
| 그리디 vs 그리디 (sanity) | ~48% | — | 하네스 중립 기준선 |
| 그리디 복사기 상한 | 52.5% | — | 그리디 모방 학생이 도달하는 최대 |
| 탐색 기반 교사 | 56.9% | +0.46 | 정보 일관 롤아웃 + 엔드게임 DFS |
| **오프폴리시 증류 (대조군)** | **28.7%** | −9.12 | 같은 레시피, DAgger 없음 → 붕괴 |
| **DAgger 학생 (160페어)** | **70.3%** | +0.99 | 복사기 상한으로부터 ~7σ |
| **DAgger 학생 (1,000페어)** | **67.8%** | +0.29 | 대규모 확정, 50%로부터 ~17σ |
| 완전 오라클 (손패+덱 치팅) | ~89% | — | 이론 참조값; 덱 운에 지배됨 |

모든 비교는 딜 운을 상쇄하기 위해 **미러페어**(같은 덱, 자리 교대) 평가를 사용합니다. 표준 룰(초기 등록 ≥ 30), 학습/평가 시드는 disjoint.

---

## 왜 이기는가

우위는 솔버의 후보 집합을 고정하고 **선택 함수**만 바꾸는 ablation으로 *분리*됩니다:

| 동일 후보에 대한 선택 방식 | 승률 |
|---|---:|
| 균등 무작위 | 40.6% |
| 그리디 최다타일 휴리스틱 | 46.2% |
| **학습된 네트워크** | **70.3%** |

무작위 선택이 그리디보다 *더 낮으므로*, 후보 집합만으로는 아무 이득이 없습니다 — +24~30%p 격차는 전적으로 네트워크의 학습된 선택입니다. 계측 결과 네트워크는 실제로 능동적이며(결정의 **34%**에서 그리디 최다타일을 재정의, 그중 **19.5%**는 그리디라면 항상 냈을 타일을 전략적으로 *보류*), 그 이탈은 **교사와 정렬**됩니다(교사가 그리디를 벗어날 때 학생도 77% 함께 벗어나며, 정확 일치 49%).

---

## 작동 방식 (파이프라인)

```
                     ┌── 정보 일관 PIMC 롤아웃 (det=8)
1. 교사     ─────────┤
   (느림, 강함)      └── 엔드게임 승리강제 DFS  (+ greedy-margin 가드)

2. DAgger 데이터   학생이 둠 → 자기 상태를 방문 → 교사가 각 상태에서
                   정답 수(+후보별 롤아웃 점수)를 재라벨

3. 증류            soft cross-entropy(교사 롤아웃 마진) + 가치 회귀
                   → forward-only DistillStudent 네트워크 (추론 시 탐색 없음)
```

교사는 오라클을 쓰지 않습니다: 각 후보의 가치를 정보 일관된 그럴듯한 진행을 시뮬레이션해 추정하고, 엔드게임에서는 탐색으로 강제승을 증명합니다. 학생은 그 *결정*들을 빠른 반응 정책으로 증류하며, 노이즈 있는 1-ply 교사의 수별 분산을 평균 냄으로써 교사를 능가합니다 — expert-iteration(ExIt) 효과입니다.

### 네트워크 (`DistillStudent` — 116,919 파라미터; 추론 시 ≈93.5K 사용)

![DistillStudent 구조](docs/architecture.svg)

순열 불변(permutation-invariant) 스코어링 네트워크: 공유 후보 인코더가 각 합법 수를 *(상태, 그 후보)*만으로 점수화하므로, 출력은 후보 순서에 의존하지 않습니다 — 네트워크는 슬롯을 외우는 대신 각 수의 특징을 읽어야 합니다.

| 모듈 | Shape | 파라미터 | 추론 사용 |
|---|---|---:|---|
| `state_encoder` | 108→128→128 | 30,464 | ✅ |
| `cand_encoder` | 104→128→128 (공유 ×20) | 29,952 | ✅ |
| `score_head` | 256→128→1 | 33,025 | ✅ |
| `draw_head` | 128→1 (bias −1.97) | 129 | ✅ |
| `opp_hand_head` | 128→128→52 | 23,220 | 학습 전용 |
| `critic` | 128→1 | 129 | 학습 전용 |

---

## 저장소 구조

| 파일 | 역할 |
|---|---|
| `rummikub_solver.py` | 한 턴 최적화 ILP (`solve`), 후보 다양화 (`solve_many`) |
| `rummikub_dp.py` | 핫패스용 van Rijn & Takes 스타일 DP 솔버 (ILP 대비 ≈26×) |
| `rummikub_env.py` | 게임 상태(덱 / 손패 / 테이블) + 솔버 래퍼 |
| `ppo_env.py` | Gymnasium 환경: 관측, 보상, 상대(그리디 / 랜덤) |
| `ppo_model.py` | `ActorCritic` + `DistillStudent` (상태 인코더 + 후보 인코더 + score head) |
| `rollout_agent.py` | determinized-rollout 교사 + 엔드게임 승리강제 DFS |
| `selfplay_data.py` | 자가대전 / DAgger 데이터 생성 (`--actor student`가 학생 궤적을 재라벨) |
| `distill.py` | 지도 증류 (soft CE + 가치 회귀 + 선택적 보조 헤드) |
| `eval_mirror.py` | 저분산 미러페어 평가 (`--policy student/greedy/rollout/random`) |
| `autopsy_oracle.py` | 패배 게임 부검 — 내 수 분기 DFS로 승리 가능성 증명 |
| `train_ppo.py`, `eval_ppo.py` | PPO 학습 / 평가 (R1–R7 시대) |

R10 전략은 `ROADMAP.md`, 논문 계획은 `docs/paper_chapter_plan.md`, 상세 게임 룰과 관측/행동 인코딩은 `docs/game_rules.md`, HiGHS presolve 버그의 독립 재현은 `docs/highs_presolve_bug/`를 참조하세요.

---

## 빠른 시작

```bash
# 환경 (conda-forge; pulp 솔버는 in-process HiGHS, presolve off 사용)
conda create -n rummikub python=3.11
conda activate rummikub
conda install -c conda-forge numpy pytorch pulp highspy gymnasium
```

헤드라인 평가 재현. 모델(`*.pt`)과 데이터셋(`data/`)은 추적되지 않으므로 파이프라인으로 재생성합니다:

```bash
# 1) DAgger 데이터 생성 (학생이 두고, 교사가 재라벨)
python selfplay_data.py --teacher rollout --consistent --greedy-margin 1.0 --endgame-search \
    --determinizations 8 --rollout-turns 24 --candidate-cap 4 \
    --actor student --actor-model <이전_학생>.pt \
    --pairs 500 --seed 220000 --initial-meld-value 30 --out data/dagger1 --workers 8

# 2) 학생 증류
python distill.py --data data/s1s_dagger1 --tag s1s_dagger1 \
    --epochs 10 --soft-temp 0.3 --value-coef 0.5

# 3) 평가 (미러페어) — 헤드라인 수치
python eval_mirror.py --policy student --model distill_s1s_dagger1.pt \
    --pairs 160 --seed 2000 --initial-meld-value 30 --workers 8

# ablation: 동일 후보, 서로 다른 선택 함수
python eval_mirror.py --policy random --pairs 160 --seed 2000 --initial-meld-value 30 --workers 8
python eval_mirror.py --policy greedy --pairs 160 --seed 2000 --initial-meld-value 30 --workers 8
```

---

## 한계

- **솔버 의존적 후보.** 합법 수 열거는 솔버(규칙 엔진)가 하고, 네트워크는 모든 *선택*을 합니다. 후보까지 생성하는 완전 end-to-end 네트워크는 future work입니다.
- **교사 상한 발견.** 증류는 교사가 시연하지 않은 전략을 발명할 수 없습니다(ExIt 태생 한계).
- **단일 게임 / 룰 / 교사.** 결과는 하나의 교사 설정, meld ≥ 30의 2인 루미큐브에 대한 것입니다. 게임 간 이식은 실증이 아니라 추측입니다.
- **운 요소가 큰 게임.** 딜 분산이 큽니다(그리디도 랜덤 상대를 겨우 ~52%로 이김). 절대 승률 상한은 게임 특화적이며, 그래서 모든 비교에 미러페어 평가를 씁니다.

---

## 참고문헌

- Ross, Gordon & Bagnell (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger).
- Anthony, Tian & Barber (2017). *Thinking Fast and Slow with Deep Learning and Tree Search* (Expert Iteration).
- Long, Sturtevant, Buro & Furtak (2010). *Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search*.
- van Rijn & Takes (2016). *The Complexity of Rummikub Problems* ([arXiv:1604.07553](https://arxiv.org/abs/1604.07553)).
