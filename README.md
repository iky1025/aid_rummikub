# ILP-guided PPO Rummikub Agent

2인 루미큐브에서 ILP(Integer Linear Programming)가 합법적인 타일 배치
후보를 생성하고, PPO(Proximal Policy Optimization)가 후보 중 장기적으로
유리한 행동을 선택하도록 학습하는 하이브리드 강화학습 프로젝트이다.

현재 구현은 조커를 사용하지 않으며, 4색 x 13숫자 x 2벌로 구성된 104개
타일을 사용한다.

## 프로젝트 목표

루미큐브는 손패에서 어떤 타일을 낼지뿐 아니라 기존 장판을 어떻게
재배치할지까지 결정해야 하므로 행동 공간이 매우 크다. 모든 타일 조합을
PPO가 직접 생성하게 하면 합법 행동을 찾는 것부터 학습해야 하며 탐색
공간이 지나치게 커진다.

이 프로젝트에서는 역할을 다음과 같이 분리한다.

- ILP: 현재 상태에서 규칙을 만족하는 합법 배치 후보 생성
- PPO: ILP 후보 또는 draw 중 장기 누적 보상이 높은 행동 선택
- 상대 플레이어: 매 턴 ILP 최적해를 선택하는 greedy ILP 플레이어

```text
게임 상태
  -> ILP 합법 후보 생성
  -> 후보별 다음 상태 feature 생성
  -> Actor-Critic PPO가 후보 또는 draw 선택
  -> PPO 행동 적용
  -> 상대 ILP 행동 적용
  -> reward 계산 및 PPO 업데이트
```

## 파일 구조

```text
aid_rummikub/
|-- rummikub_solver.py       # 타일/세트 생성과 ILP solver
|-- rummikub_env.py          # 손패, 덱, 장판, 첫 등록 규칙
|-- ppo_env.py               # 2인 PPO 학습 환경과 reward
|-- ppo_model.py             # Actor-Critic 신경망
|-- train_ppo.py             # rollout, GAE, PPO update
|-- eval_ppo.py              # 미러 평가, 저장 및 재개
|-- main.py                  # ILP 후보 확인용 스크립트
|-- run_full_experiment.ps1  # 학습 후 평가 자동 실행
|-- run_evaluation.ps1       # 평가만 실행 또는 재개
|-- test_ppo_env.py
|-- test_ppo_model.py
|-- test_candidate_diversity.py
`-- test_reward_design.py
```

## 루미큐브 규칙 구현

유효한 세트는 다음 두 종류이다.

- Run: 같은 색의 연속 숫자 3개 이상
- Group: 같은 숫자의 서로 다른 색 3~4개

플레이어별로 `initial_meld_done` 상태를 관리한다. 첫 등록 전에는 다음
제약을 적용한다.

- 기존 장판 타일을 사용하거나 재배치할 수 없음
- 자신의 손패만으로 유효한 세트를 구성해야 함
- 내려놓는 손패의 숫자 합이 30점 이상이어야 함

첫 등록이 완료된 뒤에는 기존 장판의 모든 타일을 유효한 세트로 유지하는
조건 아래 장판 전체를 재배치할 수 있다.

## ILP Solver

각 유효 세트 후보 `i`에 대해 이진 변수 `x_i`를 정의한다.

```text
x_i = 1: 세트 후보 i를 선택
x_i = 0: 세트 후보 i를 선택하지 않음
```

기본 목적식은 손패에서 사용하는 타일 수를 최대화하는 것이다.

```text
maximize used_hand_tile_count
```

주요 제약은 다음과 같다.

- 각 타일은 실제 보유 수량을 초과해 사용할 수 없음
- 장판에 있던 타일은 결과 장판에서도 모두 사용되어야 함
- 행동 후보는 손패 타일을 최소 1개 사용해야 함
- 첫 등록 전에는 손패 점수 30점 이상을 만족해야 함

CBC solver 객체를 재사용하며 후보 탐색에는 solver time limit과 Windows
watchdog을 함께 적용한다. ILP가 비정상적으로 오래 실행되면 watchdog이
CBC 자식 프로세스를 종료해 학습 전체가 멈추는 것을 방지한다.

## 다목적 후보 생성

초기 구현은 같은 최대 타일 수 목적식의 해를 반복해서 제외하는 방식으로
최대 5개 후보를 생성했다. 실제 평가에서는 턴당 고유 후보가 평균 1개보다
적어 PPO가 greedy와 다른 선택을 학습할 여지가 부족했다.

개선한 구현은 최적해보다 최대 2장 적게 사용하는 해까지 허용하고 다음
목적식으로 후보 풀을 생성한다.

- 사용 손패 타일 수 최대화
- 높은 숫자 타일 제거
- 고립 타일 제거
- 중복 타일 및 끝 숫자(1, 2, 12, 13) 제거
- 남은 손패의 Run 연결성 보존
- 남은 손패의 Group 연결성 보존
- Run/Group 연결성 균형 보존
- 기존 장판 변경 최소화
- 긴 Run 생성
- Run 중심 또는 Group 중심 장판 생성
- 장판 세트 수 최소화 또는 최대화

생성된 원본 해는 다음 손패와 다음 장판의 실제 타일 구성이 같으면
중복으로 제거한다. 이후 남은 손패와 장판 타일의 거리를 이용해 서로 가장
다른 후보를 최대 10개 선택한다. 세트 분할만 다르고 다음 게임 상태가 같은
해는 별도 후보로 취급하지 않는다.

후보가 없는 턴은 ILP를 한 번만 실행하고 draw로 넘어간다. greedy 기준선처럼
후보 1개만 요청하는 경우에도 기본 최대 타일 해만 한 번 계산한다.

## PPO 환경

### Observation

현재 상태 observation은 109차원이다.

```text
내 손패 타일 벡터       52
장판 타일 벡터          52
덱에 남은 타일 비율      1
상대 손패 개수 비율      1
내 첫 등록 완료 여부      1
상대 첫 등록 완료 여부    1
현재 턴 진행 비율         1
합계                    109
```

### Candidate feature

각 후보 feature는 116차원이다.

```text
행동 후 내 손패 벡터     52
행동 후 장판 벡터        52
후보 구조 지표           12
합계                    116
```

후보 구조 지표에는 사용 타일 비율, 사용 점수, 남은 손패 점수, 고립/중복
비율, Run/Group 연결 수, 장판 세트 수, Run/Group 비율, 평균 세트 길이,
기존 장판 보존율이 포함된다.

### Action space

```text
0~9: ILP 후보 선택
10 : draw
```

후보가 3개라면 후보 0~2와 draw만 action mask로 활성화한다.

## PPO 모델

Actor-Critic 구조를 사용한다.

- state encoder: 현재 observation을 128차원 embedding으로 변환
- candidate encoder: 모든 후보에 공유되는 encoder
- score head: `(state embedding, candidate embedding)`으로 후보별 logit 계산
- draw head: state embedding만 사용해 draw logit 계산
- critic: 현재 상태의 value 예측

후보마다 동일한 encoder와 score head를 사용하므로 후보 슬롯 번호가 아니라
각 후보의 상태와 구조를 보고 점수를 계산한다.

## Reward V2

최종 승리를 가장 중요하게 반영하고, 중간에는 potential-based shaping을
사용한다.

```text
PPO 승리: +20
PPO 패배: -20

시간 초과 보상:
clip(2 * (상대 손패 수 - PPO 손패 수), -10, +10)
```

상태 potential은 다음 요소를 사용한다.

```text
potential =
    (상대 손패 수 - 내 손패 수)
  + 0.5  * 첫 등록 우위
  + 0.15 * 남은 손패의 Run/Group 연결 수
  - 0.30 * 고립 타일 수
  - 0.10 * 중복 타일 수
  - 0.01 * 남은 손패 점수
```

```text
shaping_reward =
    0.1 * (0.99 * next_potential - previous_potential)
```

게임 종료 상태에서는 `next_potential=0`으로 처리해 shaping이 최종 승패
보상을 왜곡하지 않게 한다. draw 고정 패널티, 매 턴 패널티, 사용 타일당
직접 보상은 사용하지 않는다. draw가 전략적으로 올바른 상황을 방해하거나
정책이 다시 ILP top-1만 모방하는 것을 피하기 위해서다.

## PPO 학습 설정

```text
n_steps       = 1000
batch_size    = 64
ppo_epochs    = 4
learning_rate = 3e-4
gamma         = 0.99
gae_lambda    = 0.95
clip_range    = 0.2
value_coef    = 0.5
entropy_coef  = 0.01
max_turns     = 100
```

현재 2차 실험은 다목적 후보와 Reward V2를 사용해 새 모델을 처음부터
600에피소드 학습하도록 설정되어 있다. 기존 104차원 후보 feature로 학습한
체크포인트와 입력 차원이 다르므로 기존 모델을 이어서 사용하지 않는다.

```text
새 모델: rummikub_ppo_diverse_model.pt
학습: 600 episodes
평가: PPO 200 games + greedy baseline 200 games
```

## 평가 방법

같은 seed로 생성한 패를 사용하고 PPO의 좌석을 0과 1로 바꾸는 미러 게임을
실행한다. 이를 통해 선공과 패 분배의 영향을 줄인다.

비교 대상은 다음과 같다.

- PPO + ILP candidates vs greedy ILP
- greedy ILP top-1 vs greedy ILP

평가 결과는 게임마다 JSONL에 즉시 저장한다. 중단 후 같은 모델, seed,
설정으로 다시 실행하면 완료된 게임을 건너뛰고 남은 게임만 수행한다.
50게임마다 진행률, 처리 속도와 ETA를 출력한다.

## 1차 실험 결과

초기 후보 생성 방식과 이전 reward로 1,000게임씩 평가한 결과이다.

| 정책 | 승리 | 패배 | 타임아웃 | 승률 |
|---|---:|---:|---:|---:|
| PPO + ILP candidates | 492 | 506 | 2 | 49.2% |
| Greedy top-1 | 494 | 506 | 0 | 49.4% |

평균 최종 손패 수는 PPO 1.32장, PPO의 상대 ILP 1.21장이었으며, greedy
평가에서는 greedy 1.18장, 상대 ILP 1.15장이었다. 두 승률의 신뢰구간이
겹치므로 PPO가 greedy보다 강하다는 근거는 얻지 못했다.

### 1차 실험에서 확인한 한계

- 최대 5개 설정에도 실제 고유 후보는 턴당 평균 0.82개
- 원본 후보 1.67개 중 평균 0.85개가 같은 다음 상태를 만드는 중복 후보
- ILP 목적과 reward가 모두 당장 손패를 많이 줄이는 방향
- PPO가 선택할 수 있는 전략적으로 다른 행동이 부족함
- 조합 탐색의 대부분이 CPU ILP이므로 학습 시간이 오래 걸림

이 분석을 바탕으로 다목적 후보 생성, 후보 구조 feature 12개, Reward V2를
추가했다. 2차 실험 결과는 600에피소드 학습 및 평가가 완료된 뒤 기존
결과와 비교한다.

## 실행 방법

### 전체 학습 및 평가

PowerShell에서 다음 명령을 실행한다.

```powershell
.\run_full_experiment.ps1
```

600에피소드 학습이 완료되면 PPO 200게임과 greedy baseline 200게임 평가를
자동으로 실행한다.

### 평가만 실행 또는 재개

```powershell
.\run_evaluation.ps1
```

### 개별 테스트

```powershell
python test_candidate_diversity.py
python test_reward_design.py
python test_ppo_model.py
python test_ppo_env.py
```

## 현재 결론

ILP로 합법 행동 공간을 줄이고 PPO가 후보 선택을 학습하는 하이브리드
강화학습 구조를 구현했다. 1차 실험에서는 PPO가 greedy ILP보다 성능을
개선하지 못했으며, 그 원인이 학습 알고리즘 자체뿐 아니라 후보 다양성과
reward 설계에 있음을 확인했다.

현재는 전략적으로 다른 후보와 장기 승리 중심 reward를 적용한 2차 실험을
진행하고 있다. 최종적으로는 PPO가 동일한 ILP 후보를 사용하는 greedy
top-1 기준선보다 높은 승률을 보이는지를 검증한다.
