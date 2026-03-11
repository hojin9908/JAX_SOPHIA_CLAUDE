# optimize.py 호출 구조 정리

`optimize.py`는 SPH(Smoothed Particle Hydrodynamics) 시뮬레이션의 초기 입자 위치를
gradient descent로 최적화하는 진입점이다. 이 문서는 `__main__`부터 시작하여
모든 함수 호출 관계와 각 함수의 input/output(타입, 차원)을 정리한다.

---

## 1. 전체 호출 흐름도

```
__main__
├── parsing()                                          # optimize.py:171
│   └── return args: dict
│
└── optimize(args)                                     # optimize.py:51
    │
    ├── generate_boundary(...)                         # src/particle_gen.py:8
    │   └── return (pos_bnd, mass_bnd)
    │
    ├── generate_particles(...)                        # src/particle_gen.py:60
    │   └── return (pos_ptl, vel_ptl, mass_ptl, actual_num)
    │
    ├── make_loss_fn(...)                              # optimize.py:22
    │   └── return loss_fn  (closure)
    │       └── loss_fn(pos_ptl)
    │           └── simulate_final(...)                # src/simulation.py:207
    │               ├── _compute_shepard(...)           # src/simulation.py:11
    │               │   └── kernel_matrix() x4         # src/kernels.py:26
    │               │       ├── safe_norm()            # src/kernels.py:9
    │               │       └── wendland_c2()          # src/kernels.py:14
    │               │
    │               ├── _make_step_fn(...)              # src/simulation.py:44
    │               │   └── return step_fn  (closure)
    │               │       ├── _compute_shepard(...)   # (조건부: step_idx % shepard_step == 0)
    │               │       ├── kernel_matrix() x4     # src/kernels.py:26
    │               │       ├── grad_kernel_matrix() x2 # src/kernels.py:42
    │               │       │   ├── safe_norm()        # src/kernels.py:9
    │               │       │   └── wendland_c2() (간접 — dW/dr 수식 직접 계산)
    │               │       └── tait_eos() x2          # src/eos.py:7
    │               │
    │               └── lax.scan(segment_fn, ...)
    │                   └── lax.scan(inner_step → step_fn, ...)
    │
    ├── opt_step(pos_ptl)  [@jax.jit]                  # optimize.py:111
    │   └── jax.value_and_grad(loss_fn)(pos_ptl)
    │       └── loss_fn(pos_ptl)  → (위 호출 체인과 동일)
    │
    └── [선택] OptimizationAnimator(...)               # visualize/optimization_animator.py:19
        └── create_gif()                               # visualize/optimization_animator.py:113
            ├── _run_simulation(pos)                   # visualize/optimization_animator.py:76
            │   └── simulate(...)                      # src/simulation.py:159
            │       ├── _compute_shepard(...)
            │       ├── _make_step_fn(..., accumulate_state=True)
            │       └── lax.scan(step_fn, ...)
            │
            └── _render_frame(positions, title)        # visualize/optimization_animator.py:87
```

---

## 2. 모듈별 함수 상세

### 2.1 optimize.py

#### `parsing()`
| 항목 | 내용 |
|------|------|
| **위치** | `optimize.py:171` |
| **역할** | argparse를 사용하여 CLI 인자를 파싱하고 `dict`로 반환 |
| **Input** | 없음 (sys.argv에서 읽음) |
| **Output** | `args: dict` — 모든 CLI 인자를 key-value로 담은 딕셔너리 |
| **호출 하위 함수** | 없음 |

주요 키: `tag`, `save_dir`, `seed`, `x`, `y`, `x0`, `y0`, `uniform`, `shepard_step`,
`rho0`, `g`, `c0`, `gamma`, `num_ptl`, `bnd_layer`, `bnd_loss`, `t`, `dt`,
`checkpoint_every`, `lr`, `opt_steps`, `use_soft_max`, `temperature`, `create_gif`

---

#### `optimize(args)`
| 항목 | 내용 |
|------|------|
| **위치** | `optimize.py:51` |
| **역할** | 입자 생성 → loss 함수 빌드 → gradient descent 최적화 루프 실행 → 결과 저장 |
| **Input** | `args: dict` — `parsing()`이 반환한 설정 딕셔너리 |
| **Output** | `None` (결과를 `.npy` 파일로 저장) |
| **호출 하위 함수** | `generate_boundary`, `generate_particles`, `make_loss_fn`, `opt_step`, `OptimizationAnimator` (선택) |

저장 파일:
- `initial_pos.npy` — `[num_ptl, 2]`
- `optimization_positions.npy` — `[opt_steps+1, num_ptl, 2]`
- `optimized_pos.npy` — `[num_ptl, 2]`
- `optimization.gif` — (선택, `--create_gif` 플래그 사용 시)

---

#### `make_loss_fn(vel_ptl, mass_ptl, pos_bnd, mass_bnd, *, h, g, dt, rho0, c0, gamma, n_steps, shepard_step, checkpoint_every, use_soft_max, temperature)`
| 항목 | 내용 |
|------|------|
| **위치** | `optimize.py:22` |
| **역할** | `pos_ptl`만을 인자로 받는 loss 함수(클로저)를 생성. `jax.value_and_grad`에 넘기기 위해 다른 인자를 클로저로 고정 |
| **Input** | |
| - `vel_ptl` | `jax.Array [num_ptl, 2]` — 입자 초기 속도 |
| - `mass_ptl` | `jax.Array [num_ptl]` — 입자 질량 |
| - `pos_bnd` | `jax.Array [num_bnd, 2]` — 경계 입자 위치 |
| - `mass_bnd` | `jax.Array [num_bnd]` — 경계 입자 질량 |
| - `h` | `float` — smoothing length |
| - `g` | `float` — 중력 가속도 |
| - `dt` | `float` — 시간 간격 |
| - `rho0` | `float` — 기준 밀도 |
| - `c0` | `float` — 음속 |
| - `gamma` | `float` — EOS 강성 파라미터 |
| - `n_steps` | `int` — 시뮬레이션 스텝 수 |
| - `shepard_step` | `int` — Shepard 필터 갱신 주기 |
| - `checkpoint_every` | `int` — 체크포인트 구간 크기 |
| - `use_soft_max` | `bool` — soft max 사용 여부 (기본 `False`) |
| - `temperature` | `float` — soft max 온도 (기본 `0.01`) |
| **Output** | `loss_fn: Callable[[jax.Array], jax.Array]` — `pos_ptl [num_ptl, 2]` → `scalar` |
| **호출 하위 함수** | 반환된 `loss_fn` 내부에서 `simulate_final` 호출 |

`loss_fn` 내부 동작:
- `simulate_final(pos_ptl, ...)` → `(final_pos [num_ptl, 2], final_vel [num_ptl, 2])`
- `use_soft_max=True`이면: `temperature * logsumexp(y_coords / temperature)` → `scalar`
- `use_soft_max=False`이면: `jnp.max(y_coords)` → `scalar`

---

#### `opt_step(pos_ptl)` (JIT 컴파일된 내부 함수)
| 항목 | 내용 |
|------|------|
| **위치** | `optimize.py:111` (`optimize` 함수 내부, `@jax.jit` 데코레이터) |
| **역할** | 한 번의 gradient descent 스텝: loss와 gradient를 계산하고 위치를 업데이트 |
| **Input** | `pos_ptl: jax.Array [num_ptl, 2]` — 현재 입자 위치 |
| **Output** | `tuple` |
| - `new_pos` | `jax.Array [num_ptl, 2]` — 업데이트된 위치 (`pos_ptl - lr * grad_val`) |
| - `loss_val` | `jax.Array scalar` — loss 값 |
| - `grad_norm` | `jax.Array scalar` — gradient의 L2 norm |
| **호출 하위 함수** | `jax.value_and_grad(loss_fn)` → `loss_fn` → `simulate_final` |

---

### 2.2 src/particle_gen.py

#### `generate_boundary(x, y, spacing, bnd_layer, bnd_loss, mass)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/particle_gen.py:8` |
| **역할** | 2D 댐 경계 입자 생성 (왼쪽 벽, 바닥, 오른쪽 벽, 다중 레이어) |
| **Input** | |
| - `x` | `float` — 댐 너비 |
| - `y` | `float` — 댐 높이 |
| - `spacing` | `float` — 기본 입자 간격 |
| - `bnd_layer` | `int` — 경계 레이어 수 |
| - `bnd_loss` | `float` — 경계 입자 간격 배율 |
| - `mass` | `float` — 입자 질량 |
| **Output** | `tuple[jax.Array, jax.Array]` |
| - `pos_bnd` | `jax.Array [N_bnd, 2]` — 경계 입자 위치 |
| - `mass_bnd` | `jax.Array [N_bnd]` — 경계 입자 질량 |
| **호출 하위 함수** | 없음 (NumPy 연산만 사용) |

---

#### `generate_particles(num, x0, y0, rho0, spacing, uniform, seed)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/particle_gen.py:60` |
| **역할** | 초기 유체 입자를 균일 격자 또는 랜덤으로 생성 |
| **Input** | |
| - `num` | `int` — 요청 입자 수 |
| - `x0` | `float` — 초기 분포 너비 |
| - `y0` | `float` — 초기 분포 높이 |
| - `rho0` | `float` — 기준 밀도 |
| - `spacing` | `float` — 입자 간격 (균일 격자용) |
| - `uniform` | `bool` — 균일 격자 여부 (기본 `True`) |
| - `seed` | `int` — 난수 시드 (기본 `42`) |
| **Output** | `tuple[jax.Array, jax.Array, jax.Array, int]` |
| - `pos` | `jax.Array [N, 2]` — 초기 위치 |
| - `vel` | `jax.Array [N, 2]` — 초기 속도 (0 벡터) |
| - `mass_arr` | `jax.Array [N]` — 입자 질량 |
| - `actual_num` | `int` — 실제 생성된 입자 수 |
| **호출 하위 함수** | 없음 (NumPy 연산만 사용) |

---

### 2.3 src/simulation.py

#### `simulate_final(pos_ptl, vel_ptl, mass_ptl, pos_bnd, mass_bnd, *, h, g, dt, rho0, c0, gamma, n_steps, shepard_step, checkpoint_every)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/simulation.py:207` |
| **역할** | 시뮬레이션을 실행하고 최종 상태만 반환 (메모리 효율적). 세그먼트별 gradient checkpointing 적용 |
| **Input** | |
| - `pos_ptl` | `jax.Array [num_ptl, 2]` — 초기 입자 위치 |
| - `vel_ptl` | `jax.Array [num_ptl, 2]` — 초기 입자 속도 |
| - `mass_ptl` | `jax.Array [num_ptl]` — 입자 질량 |
| - `pos_bnd` | `jax.Array [num_bnd, 2]` — 경계 위치 (고정) |
| - `mass_bnd` | `jax.Array [num_bnd]` — 경계 질량 |
| - `h` | `float` — smoothing length |
| - `g` | `float` — 중력 가속도 |
| - `dt` | `float` — 시간 간격 |
| - `rho0` | `float` — 기준 밀도 |
| - `c0` | `float` — 음속 |
| - `gamma` | `float` — EOS 강성 파라미터 |
| - `n_steps` | `int` — 시뮬레이션 스텝 수 (`checkpoint_every`로 나누어 떨어져야 함) |
| - `shepard_step` | `int` — Shepard 필터 갱신 주기 |
| - `checkpoint_every` | `int` — 체크포인트 구간 크기 (기본 `100`) |
| **Output** | `tuple[jax.Array, jax.Array]` |
| - `final_pos` | `jax.Array [num_ptl, 2]` — 최종 입자 위치 |
| - `final_vel` | `jax.Array [num_ptl, 2]` — 최종 입자 속도 |
| **호출 하위 함수** | `_compute_shepard`, `_make_step_fn`, `lax.scan` (세그먼트 + 내부 스텝) |

내부 구조:
- `n_segments = n_steps // checkpoint_every`
- outer `lax.scan` → `segment_fn` (`@jax.checkpoint`)
  - inner `lax.scan` → `inner_step` → `step_fn`

---

#### `simulate(pos_ptl, vel_ptl, mass_ptl, pos_bnd, mass_bnd, *, h, g, dt, rho0, c0, gamma, n_steps, shepard_step)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/simulation.py:159` |
| **역할** | 전체 시뮬레이션을 실행하고 모든 스텝의 trajectory를 반환 |
| **Input** | |
| - `pos_ptl` | `jax.Array [N, 2]` — 초기 입자 위치 |
| - `vel_ptl` | `jax.Array [N, 2]` — 초기 입자 속도 |
| - `mass_ptl` | `jax.Array [N]` — 입자 질량 |
| - `pos_bnd` | `jax.Array [M, 2]` — 경계 위치 (고정) |
| - `mass_bnd` | `jax.Array [M]` — 경계 질량 |
| - `h` | `float` — smoothing length |
| - `g` | `float` — 중력 가속도 |
| - `dt` | `float` — 시간 간격 |
| - `rho0` | `float` — 기준 밀도 |
| - `c0` | `float` — 음속 |
| - `gamma` | `float` — EOS 강성 파라미터 |
| - `n_steps` | `int` — 시뮬레이션 스텝 수 |
| - `shepard_step` | `int` — Shepard 필터 갱신 주기 |
| **Output** | `jax.Array [n_steps, N, 5]` — 각 스텝의 (x, y, vx, vy, mass) |
| **호출 하위 함수** | `_compute_shepard`, `_make_step_fn`, `lax.scan` |

---

#### `_make_step_fn(pos_bnd, mass_ptl, mass_bnd, *, h, g, dt, rho0, c0, gamma, shepard_step, accumulate_state)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/simulation.py:44` |
| **역할** | 단일 SPH 타임스텝 함수(클로저)를 생성. `lax.scan`에 전달 가능한 형태 |
| **Input** | |
| - `pos_bnd` | `jax.Array [num_bnd, 2]` — 경계 위치 (고정) |
| - `mass_ptl` | `jax.Array [num_ptl]` — 입자 질량 |
| - `mass_bnd` | `jax.Array [num_bnd]` — 경계 질량 |
| - `h` | `float` — smoothing length |
| - `g` | `float` — 중력 가속도 |
| - `dt` | `float` — 시간 간격 |
| - `rho0` | `float` — 기준 밀도 |
| - `c0` | `float` — 음속 |
| - `gamma` | `float` — EOS 강성 파라미터 |
| - `shepard_step` | `int` — Shepard 필터 갱신 주기 |
| - `accumulate_state` | `bool` — `True`이면 trajectory 누적, `False`이면 `None` 반환 (기본 `True`) |
| **Output** | `step_fn: Callable[[carry, step_idx], [new_carry, state_out]]` |
| **호출 하위 함수** | 반환된 `step_fn` 내부에서: `_compute_shepard` (조건부), `kernel_matrix` x4, `grad_kernel_matrix` x2, `tait_eos` x2 |

`step_fn`의 시그니처:
- **carry** (input/output): `tuple`
  - `pos_p: [num_ptl, 2]`, `vel_p: [num_ptl, 2]`, `rho_p: [num_ptl]`, `rho_b: [num_bnd]`, `s_ptl: [num_ptl]`, `s_bnd: [num_bnd]`
- **step_idx**: `scalar int`
- **state_out**: `accumulate_state=True`이면 `[num_ptl, 5]` (x, y, vx, vy, mass), 아니면 `None`

---

#### `_compute_shepard(pos_ptl, pos_bnd, mass_ptl, mass_bnd, rho_ptl, rho_bnd, h)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/simulation.py:11` |
| **역할** | Shepard 보정 필터를 계산: `filter_i = sum_j (m_j / rho_j) * W_ij` |
| **Input** | |
| - `pos_ptl` | `jax.Array [num_ptl, 2]` — 입자 위치 |
| - `pos_bnd` | `jax.Array [num_bnd, 2]` — 경계 위치 |
| - `mass_ptl` | `jax.Array [num_ptl]` — 입자 질량 |
| - `mass_bnd` | `jax.Array [num_bnd]` — 경계 질량 |
| - `rho_ptl` | `jax.Array [num_ptl]` — 입자 밀도 |
| - `rho_bnd` | `jax.Array [num_bnd]` — 경계 밀도 |
| - `h` | `float` — smoothing length |
| **Output** | `tuple[jax.Array, jax.Array]` |
| - `ptl_filter` | `jax.Array [num_ptl]` — 입자 Shepard 필터 |
| - `bnd_filter` | `jax.Array [num_bnd]` — 경계 Shepard 필터 |
| **호출 하위 함수** | `kernel_matrix` x4 (pp, pb, bp, bb) |

---

### 2.4 src/kernels.py

#### `safe_norm(dx, eps)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/kernels.py:9` |
| **역할** | 미분 가능한 L2 norm. `r=0`에서 NaN gradient를 방지하기 위해 `sqrt(sum(x^2) + eps^2)` 사용 |
| **Input** | |
| - `dx` | `jax.Array [..., D]` — 변위 벡터 (마지막 축이 좌표 차원) |
| - `eps` | `float` — 안정화 상수 (기본 `1e-6`) |
| **Output** | `jax.Array [...]` — 입력에서 마지막 축을 제거한 shape의 norm 값 |
| **호출 하위 함수** | 없음 |

---

#### `wendland_c2(r, h)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/kernels.py:14` |
| **역할** | Wendland C2 커널 (2D): `W(r,h) = (7/(4*pi*h^2)) * (1-q)^4 * (1+4q)`, `q = r/(2h)`. compact support `r < 2h` |
| **Input** | |
| - `r` | `jax.Array [...]` — 거리 |
| - `h` | `float` — smoothing length |
| **Output** | `jax.Array [...]` — 커널 값 (입력과 동일한 shape) |
| **호출 하위 함수** | 없음 |

---

#### `kernel_matrix(pos_i, pos_j, h)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/kernels.py:26` |
| **역할** | 모든 쌍에 대한 커널 평가 (all-pairs) |
| **Input** | |
| - `pos_i` | `jax.Array [N, 2]` — 주체 입자 위치 |
| - `pos_j` | `jax.Array [M, 2]` — 이웃 입자 위치 |
| - `h` | `float` — smoothing length |
| **Output** | `jax.Array [N, M]` — 커널 값 행렬 |
| **호출 하위 함수** | `safe_norm`, `wendland_c2` |

내부:
- `dx = pos_i[:, None, :] - pos_j[None, :, :]` → `[N, M, 2]`
- `r = safe_norm(dx)` → `[N, M]`
- `wendland_c2(r, h)` → `[N, M]`

---

#### `grad_kernel_matrix(pos_i, pos_j, h)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/kernels.py:42` |
| **역할** | Wendland C2 커널의 `pos_i`에 대한 gradient를 모든 쌍에 대해 계산: `grad_i W = dW/dr * (x_i - x_j) / r` |
| **Input** | |
| - `pos_i` | `jax.Array [N, 2]` — 주체 입자 위치 |
| - `pos_j` | `jax.Array [M, 2]` — 이웃 입자 위치 |
| - `h` | `float` — smoothing length |
| **Output** | `jax.Array [N, M, 2]` — 커널 gradient 행렬 |
| **호출 하위 함수** | `safe_norm` |

내부:
- `dx = pos_i[:, None, :] - pos_j[None, :, :]` → `[N, M, 2]`
- `r = safe_norm(dx)` → `[N, M]`
- `dWdr = const/(2h) * (-20) * (1-q)^3 * q` → `[N, M]`
- `grad = dWdr[..., None] * dx / r[..., None]` → `[N, M, 2]`
- compact support 마스킹: `r < 2h`

---

### 2.5 src/eos.py

#### `tait_eos(rho, rho0, c0, gamma)`
| 항목 | 내용 |
|------|------|
| **위치** | `src/eos.py:7` |
| **역할** | Tait 상태 방정식: `p = (c0^2 * rho0 / gamma) * ((rho / rho0)^gamma - 1)` |
| **Input** | |
| - `rho` | `jax.Array [...]` — 밀도 |
| - `rho0` | `float` — 기준 밀도 |
| - `c0` | `float` — 음속 |
| - `gamma` | `float` — 강성 파라미터 |
| **Output** | `jax.Array [...]` — 압력 (입력과 동일한 shape) |
| **호출 하위 함수** | 없음 |

---

### 2.6 visualize/optimization_animator.py

#### `OptimizationAnimator` 클래스
| 항목 | 내용 |
|------|------|
| **위치** | `visualize/optimization_animator.py:19` |
| **역할** | 최적화 과정을 GIF로 시각화. 각 최적화 스텝마다 초기 상태 → 시뮬레이션 리플레이 → 다음 스텝 전환 애니메이션 생성 |

##### `__init__(self, pos_history, vel_ptl, mass_ptl, pos_bnd, mass_bnd, sim_params, save_path, *, fps, initial_duration, sim_duration, transition_duration, x, y, ptl_s, bnd_s)`
| 항목 | 내용 |
|------|------|
| **위치** | `visualize/optimization_animator.py:50` |
| **Input** | |
| - `pos_history` | `ndarray [N_steps+1, num_ptl, 2]` — 각 최적화 스텝의 입자 위치 |
| - `vel_ptl` | `jax.Array [num_ptl, 2]` — 고정 초기 속도 |
| - `mass_ptl` | `jax.Array [num_ptl]` — 고정 입자 질량 |
| - `pos_bnd` | `jax.Array [num_bnd, 2]` — 경계 위치 |
| - `mass_bnd` | `jax.Array [num_bnd]` — 경계 질량 |
| - `sim_params` | `dict` — `{h, g, dt, rho0, c0, gamma, n_steps, shepard_step}` |
| - `save_path` | `str` — GIF 저장 경로 |
| - `fps` | `int` — 프레임 레이트 (기본 `30`) |
| - `initial_duration` | `float` — 초기 상태 표시 시간(초) (기본 `0.3`) |
| - `sim_duration` | `float` — 시뮬레이션 재생 시간(초) (기본 `0.5`) |
| - `transition_duration` | `float` — 전환 애니메이션 시간(초) (기본 `0.3`) |
| - `x`, `y` | `float` — 댐 치수 (축 범위용) (기본 `5.0`, `3.0`) |
| - `ptl_s`, `bnd_s` | `float` — scatter 점 크기 (기본 `0.1`) |

##### `_run_simulation(self, pos_ptl)`
| 항목 | 내용 |
|------|------|
| **위치** | `visualize/optimization_animator.py:76` |
| **역할** | 주어진 초기 위치에 대해 전체 시뮬레이션을 실행하여 trajectory 반환 |
| **Input** | `pos_ptl: ndarray [num_ptl, 2]` — 초기 입자 위치 |
| **Output** | `ndarray [n_steps, num_ptl, 5]` — trajectory (x, y, vx, vy, mass) |
| **호출 하위 함수** | `simulate` (`src/simulation.py`) |

##### `_render_frame(self, positions, title)`
| 항목 | 내용 |
|------|------|
| **위치** | `visualize/optimization_animator.py:87` |
| **역할** | 입자 위치를 matplotlib scatter plot으로 렌더링하여 PIL Image 반환 |
| **Input** | |
| - `positions` | `ndarray [num_ptl, 2]` — 입자 위치 |
| - `title` | `str` — 프레임 제목 |
| **Output** | `PIL.Image.Image` — RGB 이미지 |
| **호출 하위 함수** | 없음 (matplotlib, PIL 사용) |

##### `create_gif(self)`
| 항목 | 내용 |
|------|------|
| **위치** | `visualize/optimization_animator.py:113` |
| **역할** | 전체 최적화 GIF를 생성하고 저장. 각 스텝마다 3단계: 정적 표시 → 시뮬레이션 리플레이 → 전환 |
| **Input** | 없음 (인스턴스 속성 사용) |
| **Output** | `None` (GIF 파일을 `self.save_path`에 저장) |
| **호출 하위 함수** | `_run_simulation`, `_render_frame` |

---

## 3. 호출 빈도 요약

| 함수 | 호출 위치 | 비고 |
|------|-----------|------|
| `kernel_matrix` | `_compute_shepard` x4, `step_fn` x4 | 가장 빈번하게 호출 |
| `grad_kernel_matrix` | `step_fn` x2 | 압력 가속도 계산용 |
| `safe_norm` | `kernel_matrix`, `grad_kernel_matrix` | 모든 커널 연산의 기저 |
| `wendland_c2` | `kernel_matrix` | 커널 값 계산 |
| `tait_eos` | `step_fn` x2 | 입자/경계 압력 계산 |
| `_compute_shepard` | 초기화 시 1회 + 매 `shepard_step` 스텝마다 | 조건부 갱신 |
