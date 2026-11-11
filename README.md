# SpaceGuard RL – Autonomous Satellite Collision Avoidance
**RL Innovation Hackathon – Build an Intelligent Learning Agent**

A satellite must hold its orbital slot while dodging space debris, spending as little
fuel as possible. A **Q-learning agent learns the manoeuvre policy from scratch** by
interacting with a custom simulated environment.

## Run it
```bash
pip install -r requirements.txt
python spaceguard_rl.py all            # train (~1 min) + evaluate + demo GIF
python spaceguard_rl.py demo --live    # live animation window (uses saved model)
```
Other modes: `train`, `evaluate`, options `--episodes N`, `--eval-episodes N`, `--out DIR`.
Outputs go to `results/`: `learning_curves.png`, `evaluation.png`, `demo.gif`,
`q_table.pkl` (trained model), `evaluation.json`, `training_log.json`.

## RL problem definition
| Element | Definition |
|---|---|
| **Agent** | The satellite (position, velocity, limited propellant) |
| **Environment** | 2-D orbital-plane region with 4 debris objects crossing the slot on random paths (speed 1–2 units/step); debris respawns when it leaves. Satellite has a gentle restoring force + damping toward its slot. |
| **State** (discretised, 2,175 states) | Most dangerous approaching debris: direction of relative velocity (8 sectors), which side it passes (2), predicted miss distance (3 bins), time-to-closest-approach (3 bins) + no-threat flag; deviation from slot (5 bins); fuel level (3 bins) |
| **Actions** | Coast, thrust +X, -X, +Y, -Y (impulse Δv = 0.6 per burn) |
| **Reward** | +0.2 per step alive, −0.01 × distance from slot, −1.0 per burn (fuel cost), small near-miss penalty (<6 units), **−200 on collision** (episode ends) |
| **Episode** | 300 steps, max 200 burns, collision radius 3 |
| **Goal** | Survive the whole episode collision-free, stay near slot, save fuel |

## Algorithm & justification
**Tabular Q-learning** (off-policy TD control, ε-greedy, ε: 1.0→0.03, α: 0.2→0.03, γ=0.97, 30,000 episodes).
Chosen because the state space can be engineered to be small, it trains in about a minute on a laptop CPU
(well inside the 12-hour limit), and the learned Q-table is fully inspectable and explainable —
every team member can explain it.

## Results (500 unseen episodes, identical scenarios for every policy)
| Policy | Success % | Avg Δv | Avg slot deviation |
|---|---|---|---|
| No control | 0.0 | 0.0 | 0.0 |
| Random | 0.0 | 15.0 | 3.8 |
| Rule-based (hand-coded) | 94.4 | 25.3 | 3.0 |
| **SpaceGuard RL** | **85.6** | 65.1 | 8.1 |

Learning behaviour: success rate rises from ~0% to ~83% during training (`learning_curves.png`).

**Honest analysis:** the RL agent learns avoidance with *no hand-coded dodging knowledge* and
massively outperforms no-control/random, but a hand-engineered rule baseline (which uses exact
geometry) is still better and more fuel-efficient. RL's weaknesses: coarse state discretisation
(aliasing), only the single most dangerous debris is observed, and it over-burns fuel.

## Implemented vs. proposed future enhancements
**Implemented:** custom environment, tabular Q-learning, 3 baselines, fuel-aware state,
evaluation on unseen scenarios, learning curves, side-by-side demo.
**Future work:** Deep Q-Network / PPO on continuous state (removes discretisation),
3-D orbital mechanics (Clohessy-Wiltshire equations), observe all debris, noisy tracking
data (POMDP), real TLE/debris catalogue data, multi-satellite cooperation.

## Resources
Python, NumPy, Matplotlib, Pillow. Algorithm: Sutton & Barto, *Reinforcement Learning: An Introduction* (Q-learning, ch. 6).
