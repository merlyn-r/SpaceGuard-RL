#!/usr/bin/env python3
"""
SpaceGuard RL - Autonomous Satellite Collision Avoidance with Reinforcement Learning
====================================================================================
A satellite must hold its orbital slot while dodging space debris, using as little
fuel as possible.  A tabular Q-learning agent learns the manoeuvre policy purely by
interacting with a custom simulated environment (no pre-programmed dodging rule).

Usage
-----
  python spaceguard_rl.py train                 # train + save model, plots, metrics
  python spaceguard_rl.py evaluate              # compare RL vs baselines
  python spaceguard_rl.py demo                  # save side-by-side GIF (no control vs RL)
  python spaceguard_rl.py demo --live           # live matplotlib window
  python spaceguard_rl.py all                   # train + evaluate + demo   (default)
"""
import argparse
import json
import math
import os
import pickle
import random
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1. ENVIRONMENT
# --------------------------------------------------------------------------- #
ACTIONS = ["COAST", "THRUST +X", "THRUST -X", "THRUST +Y", "THRUST -Y"]
ACTION_VEC = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
N_ACTIONS = len(ACTIONS)


class SpaceDebrisEnv:
    """
    2-D orbital-plane abstraction (relative frame centred on the satellite's slot).

    Agent      : the satellite (position p, velocity v, limited fuel)
    Environment: a field of debris objects crossing the slot region on random paths
    State      : discretised threat geometry + deviation from slot + fuel level
    Actions    : 0 coast, 1..4 thrust along +X, -X, +Y, -Y   (impulse of DV)
    Reward     : survive (+), stay near slot (-dist), burn fuel (-), near-miss (-),
                 collision (-200, episode ends)
    """
    HALF = 40.0            # half-width of the simulated region
    N_DEBRIS = 4
    MAX_STEPS = 300
    DV = 0.6               # delta-v per thrust
    MAX_FUEL = 200         # max number of burns per episode
    THRUST_COST = 1.0      # reward penalty per burn
    COLLISION_R = 3.0      # combined radius satellite + debris
    HORIZON = 30.0         # look-ahead for closest-approach prediction
    K_SPRING, K_DAMP = 0.02, 0.15   # gentle pull back to slot (orbital restoring force)
    VMAX = 3.0

    N_THREAT = 8 * 2 * 3 * 3 + 1
    N_STATES = 3 * 5 * N_THREAT

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    # ---- helpers -------------------------------------------------------- #
    def _spawn_debris(self):
        h = self.HALF
        side = self.rng.randrange(4)
        t = self.rng.uniform(-h, h)
        x, y = [(t, h), (t, -h), (h, t), (-h, t)][side]
        tx, ty = self.rng.uniform(-9, 9), self.rng.uniform(-9, 9)   # aim near slot
        ang = math.atan2(ty - y, tx - x)
        sp = self.rng.uniform(1.0, 2.0)
        return [x, y, sp * math.cos(ang), sp * math.sin(ang)]

    def reset(self):
        self.p = [0.0, 0.0]
        self.v = [0.0, 0.0]
        self.fuel = self.MAX_FUEL
        self.t = 0
        self.debris = []
        for _ in range(self.N_DEBRIS):
            d = self._spawn_debris()
            f = self.rng.uniform(0.55, 1.0)        # scatter start positions (away from slot)
            d[0] *= f; d[1] *= f
            self.debris.append(d)
        self.dv_used = 0.0
        self.collided = False
        return self.state()

    # ---- threat model ----------------------------------------------------- #
    def threat(self):
        """Most dangerous approaching debris: (tca, miss, rel_pos, rel_vel) or None."""
        best, best_score = None, 1e9
        for d in self.debris:
            rx, ry = d[0] - self.p[0], d[1] - self.p[1]
            ux, uy = d[2] - self.v[0], d[3] - self.v[1]
            rdu = rx * ux + ry * uy
            if rdu >= 0:                         # receding -> no threat
                continue
            uu = ux * ux + uy * uy
            tca = min(-rdu / uu, self.HORIZON)
            mx, my = rx + ux * tca, ry + uy * tca
            miss = math.hypot(mx, my)
            score = miss + 0.5 * tca
            if score < best_score:
                best_score, best = score, (tca, miss, (rx, ry), (ux, uy))
        return best

    @staticmethod
    def _sector(x, y):
        return int(((math.atan2(y, x) + math.pi) / (2 * math.pi)) * 8) % 8

    def state(self):
        th = self.threat()
        if th is None:
            ti = 0
        else:
            tca, miss, (rx, ry), (ux, uy) = th
            usec = self._sector(ux, uy)
            side = 1 if (rx * uy - ry * ux) > 0 else 0
            mb = 0 if miss < 4.5 else (1 if miss < 9.0 else 2)
            tb = 0 if tca < 6 else (1 if tca < 14 else 2)
            ti = 1 + (((usec * 2 + side) * 3 + mb) * 3 + tb)
        px, py = self.p
        if math.hypot(px, py) < 5:
            dev = 0
        elif abs(px) >= abs(py):
            dev = 1 if px > 0 else 2
        else:
            dev = 3 if py > 0 else 4
        fb = 0 if self.fuel < 25 else (1 if self.fuel < 80 else 2)   # fuel level bin
        return (fb * 5 + dev) * self.N_THREAT + ti

    # ---- dynamics --------------------------------------------------------- #
    def step(self, action):
        thrust = 0
        if action != 0 and self.fuel > 0:
            ax, ay = ACTION_VEC[action]
            self.v[0] += ax * self.DV
            self.v[1] += ay * self.DV
            self.fuel -= 1
            self.dv_used += self.DV
            thrust = 1
        # restoring force + damping, speed limit
        for i in range(2):
            self.v[i] += -self.K_SPRING * self.p[i] - self.K_DAMP * self.v[i]
        sp = math.hypot(*self.v)
        if sp > self.VMAX:
            self.v[0] *= self.VMAX / sp; self.v[1] *= self.VMAX / sp

        # swept collision test (relative linear motion over this step)
        old_sat = tuple(self.p)
        self.p[0] += self.v[0]; self.p[1] += self.v[1]
        dmin = 1e9
        for k, d in enumerate(self.debris):
            ox, oy = d[0] - old_sat[0], d[1] - old_sat[1]
            ux, uy = d[2] - self.v[0], d[3] - self.v[1]
            uu = ux * ux + uy * uy
            s = 0.0 if uu < 1e-9 else max(0.0, min(1.0, -(ox * ux + oy * uy) / uu))
            dist = math.hypot(ox + ux * s, oy + uy * s)
            dmin = min(dmin, dist)
            d[0] += d[2]; d[1] += d[3]
            if abs(d[0]) > self.HALF + 5 or abs(d[1]) > self.HALF + 5:
                self.debris[k] = self._spawn_debris()
        self.t += 1

        dev = math.hypot(*self.p)
        reward = 0.2 - 0.01 * dev - self.THRUST_COST * thrust     # alive bonus, slot-keeping, fuel cost
        if dmin < 6.0:
            reward -= (6.0 - dmin) * 0.1          # mild near-miss shaping
        done = False
        if dmin < self.COLLISION_R:
            reward -= 200.0
            done = True
            self.collided = True
        elif self.t >= self.MAX_STEPS:
            done = True
        return self.state(), reward, done, {"dev": dev, "dmin": dmin}


# --------------------------------------------------------------------------- #
# 2. AGENTS  (RL agent + baselines for comparison)
# --------------------------------------------------------------------------- #
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.97):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.rng = random.Random(0)

    def act(self, s, eps=0.0):
        if self.rng.random() < eps:
            return self.rng.randrange(self.Q.shape[1])
        q = self.Q[s]
        m = q.max()
        return self.rng.choice([a for a in range(len(q)) if q[a] == m])  # random tie-break

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)


class RandomPolicy:
    name = "Random"
    def __init__(self): self.rng = random.Random(1)
    def act(self, env, s): return self.rng.randrange(N_ACTIONS)


class NoControlPolicy:
    name = "No control"
    def act(self, env, s): return 0


class RuleBasedPolicy:
    """Hand-written reactive controller (a classical engineering baseline)."""
    name = "Rule-based"
    def act(self, env, s):
        th = env.threat()
        if th is None:
            return 0
        tca, miss, (rx, ry), (ux, uy) = th
        if miss > 6.0 or tca > 12:
            return 0
        # miss vector; dodge away from it (perpendicular to approach)
        mx, my = rx + ux * tca, ry + uy * tca
        if miss < 1e-6:
            mx, my = -uy, ux
        dx, dy = -mx, -my
        if abs(dx) >= abs(dy):
            return 1 if dx > 0 else 2
        return 3 if dy > 0 else 4


class RLPolicy:
    name = "SpaceGuard RL"
    def __init__(self, agent): self.agent = agent
    def act(self, env, s): return self.agent.act(s, eps=0.0)


# --------------------------------------------------------------------------- #
# 3. TRAINING
# --------------------------------------------------------------------------- #
def train(episodes=30000, outdir="results", seed=42, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    env = SpaceDebrisEnv(seed=seed)
    agent = QLearningAgent(env.N_STATES, N_ACTIONS)
    eps_start, eps_end, decay_until = 1.0, 0.03, int(0.6 * episodes)
    alpha_start, alpha_end = 0.2, 0.03          # learning-rate decay
    log = {"reward": [], "survived": [], "fuel": [], "epsilon": [], "steps": []}
    t0 = time.time()
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_until)
        agent.alpha = alpha_start + (alpha_end - alpha_start) * ep / episodes
        s = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.act(s, eps)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s, total = s2, total + r
        log["reward"].append(total)
        log["survived"].append(0 if env.collided else 1)
        log["fuel"].append(env.dv_used)
        log["steps"].append(env.t)
        log["epsilon"].append(eps)
        if verbose and (ep + 1) % 2500 == 0:
            w = slice(-2500, None)
            print(f"  ep {ep+1:5d}/{episodes}  eps={eps:.2f}  "
                  f"avg reward={np.mean(log['reward'][w]):8.1f}  "
                  f"survival={np.mean(log['survived'][w])*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)")
    agent.save(os.path.join(outdir, "q_table.pkl"))
    with open(os.path.join(outdir, "training_log.json"), "w") as f:
        json.dump(log, f)
    plot_learning(log, outdir)
    return agent, log


def moving_avg(x, w=200):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def plot_learning(log, outdir):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    w = 200
    ax[0].plot(moving_avg(log["reward"], w), color="tab:blue")
    ax[0].set_title("Cumulative reward per episode"); ax[0].set_xlabel("Episode")
    ax[0].set_ylabel(f"Reward ({w}-ep moving avg)")
    ax[1].plot(moving_avg(log["survived"], w) * 100, color="tab:green")
    ax[1].set_title("Collision-free episodes (learning progress)"); ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Success rate %"); ax[1].set_ylim(0, 102)
    ax[2].plot(moving_avg(log["fuel"], w), color="tab:orange", label="Delta-v used")
    ax2 = ax[2].twinx(); ax2.plot(log["epsilon"], color="gray", ls="--", label="epsilon")
    ax2.set_ylabel("epsilon (exploration)")
    ax[2].set_title("Fuel usage & exploration"); ax[2].set_xlabel("Episode"); ax[2].set_ylabel("Delta-v")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "learning_curves.png"), dpi=130); plt.close()


# --------------------------------------------------------------------------- #
# 4. EVALUATION
# --------------------------------------------------------------------------- #
def run_episode(policy, seed):
    env = SpaceDebrisEnv(seed=seed)
    s, done, total, devs = env.reset(), False, 0.0, []
    while not done:
        s, r, done, info = env.step(policy.act(env, s))
        total += r; devs.append(info["dev"])
    return dict(survived=not env.collided, reward=total, fuel=env.dv_used,
                steps=env.t, dev=float(np.mean(devs)))


def evaluate(agent, n=500, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    policies = [NoControlPolicy(), RandomPolicy(), RuleBasedPolicy(), RLPolicy(agent)]
    seeds = range(10_000, 10_000 + n)           # unseen seeds, identical for all policies
    table = {}
    for pol in policies:
        rs = [run_episode(pol, sd) for sd in seeds]
        table[pol.name] = {
            "success_rate_%": 100 * np.mean([r["survived"] for r in rs]),
            "avg_reward": np.mean([r["reward"] for r in rs]),
            "avg_delta_v": np.mean([r["fuel"] for r in rs]),
            "avg_slot_deviation": np.mean([r["dev"] for r in rs]),
            "avg_steps_survived": np.mean([r["steps"] for r in rs]),
        }
    print(f"\nEvaluation over {n} unseen episodes ({SpaceDebrisEnv.MAX_STEPS} steps each)")
    hdr = f"{'Policy':<15}{'Success %':>11}{'Avg reward':>12}{'Delta-v':>10}{'Slot dev':>10}{'Steps':>8}"
    print(hdr); print("-" * len(hdr))
    for k, m in table.items():
        print(f"{k:<15}{m['success_rate_%']:>11.1f}{m['avg_reward']:>12.1f}"
              f"{m['avg_delta_v']:>10.1f}{m['avg_slot_deviation']:>10.2f}{m['avg_steps_survived']:>8.0f}")
    with open(os.path.join(outdir, "evaluation.json"), "w") as f:
        json.dump(table, f, indent=2)

    names = list(table)
    cols = ["tab:gray", "tab:red", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, key, title in zip(ax, ["success_rate_%", "avg_delta_v", "avg_slot_deviation"],
                             ["Mission success rate (%)", "Fuel used (delta-v)", "Avg deviation from slot"]):
        a.bar(names, [table[n_][key] for n_ in names], color=cols)
        a.set_title(title); a.tick_params(axis="x", rotation=20); a.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "evaluation.png"), dpi=130); plt.close()
    return table


# --------------------------------------------------------------------------- #
# 5. DEMO (animation)
# --------------------------------------------------------------------------- #
def record(policy, seed):
    env = SpaceDebrisEnv(seed=seed)
    s, done = env.reset(), False
    frames = []
    def snap(a, done_):
        frames.append(dict(p=tuple(env.p), d=[(d[0], d[1]) for d in env.debris], a=a,
                           fuel=env.fuel, t=env.t, hit=env.collided))
    snap(0, False)
    while not done:
        a = policy.act(env, s)
        s, _, done, _ = env.step(a)
        snap(a, done)
    return frames


def pick_demo_seed(agent):
    for sd in range(20_000, 20_200):
        if (not run_episode(NoControlPolicy(), sd)["survived"]
                and run_episode(RLPolicy(agent), sd)["survived"]):
            return sd
    return 20_000


def demo(agent, outdir="results", live=False):
    os.makedirs(outdir, exist_ok=True)
    if not live:
        matplotlib.use("Agg")
    seed = pick_demo_seed(agent)
    runs = [("No control", record(NoControlPolicy(), seed)),
            ("SpaceGuard RL", record(RLPolicy(agent), seed))]
    n = max(len(r[1]) for r in runs)
    H = SpaceDebrisEnv.HALF
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.patch.set_facecolor("#0b1026")
    arts = []
    for ax, (title, fr) in zip(axes, runs):
        ax.set_facecolor("#0b1026"); ax.set_xlim(-H, H); ax.set_ylim(-H, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, color="white", fontsize=13)
        ax.add_patch(plt.Circle((0, 0), 5, fill=False, ls="--", ec="#5ad", lw=1))   # slot
        trail, = ax.plot([], [], color="#5f5", lw=1, alpha=.6)
        sat, = ax.plot([], [], "s", color="#7CFC00", ms=9)
        deb = ax.scatter([], [], c="#ff6b6b", s=35)
        halo = ax.scatter([], [], s=0)
        txt = ax.text(-H + 1.5, H - 3, "", color="white", fontsize=9, family="monospace", va="top")
        boom = ax.text(0, 0, "", color="yellow", fontsize=22, ha="center", weight="bold")
        arts.append((fr, trail, sat, deb, txt, boom))

    def update(i):
        out = []
        for fr, trail, sat, deb, txt, boom in arts:
            k = min(i, len(fr) - 1); f = fr[k]
            xs = [q["p"][0] for q in fr[:k + 1]]; ys = [q["p"][1] for q in fr[:k + 1]]
            trail.set_data(xs[-40:], ys[-40:])
            sat.set_data([f["p"][0]], [f["p"][1]])
            deb.set_offsets(np.array(f["d"]))
            txt.set_text(f"step {f['t']:3d}  burns left {f['fuel']:2d}\naction: {ACTIONS[f['a']]}")
            if f["hit"]:
                boom.set_position(f["p"]); boom.set_text("COLLISION!"); sat.set_color("red")
            else:
                boom.set_text(""); sat.set_color("#7CFC00")
            out += [trail, sat, deb, txt, boom]
        return out

    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, update, frames=range(0, n + 15, 2), interval=60, blit=False)
    if live:
        plt.show()
    else:
        path = os.path.join(outdir, "demo.gif")
        anim.save(path, writer=PillowWriter(fps=16))
        print(f"Demo animation saved -> {path}  (scenario seed {seed})")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. CLI
# --------------------------------------------------------------------------- #
def load_agent(outdir):
    agent = QLearningAgent(SpaceDebrisEnv.N_STATES, N_ACTIONS)
    path = os.path.join(outdir, "q_table.pkl")
    if not os.path.exists(path):
        raise SystemExit("No trained model found - run:  python spaceguard_rl.py train")
    agent.load(path)
    return agent


def main():
    ap = argparse.ArgumentParser(description="SpaceGuard RL - satellite collision avoidance")
    ap.add_argument("mode", nargs="?", default="all", choices=["train", "evaluate", "demo", "all"])
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=500)
    ap.add_argument("--out", default="results")
    ap.add_argument("--live", action="store_true", help="show live window instead of saving a GIF")
    a = ap.parse_args()

    if a.mode in ("train", "all"):
        print(f"Training Q-learning agent for {a.episodes} episodes ...")
        agent, _ = train(a.episodes, a.out)
    else:
        agent = load_agent(a.out)
    if a.mode in ("evaluate", "all"):
        evaluate(agent, a.eval_episodes, a.out)
    if a.mode in ("demo", "all"):
        demo(agent, a.out, a.live)


if __name__ == "__main__":
    main()
