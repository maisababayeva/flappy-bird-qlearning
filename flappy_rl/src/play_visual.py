"""
Visual Flappy Bird – watch the trained Q-Learning agent play in a real game window.

Usage:
    python src/play_visual.py
    python src/play_visual.py --config narrow_gap
    python src/play_visual.py --config fast_pipes
    python src/play_visual.py --human
    python src/play_visual.py --learn

Controls:
    SPACE / click   – flap in human mode
    R               – restart episode
    +/-             – speed up / slow down
    Q / ESC         – quit
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import pygame

from flappy_env import (
    FlappyBirdEnv,
    SCREEN_W,
    SCREEN_H,
    PIPE_GAP,
    PIPE_VX,
    PIPE_W,
    BIRD_X,
)

from agent import QLearningAgent


# ── Colours ───────────────────────────────────────────────────────────────────
SKY = (113, 197, 207)
GROUND_COL = (222, 216, 149)
PIPE_COL = (115, 190, 75)
PIPE_DARK = (83, 154, 52)

BIRD_COL = (255, 215, 0)
BIRD_EYE = (30, 30, 30)
BIRD_BEAK = (255, 140, 0)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
GREEN = (80, 220, 120)

GROUND_H = 60
FPS_DEFAULT = 60
WIN_SCORE = 150


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_bird(surf, x, y, vy):
    r = 16

    pygame.draw.circle(surf, BIRD_COL, (int(x), int(y)), r)
    pygame.draw.circle(surf, (200, 170, 0), (int(x), int(y)), r, 2)

    wing_y = y + (4 if vy < 0 else 8)
    pygame.draw.ellipse(
        surf,
        (255, 180, 0),
        (int(x) - 8, int(wing_y) - 4, 16, 8),
    )

    ex = int(x + r * 0.45)
    ey = int(y - r * 0.2)

    pygame.draw.circle(surf, WHITE, (ex, ey), 5)
    pygame.draw.circle(surf, BIRD_EYE, (ex + 1, ey), 3)

    bx = int(x + r * 0.8)
    by = int(y + 2)

    pygame.draw.polygon(
        surf,
        BIRD_BEAK,
        [(bx, by), (bx + 10, by - 3), (bx + 10, by + 3)],
    )


def draw_pipe(surf, pipe, gap, screen_h, ground_h):
    x = int(pipe["x"])
    gap_y = pipe["gap_y"]

    cap_h = 18
    cap_w = PIPE_W + 8

    top_h = gap_y

    pygame.draw.rect(surf, PIPE_COL, (x, 0, PIPE_W, top_h))
    pygame.draw.rect(surf, PIPE_DARK, (x, 0, PIPE_W, top_h), 3)

    pygame.draw.rect(surf, PIPE_COL, (x - 4, top_h - cap_h, cap_w, cap_h))
    pygame.draw.rect(surf, PIPE_DARK, (x - 4, top_h - cap_h, cap_w, cap_h), 3)

    bot_y = gap_y + gap
    bot_h = screen_h - ground_h - bot_y

    pygame.draw.rect(surf, PIPE_COL, (x, bot_y, PIPE_W, bot_h))
    pygame.draw.rect(surf, PIPE_DARK, (x, bot_y, PIPE_W, bot_h), 3)

    pygame.draw.rect(surf, PIPE_COL, (x - 4, bot_y, cap_w, cap_h))
    pygame.draw.rect(surf, PIPE_DARK, (x - 4, bot_y, cap_w, cap_h), 3)


def draw_ground(surf, offset, screen_w, screen_h, ground_h):
    ground_y = screen_h - ground_h

    pygame.draw.rect(surf, GROUND_COL, (0, ground_y, screen_w, ground_h))
    pygame.draw.line(surf, (180, 170, 100), (0, ground_y), (screen_w, ground_y), 3)

    stripe_w = 40

    for i in range(-1, screen_w // stripe_w + 2):
        sx = i * stripe_w - int(offset) % stripe_w

        pygame.draw.line(
            surf,
            (200, 190, 120),
            (sx, ground_y + 10),
            (sx + 20, ground_y + 10),
            2,
        )


def draw_hud(
    surf,
    font_big,
    font_sm,
    score,
    best,
    episode,
    fps,
    speed,
    human_mode,
    alive,
    learning=False,
    epsilon=0.0,
):
    txt = font_big.render(str(score), True, WHITE)
    shadow = font_big.render(str(score), True, BLACK)

    cx = surf.get_width() // 2

    surf.blit(shadow, (cx - txt.get_width() // 2 + 2, 22))
    surf.blit(txt, (cx - txt.get_width() // 2, 20))

    mode_label = "LEARNING" if learning else ("Human" if human_mode else "Greedy")

    lines = [
        f"Target:  {WIN_SCORE}",
        f"Best:    {best}",
        f"Episode: {episode}",
        f"FPS:     {fps}",
        f"Speed:   {speed}x",
    ]

    if learning:
        lines.append(f"Epsilon: {epsilon:.4f}")

    lines += [
        "",
        "SPACE - flap" if human_mode else "R - restart",
        "+/- speed",
        "Q - quit",
    ]

    pad = 8
    panel_w = 165
    panel_h = len(lines) * 20 + pad * 2

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 140))

    surf.blit(panel, (8, 8))

    for i, line in enumerate(lines):
        col = (220, 220, 220)

        if "SPACE" in line or "R -" in line:
            col = (180, 255, 180)

        if "LEARNING" in line:
            col = (255, 220, 80)

        t = font_sm.render(line, True, col)
        surf.blit(t, (8 + pad, 8 + pad + i * 20))

    if not alive:
        msg = font_big.render("DEAD - press R", True, RED)
        surf.blit(msg, (cx - msg.get_width() // 2, SCREEN_H // 2 - 30))


def show_win_screen(screen, font_big, font_sm, score):
    screen.fill((0, 0, 0))

    title = font_big.render("YOU WON!", True, GREEN)
    score_text = font_sm.render(f"Score reached {score}", True, WHITE)
    target_text = font_sm.render(f"Target was {WIN_SCORE}", True, WHITE)
    quit_text = font_sm.render("Press Q or ESC to quit", True, WHITE)

    cx = SCREEN_W // 2
    cy = SCREEN_H // 2

    screen.blit(title, (cx - title.get_width() // 2, cy - 70))
    screen.blit(score_text, (cx - score_text.get_width() // 2, cy - 20))
    screen.blit(target_text, (cx - target_text.get_width() // 2, cy + 10))
    screen.blit(quit_text, (cx - quit_text.get_width() // 2, cy + 50))

    pygame.display.flip()

    waiting = True

    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    waiting = False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="baseline",
        choices=["baseline", "fast_pipes"],
        help="Which trained agent to load",
    )

    parser.add_argument(
        "--human",
        action="store_true",
        help="Play yourself instead of AI",
    )

    parser.add_argument(
        "--learn",
        action="store_true",
        help="Keep learning while playing",
    )

    args = parser.parse_args()

    cfg_map = {
        "baseline": ("Baseline", PIPE_GAP, PIPE_VX),
        "fast_pipes": ("Fast_Pipes", PIPE_GAP, -6.0),
    }

    label, pipe_gap, pipe_speed = cfg_map[args.config]

    base = os.path.join(os.path.dirname(__file__), "..")
    model_path = os.path.join(base, "models", f"agent_{label}.pkl")

    if not args.human:
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            print("Run: python src/train.py first.")
            sys.exit(1)

        agent = QLearningAgent.load(model_path)

        if args.learn:
            agent.epsilon = 0.1
            agent.epsilon_min = 0.005
            agent.epsilon_decay = 0.00005
            print("[Info] Online learning mode enabled.")

        print(
            f"[Info] Loaded: {label} | "
            f"Q-states={agent.q_table_size():,} | "
            f"epsilon={agent.epsilon:.3f}"
        )

    else:
        agent = None
        print("[Info] Human mode - SPACE to flap.")

    pygame.init()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

    mode_str = "Human" if args.human else ("Learning" if args.learn else f"AI - {label}")

    pygame.display.set_caption(f"Flappy Bird Q-Learning | {mode_str}")

    font_big = pygame.font.SysFont("Arial", 36, bold=True)
    font_sm = pygame.font.SysFont("Arial", 15)

    clock = pygame.time.Clock()

    env = FlappyBirdEnv(pipe_gap=pipe_gap, pipe_speed=pipe_speed)

    state = env.reset()
    done = False
    best_score = 0
    episode = 1
    speed_mult = 1
    ground_off = 0.0
    human_flap = False
    info = {"score": 0}

    running = True

    while running:
        human_flap = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    state = env.reset()
                    done = False
                    episode += 1
                    info = {"score": 0}

                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed_mult = min(speed_mult + 1, 8)

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_mult = max(speed_mult - 1, 1)

                elif event.key == pygame.K_UP:
                    speed_mult = min(speed_mult + 1, 8)

                elif event.key == pygame.K_DOWN:
                    speed_mult = max(speed_mult - 1, 1)

                elif event.key == pygame.K_SPACE and args.human:
                    human_flap = True

            elif event.type == pygame.MOUSEBUTTONDOWN and args.human:
                human_flap = True

        for _ in range(speed_mult):
            if done:
                break

            if args.human:
                action = 1 if human_flap else 0
                human_flap = False

            elif args.learn:
                action = agent.choose_action(state)

            else:
                action = agent.choose_action_greedy(state)

            prev_state = state
            state, reward, done, info = env.step(action)

            if args.learn and not args.human:
                agent.update(prev_state, action, reward, state, done)

                if done:
                    agent.decay_epsilon()

            if info["score"] >= WIN_SCORE:
                best_score = max(best_score, info["score"])
                show_win_screen(screen, font_big, font_sm, info["score"])
                running = False
                break

        if done:
            best_score = max(best_score, info["score"])

        if done and not args.human and running:
            pygame.time.wait(400)
            state = env.reset()
            done = False
            episode += 1
            info = {"score": 0}

        ground_off -= abs(pipe_speed) * speed_mult

        if ground_off < -40:
            ground_off += 40

        if not running:
            break

        screen.fill(SKY)

        for cx, cy in [(60, 80), (160, 50), (240, 100)]:
            for dx, dy, r in [
                (-18, 0, 18),
                (0, -10, 22),
                (18, 0, 18),
                (0, 8, 16),
            ]:
                pygame.draw.circle(screen, WHITE, (cx + dx, cy + dy), r)

        for pipe in env.pipes:
            draw_pipe(screen, pipe, env.pipe_gap, SCREEN_H, GROUND_H)

        draw_ground(screen, ground_off, SCREEN_W, SCREEN_H, GROUND_H)

        draw_bird(screen, BIRD_X, env.bird_y, env.bird_vy)

        eps = agent.epsilon if (agent and args.learn) else 0.0

        draw_hud(
            screen,
            font_big,
            font_sm,
            info["score"],
            best_score,
            episode,
            int(clock.get_fps()),
            speed_mult,
            args.human,
            not done,
            learning=args.learn,
            epsilon=eps,
        )

        pygame.display.flip()
        clock.tick(FPS_DEFAULT)

    pygame.quit()

    print(f"[Done] Best: {best_score} | Episodes: {episode}")


if __name__ == "__main__":
    main()