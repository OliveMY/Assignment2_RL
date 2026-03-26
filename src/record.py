"""
Record video of trained PPO agents playing Unity ball game.

Uses screen capture (mss) to grab frames from the Unity window,
then encodes them as MP4 videos with imageio.

Usage:
    python src/record.py --env simple --rounds 3
    python src/record.py --env hard --model models/hard/checkpoint_550000_steps.zip --rounds 5
    python src/record.py --env simple --rounds 2 --region "100,100,800,600"

Requires: pip install mss "imageio[ffmpeg]"
Note: macOS requires Screen Recording permission for your terminal app.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
from stable_baselines3 import PPO
from env_wrapper import make_env

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Fail fast with clear messages if mss or imageio are missing."""
    missing = []
    try:
        import mss
    except ImportError:
        missing.append("mss")
    try:
        import imageio
    except ImportError:
        missing.append("imageio[ffmpeg]")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def parse_region(region_str):
    """Parse 'x,y,w,h' string into mss monitor dict."""
    x, y, w, h = map(int, region_str.split(","))
    return {"left": x, "top": y, "width": w, "height": h}


class FrameRecorder:
    """Captures screen frames in a background thread at a fixed interval."""

    def __init__(self, monitor, fps):
        import mss
        self._sct = mss.mss()
        self._monitor = monitor
        self._interval = 1.0 / fps
        self._frames = []
        self._recording = False
        self._thread = None
        self._lock = threading.Lock()

    def _capture_loop(self):
        next_capture = time.monotonic()
        while self._recording:
            next_capture += self._interval
            screenshot = self._sct.grab(self._monitor)
            frame = np.array(screenshot)
            frame = frame[:, :, :3]    # drop alpha channel
            frame = frame[:, :, ::-1]  # BGR -> RGB
            with self._lock:
                self._frames.append(frame.copy())
            sleep_time = next_capture - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """Start capturing frames."""
        with self._lock:
            self._frames = []
        self._recording = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop capturing and return collected frames."""
        self._recording = False
        if self._thread:
            self._thread.join()
        with self._lock:
            frames = self._frames
            self._frames = []
        return frames


def record(env_name, model_path, num_rounds, output_dir, fps, region,
           worker_id=1):
    import imageio

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Recording: {env_name.upper()}")
    print(f"Model: {model_path}")
    print(f"Rounds: {num_rounds}")
    print(f"Output: {output_dir}")
    print(f"FPS: {fps}")
    print(f"{'='*50}")

    # Create environment with graphics enabled at real-time speed
    print(f"Creating {env_name} environment (graphics enabled, real-time)...")
    env = make_env(env_name=env_name, time_scale=1.0, no_graphics=False,
                   worker_id=worker_id)
    model = PPO.load(model_path)

    print("Waiting 3 seconds for Unity window to initialize...")
    time.sleep(3)

    import mss
    sct = mss.mss()
    if region:
        monitor = parse_region(region)
    else:
        monitor = sct.monitors[1]  # Primary monitor
    sct.close()

    recorder = FrameRecorder(monitor, fps)
    results = []

    for episode in range(num_rounds):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        recorder.start()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1

        frames = recorder.stop()

        # Determine outcome
        if episode_reward > 0:
            outcome = "win"
        elif episode_reward < 0:
            outcome = "loss"
        else:
            outcome = "draw"

        # Write video
        filename = f"episode_{episode+1:02d}_{outcome}_r{episode_reward:+.2f}.mp4"
        filepath = os.path.join(output_dir, filename)

        writer = imageio.get_writer(filepath, fps=fps, codec="libx264",
                                    quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        print(f"  Episode {episode+1}/{num_rounds}: {outcome.upper()} "
              f"(reward={episode_reward:+.3f}, {step_count} steps, "
              f"{len(frames)} frames) -> {filename}")
        results.append({"outcome": outcome, "reward": episode_reward,
                        "file": filename})

    env.close()

    # Summary
    wins = sum(1 for r in results if r["outcome"] == "win")
    print(f"\n--- Summary ---")
    print(f"  Win rate: {wins}/{num_rounds} ({100*wins/num_rounds:.0f}%)")
    print(f"  Recordings saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Record video of trained PPO agents playing Unity ball game")
    parser.add_argument("--env", type=str, required=True,
                        choices=["simple", "medium", "hard"])
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip (default: models/{env}/final.zip)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of episodes to record (default: 3)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: recordings/{env})")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video frame rate (default: 30)")
    parser.add_argument("--region", type=str, default=None,
                        help="Capture region as 'x,y,w,h' (default: full primary monitor)")
    parser.add_argument("--worker-id", type=int, default=1,
                        help="Unity worker ID, use different ID if training is running (default: 1)")
    args = parser.parse_args()

    check_dependencies()

    model_path = args.model or os.path.join(BASE_DIR, "models", args.env, "final.zip")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(BASE_DIR, "recordings", args.env)
    record(args.env, model_path, args.rounds, output_dir, args.fps, args.region,
           args.worker_id)


if __name__ == "__main__":
    main()
