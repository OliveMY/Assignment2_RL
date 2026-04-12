"""
Export SB3 PPO models to Unity ML-Agents compatible ONNX format.

Unity ML-Agents 2.0.2 (Barracuda) expects a specific ONNX layout:
  - Input:  obs_0 [1, obs_size]  (float32, fixed batch=1)
  - Output: version_number (int, =3), memory_size (int, =0),
            discrete_actions [1, num_branches] (float32),
            discrete_action_output_shape [num_branches]

Barracuda constraints:
  - All tensors must be float32 (no int64 outputs)
  - Fixed batch dimension of 1 (no dynamic axes)
  - Opset 9 for maximum compatibility

Usage:
    # Export a single model
    python src/export_onnx.py --env simple --model models/simple/final.zip

    # Export all three environments
    python src/export_onnx.py --env all

    # Export directly into Unity Assets folder
    python src/export_onnx.py --env all --output-dir /path/to/Unity/Assets/Models
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Observation size and action branches for RollerAgent
OBS_SIZE = 12
DISCRETE_BRANCHES = [3, 3]  # MultiDiscrete([3, 3])


class UnityONNXWrapper(nn.Module):
    """Wraps an SB3 PPO policy network for Unity ML-Agents ONNX inference.

    Barracuda requires all outputs to be float32 and works best with
    fixed batch size of 1. This wrapper:
      1. Runs observations through the SB3 policy MLP
      2. Outputs per-branch logits (NOT argmax) — Unity takes argmax itself
      3. Provides version_number and memory_size as float constants

    For version 3 (MLAgents2_0), Unity's DiscreteActionOutputApplier reads
    the discrete_actions tensor directly as action indices (not logits).
    So we must output argmax per branch, cast to float32 for Barracuda.
    """

    def __init__(self, sb3_model: PPO):
        super().__init__()
        policy = sb3_model.policy

        # Extract the MLP feature extractor and action network
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

        # Branch config: both branches have 3 options
        self.num_branches = len(DISCRETE_BRANCHES)
        self.branch_size = DISCRETE_BRANCHES[0]  # 3

        # Constants Unity expects (as float32 for Barracuda compatibility)
        self.register_buffer(
            "version_number", torch.tensor([3], dtype=torch.float32)
        )
        self.register_buffer(
            "memory_size", torch.tensor([0], dtype=torch.float32)
        )

    def forward(self, obs_0: torch.Tensor, action_masks: torch.Tensor):
        # Run through SB3's policy network to get logits
        features = self.features_extractor(obs_0)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        # logits shape: [1, 6]

        # Apply action masks: Unity passes 1.0 for allowed, 0.0 for blocked.
        logits = logits + (action_masks - 1.0) * 1e8

        # Reshape to [1, num_branches, branch_size] = [1, 2, 3]
        # then argmax on last dim → [1, 2]
        # This avoids Split which Barracuda's ONNX importer can't handle.
        logits = logits.reshape(1, self.num_branches, self.branch_size)
        discrete_actions = logits.argmax(dim=2).float()

        return discrete_actions, self.version_number, self.memory_size


def _add_metadata_to_onnx(onnx_path: str):
    """Add discrete_action_output_shape metadata that Unity ML-Agents needs.

    Unity's BarracudaPolicy looks for this constant to know how to split
    the logits tensor into per-branch segments.
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    model = onnx.load(onnx_path)

    # Add discrete_action_output_shape as a constant initializer
    shape_array = np.array(DISCRETE_BRANCHES, dtype=np.int32)
    shape_tensor = numpy_helper.from_array(
        shape_array, name="discrete_action_output_shape"
    )
    model.graph.initializer.append(shape_tensor)

    # Also add it as a graph output so Unity can find it
    shape_output = onnx.helper.make_tensor_value_info(
        "discrete_action_output_shape",
        TensorProto.INT32,
        [len(DISCRETE_BRANCHES)],
    )
    model.graph.output.append(shape_output)

    # Add continuous_action_output_shape as empty (no continuous actions)
    cont_array = np.array([0], dtype=np.int32)
    cont_tensor = numpy_helper.from_array(
        cont_array, name="continuous_action_output_shape"
    )
    model.graph.initializer.append(cont_tensor)
    cont_output = onnx.helper.make_tensor_value_info(
        "continuous_action_output_shape",
        TensorProto.INT32,
        [1],
    )
    model.graph.output.append(cont_output)

    onnx.save(model, onnx_path)


def export_model(env_name: str, model_path: str, output_dir: str):
    """Export a single SB3 model to Unity ONNX format."""
    print(f"\n{'='*50}")
    print(f"Exporting: {env_name.upper()}")
    print(f"  Model:  {model_path}")

    # Load SB3 model
    model = PPO.load(model_path, device="cpu")
    model.policy.eval()

    # Wrap for Unity
    wrapper = UnityONNXWrapper(model)
    wrapper.eval()

    # Fixed batch=1 inputs (Barracuda prefers fixed dimensions)
    dummy_obs = torch.randn(1, OBS_SIZE, dtype=torch.float32)
    # action_masks: 1.0 = allowed, 0.0 = blocked. All ones = no masking.
    mask_size = sum(DISCRETE_BRANCHES)
    dummy_masks = torch.ones(1, mask_size, dtype=torch.float32)

    # Test forward pass
    with torch.no_grad():
        actions, ver, mem = wrapper(dummy_obs, dummy_masks)
        print(f"  Test forward pass: actions={actions.numpy().astype(int)}, "
              f"version={ver.item():.0f}, memory_size={mem.item():.0f}")

    # Export to ONNX
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"RollingControl_{env_name}.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_masks),
        output_path,
        input_names=["obs_0", "action_masks"],
        output_names=["discrete_actions", "version_number", "memory_size"],
        opset_version=9,  # Barracuda works best with opset 9
        dynamo=False,     # Use legacy exporter for opset 9 compatibility
    )

    # Add Unity ML-Agents metadata (discrete_action_output_shape, etc.)
    try:
        _add_metadata_to_onnx(output_path)
        print(f"  Added Unity metadata: discrete_action_output_shape={DISCRETE_BRANCHES}")
    except ImportError:
        print(f"  WARNING: 'onnx' package not installed, skipping metadata.")
        print(f"  Install with: pip install onnx")
        print(f"  The model may not work in Unity without metadata.")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX validation: PASSED")

        # Print summary
        print(f"  Opset: {onnx_model.opset_import[0].version}")
        print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    except ImportError:
        pass
    except Exception as e:
        print(f"  ONNX validation: WARNING - {e}")

    file_size = os.path.getsize(output_path) / 1024
    print(f"  Output: {output_path} ({file_size:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export SB3 PPO models to Unity ML-Agents ONNX format")
    parser.add_argument("--env", type=str, required=True,
                        choices=["simple", "medium", "hard", "all"],
                        help="Which environment model to export")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip (default: models/{env}/final.zip)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: models/onnx)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(BASE_DIR, "models", "onnx")

    envs = ["simple", "medium", "hard"] if args.env == "all" else [args.env]
    exported = []

    for env_name in envs:
        model_path = args.model or os.path.join(
            BASE_DIR, "models", env_name, "final.zip"
        )
        if not os.path.exists(model_path):
            print(f"\nSkipping {env_name}: model not found at {model_path}")
            continue
        path = export_model(env_name, model_path, output_dir)
        exported.append(path)

    if exported:
        print(f"\n{'='*50}")
        print(f"Exported {len(exported)} model(s) to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Copy .onnx files to Unity project Assets/Models/")
        print(f"  2. In Unity, select AgentA in each scene")
        print(f"  3. Set BehaviorParameters > Behavior Type = 'Inference Only'")
        print(f"  4. Drag the .onnx file to BehaviorParameters > Model")
        print(f"  5. Build for Windows (File > Build Settings > Windows)")
    else:
        print("\nNo models exported. Check that model files exist.")


if __name__ == "__main__":
    main()
