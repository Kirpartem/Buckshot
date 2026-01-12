import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.preprocessing import get_obs_shape

# 1. Load the model
model_path = "agent/models/champion.zip"
model = MaskablePPO.load(model_path)

# 2. Extract the Policy (the actual neural network)
# This ignores the optimizer, learning rate schedule, etc.
policy = model.policy.to("cpu")

# 3. Define Input Dimensions
# Replace these with your actual observation space shape
# Example: (1, 64) for a flat vector or (1, 3, 64, 64) for an image
obs_shape = (1, *get_obs_shape(model.observation_space))
dummy_input = torch.randn(obs_shape)  # type: ignore

# 4. Export to ONNX
onnx_file = "model.onnx"


# We wrap the policy in a small class to ensure only the forward pass (actor) is exported
class OnnxWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        features = self.policy.extract_features(observation)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features = self.policy.pi_features_extractor(features)
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        return self.policy.action_net(latent_pi)


wrapped_model = OnnxWrapper(policy)

torch.onnx.export(
    wrapped_model,
    dummy_input,
    onnx_file,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model successfully converted to {onnx_file}")
