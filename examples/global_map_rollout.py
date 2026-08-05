from __future__ import annotations

from mapf_map_transformer import Action, GlobalMapWindowProvider, MapTokenRuntime, ModelConfig, StructuredMapTransformer
from mapf_map_transformer.synthetic import generate_global_map


global_map = generate_global_map(11, size=49)
provider = GlobalMapWindowProvider(global_map)
center = (24, 24)

model = StructuredMapTransformer(ModelConfig(dropout=0.0))
runtime = MapTokenRuntime(model, device="cpu")
runtime.reset(provider.crop(center))

new_center = (24, 25)
result = runtime.step(
    Action.RIGHT,
    moved=True,
    incoming_strip=provider.incoming_strip(new_center, Action.RIGHT),
)
print(result.latent_tokens.shape, result.reused)

wait_result = runtime.step(Action.WAIT, moved=False)
print(wait_result.latent_tokens.shape, wait_result.reused)
