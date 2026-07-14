from pathlib import Path

from diffusion_conductivity import get_replica_specs, load_toml


def test_lifsi_example_uses_shared_topology_replicas() -> None:
    path = Path(__file__).parents[1] / "examples" / "lifsi_dme_tol.toml"
    config = load_toml(path)
    replicas = get_replica_specs(config)

    assert len(replicas) == 3
    assert all(item["topology"] == "traj.tpr" for item in replicas)
    assert config["ions"]["species"][0]["name"] == "Li"
