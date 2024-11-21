import argh

from datasets import NuScenesDataset as Dataset
from vdbfusion_pipeline import VDBFusionPipeline as Pipeline


def main(
    root_dir: str,
    sequence: int = 0,
    config: str = "config/nuscnes.yaml",
    n_scans: int = -1,
    jump: int = 0,
    visualize: bool = False,
):
    """Help here!"""
    dataset = Dataset(root_dir, sequence, config)
    print("Length of lidar_tokens:", len(dataset))
    pipeline = Pipeline(
        dataset, config, jump=jump, n_scans=n_scans, map_name=f"nuscenes_{sequence}"
    )
    pipeline.run()
    pipeline.visualize() if visualize else None


if __name__ == "__main__":
    argh.dispatch_command(main)