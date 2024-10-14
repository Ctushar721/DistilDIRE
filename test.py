import wandb
from utils.config import cfg

    
def main(run, cfg):
    print("cfg", vars(cfg))
    from utils.trainer import Trainer
    from torch.utils.data import DataLoader
    from dataset import TMEPSOnlyDataset, TMIMGOnlyDataset
    print(cfg.dataset_test_root)
    # dataset =TMEPSOnlyDataset(cfg.dataset_test_root, False)
    dataset = TMIMGOnlyDataset(cfg.dataset_test_root, istrain=False)
    dataloader = DataLoader(dataset, 
                            batch_size=100,
                            num_workers=4,
                            pin_memory=True)
    # shuffle=True, num_workers=2
    print("Dataloader done")
    trainer = Trainer(cfg, dataloader, dataloader, run, 0, False, 1)
    assert len(cfg.pretrained_weights) != 0, "Give proper checkpoint path"
    print("trainer loaded")
    trainer.load_networks(cfg.pretrained_weights)
    print("network loaded")
    trainer.validate(False, save=True, save_name=f"{cfg.root_dir}/{cfg.datasets_test}_{cfg.pretrained_weights.split('/')[-1]}_results.txt")

if __name__ == "__main__":
    main(None, cfg)
    