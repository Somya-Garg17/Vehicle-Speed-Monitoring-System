# main.py
import hydra
from ultralytics.yolo.utils import DEFAULT_CONFIG
from predict import DetectionPredictor, init_tracker, results
from util import write_csv

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def main(cfg):
    init_tracker()
    predictor = DetectionPredictor(cfg)
    predictor()
    write_csv(results, './results/output.csv')

if __name__ == "__main__":
    main()
