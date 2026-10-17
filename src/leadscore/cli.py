import argparse
import logging

from .config import load_config

STEPS = ["ingest", "eda", "train", "evaluate", "predict", "monitor"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="leadscore", description="Lead scoring pipeline")
    parser.add_argument("command", choices=["init-db", *STEPS, "all"])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    # xgboost/matplotlib are chatty at INFO
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    cfg = load_config(args.config)

    from .db import get_engine, init_schema
    engine = get_engine()

    # predict already runs the monitoring checks for its batch
    steps = STEPS[:-1] if args.command == "all" else [args.command]
    for step in steps:
        logging.getLogger("leadscore").info("==> %s", step)
        if step == "init-db":
            init_schema(engine)
        elif step == "ingest":
            from .ingest import load_csv_to_postgres
            load_csv_to_postgres(engine, cfg)
        elif step == "eda":
            from .eda import run_eda
            run_eda(engine, cfg)
        elif step == "train":
            from .train import run_training
            run_training(engine, cfg)
        elif step == "evaluate":
            from .evaluate import run_evaluation
            run_evaluation(engine, cfg)
        elif step == "predict":
            from .predict import run_scoring
            run_scoring(engine, cfg)
        elif step == "monitor":
            from .monitoring import monitor_latest_run
            monitor_latest_run(engine, cfg)


if __name__ == "__main__":
    main()
