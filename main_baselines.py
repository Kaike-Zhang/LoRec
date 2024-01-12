from datetime import datetime
from meta_config import args
from utls.model_config import *
from utls.baslines_trainer import *
from utls.utilize import init_run, restore_stdout_stderr
import os

def main(seed=2023, main_file=""):
    # Initialize the log path & seed
    args.seed = seed
    if args.use_LLM:
        path = f"{args.model}/{args.LLM}/{args.dataset}/{main_file}/{args.baseline}" if main_file != "" else f"./log/{args.model}/{args.LLM}/{args.dataset}/{args.baseline}"
    else:
        path = f"{args.model}/No_LLM/{args.dataset}/{main_file}/{args.baseline}"  if main_file != "" else f"./log/{args.model}/No_LLM/{args.dataset}/{args.baseline}"
    init_run(log_path=os.path.join('./baseline_log', path), args=args, seed=args.seed)

    glo = globals()
    global_config = vars(args)
    global_config["main_file"] = main_file

    # Baseline config
    global_config["baseline_config"] = glo[f"get_{global_config['baseline']}_config"](global_config)

    # Backbone Model config
    global_config["model_config"] = glo[f"get_{global_config['model']}_config"](global_config)
    global_config['checkpoints'] = 'baseline_checkpoints'
    global_config["with_fd"] = False if global_config['baseline'] != "GraphRfi" else True

    # Initialize correspondding trainer 
    trainer =  glo[f"{global_config['baseline']}4{global_config['model']}Trainer"](global_config)
    trainer.train(path=os.path.join('./baseline_checkpoints', path))

    restore_stdout_stderr()


if __name__ == '__main__':
    main_file = datetime.now().strftime('%Y%m%d')
    main(seed=2023, main_file=main_file)
