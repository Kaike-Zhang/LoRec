from meta_config import args
from utls.model_config import *
from utls.trainer import *
from utls.utilize import init_run, restore_stdout_stderr

def main(seed=2023, main_file=""):
    args.seed = seed

    # Initialize the log path & seed
    if args.use_LLM:
        path = f"./log/{args.model}/{args.LLM}/{args.dataset}/{main_file}/" if main_file != "" else f"./log/{args.model}/{args.LLM}/{args.dataset}/"
        if args.with_lct:
            path = f"./log/{args.model}/{args.LLM}/{args.dataset}/LCT/{main_file}/" if main_file != "" else f"./log/{args.model}/{args.LLM}/{args.dataset}/FD/"
    else:
        path = f"./log/{args.model}/No_LLM/{args.dataset}/{main_file}/"  if main_file != "" else f"./log/{args.model}/No_LLM/{args.dataset}/"
    init_run(log_path=path, args=args, seed=args.seed)

    glo = globals()
    global_config = vars(args)
    global_config["main_file"] = main_file

    # LCT config
    global_config["lct_config"] = glo["get_LCT_config"](global_config)
    
    # Backbone Model config
    global_config["model_config"] = glo[f"get_{global_config['model']}_config"](global_config)
    global_config['checkpoints'] = 'checkpoints'

    # Initialize correspondding trainer 
    trainer =  glo[f"{global_config['model']}Trainer"](global_config)
    trainer.train()
    
    restore_stdout_stderr()


if __name__ == '__main__':
    main_file = datetime.now().strftime('%Y%m%d')
    main(seed=2023, main_file=main_file)

