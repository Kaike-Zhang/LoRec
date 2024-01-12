def get_LCT_config(org_config):
    config = {
        "LLM_size": int(org_config["LLM_size"]),
        "ffd_hidden": 128, 
        "hidden_units": int(org_config["out_dim"]), 
        "dnn_layer": 0,
        "dropout_rate": 0.1,
        "heads": 2,
        "block_num": 2,
        "max_len": 60
        }
    return config


def get_SASrec_config(org_config):
    config = {
        "device": org_config["device"],
        "hidden_units": int(org_config["out_dim"]), 
        "maxlen": 55,
        "dropout_rate": 0.1,
        "num_heads": 2,
        "num_blocks": 2,
        "dnn_layer": 0
        }
    return config


def get_FMLPrec_config(org_config):
    config = {
        "device": org_config["device"],
        "hidden_units": int(org_config["out_dim"]), 
        "maxlen": 50,
        "dropout_rate": 0.5,
        "num_heads": 2,
        "num_blocks": 2,
        "dnn_layer": 0,
        "no_filters": False,
        "initializer_range": 0.02
        }
    return config

def get_GRU4rec_config(org_config):
    config = {
        "device": org_config["device"],
        "hidden_units": int(org_config["out_dim"]), 
        "maxlen": 50,
        "dropout_rate": 0.5,
        "num_heads": 2,
        "num_blocks": 2,
        "dnn_layer": 0,
        "batch_size": org_config["batch_size"]
        }
    return config

# baselines:
def get_GraphRfi_config(org_config):
    config = {
        "device": org_config["device"],
        "hidden_units": int(org_config["out_dim"]), 
        "n_trees": 5,
        }
    return config

def get_APR_config(org_config):
    config = {
        "device": org_config["device"],
        "l2_reg": 0,
        "adv_reg": 0.1,
        "eps": 0.02,
        }
    return config

def get_Detection_config(org_config):
    config = {
        "device": org_config["device"],
        }
    return config

def get_LLM4Dec_config(org_config):
    config = {
        "device": org_config["device"],
        }
    return config

def get_Denoise_config(org_config):
    config = {
        "device": org_config["device"],
        }
    return config

def get_CL4rec_config(org_config):
    config = {
        "device": org_config["device"],
        "hidden_units": int(org_config["out_dim"]), 
        "maxlen": 55,
        "dropout_rate": 0.1,
        "num_heads": 2,
        "num_blocks": 2,
        "dnn_layer": 0,
        "lambda": 0.1,
        }
    return config

def get_ADV_config(org_config):
    config = {
        "device": org_config["device"],
        "l2_reg": 0,
        "adv_reg": 0.1,
        "eps": 0.1,
        }
    return config