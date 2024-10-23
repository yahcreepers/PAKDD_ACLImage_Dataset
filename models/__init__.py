from .llava import LLAVA

M_LIST = {
    "llava": LLAVA, 
}

def prepare_model(args):
    model = M_LIST[args.model]
    model = model.build_model(args)
    return model