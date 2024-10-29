from .llava_next import LLAVA_NEXT

M_LIST = {
    "llava_next": LLAVA_NEXT, 
}

def prepare_model(args):
    model = M_LIST[args.model]
    model = model.build_model(args)
    return model