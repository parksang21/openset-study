

def get_module(name):
    if name == "base":
        from pl_module.base import BasePL
        return BasePL

    elif name == "simclr":
        pass

    elif name == "supsimclr":
        from pl_module.sup_simclr import SupSimModule
        return SupSimModule

    else:
        raise Exception("No valid module name")
