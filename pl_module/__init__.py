

def get_module(name):
    if name == "base":
        from pl_module.base import BasePL
        return BasePL

    elif name == "simclr":
        pass

    elif name == "supsimclr":
        from pl_module.sup_simclr import SupSimModule
        return SupSimModule

    elif name == 'ce_supsim':
        from pl_module.ce_sup_sim import CESupSim
        return CESupSim
    else:
        raise Exception("No valid module name")
