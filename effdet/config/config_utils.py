from omegaconf import OmegaConf


def set_config_readonly(conf):
    OmegaConf.set_readonly(conf, True)


def set_config_writeable(conf):
    OmegaConf.set_readonly(conf, False)
