import toml
import os
from pathlib import Path

def load_config():
    """
    Retrieve classes colours and weights from config TOML

    Returns:
    Tuple: Tuple of class names, Tuple of class colours, Tuple of class weights
    """

    conf_file = os.environ.get("BIRDECTOR_CLASSES_CONF", None)
    if not conf_file:
        conf_file = Path(__file__).parent.resolve() / "classes.toml"
    
    return toml.load(conf_file)
