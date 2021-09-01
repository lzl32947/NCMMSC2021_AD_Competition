import os
import importlib


def register_model(model_path: str) -> None:
    """
    Auto-register models in given path into Registers
    :param model_path: str, path to model
    :return: None
    """
    # Get all python modules in model_path
    models = [i.replace(".py", "") for i in os.listdir(model_path)]
    # Get the package names
    module_python = model_path.replace("/", ".") if "/" in model_path else model_path.replace("\\", ".")
    # Generate the imports
    all_models = [(module_python, models), ]
    # Import modules and auto-register
    for base_dir, modules in all_models:
        for name in modules:
            try:
                # Check base path
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                # Import module
                importlib.import_module(full_name)
            except ImportError as e:
                raise e
