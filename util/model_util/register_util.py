import os
import importlib


def register_model(model_path: str):
    models = [i.replace(".py", "") for i in os.listdir(model_path)]
    module_python = model_path.replace("/", ".") if "/" in model_path else model_path.replace("\\", ".")
    all_models = [(module_python, models), ]
    for base_dir, modules in all_models:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)
            except ImportError as e:
                raise e
