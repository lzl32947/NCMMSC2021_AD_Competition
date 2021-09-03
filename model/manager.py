from typing import KeysView, Any, Callable


class Register(object):
    def __init__(self, register_name) -> None:
        # Init the dict for registration
        self._dict = {}
        # The register name
        self.name = register_name

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Override __setitem__
        :param key: Any, the key stored in dict.
        :param value: Any, but Callable is necessary for later function, the class stored in dict.
        :return: None.
        """
        # Check callable
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        # Check key
        if key is None:
            key = value.__name__
        # Check duplication
        if key in self._dict:
            raise Exception("Key already exist in {}, check if the two class with the same name.".format(self.name))
        # Store in dict
        self._dict[key] = value

    def register(self, param):
        """
        Main function of Register, serve as the decorator to register classes.
        """

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            return decorator(None, param)
        return lambda x: decorator(param, x)

    def __getitem__(self, key: Any) -> Callable:
        """
        Override __getitem__
        :param key: Any, the key in dict
        :return: Callable, the class to be instantiated
        """
        try:
            return self._dict[key]
        except Exception as e:
            raise e

    def __contains__(self, key: Any) -> bool:
        """
        Override __contains__
        :param key: Any, the key in dict
        :return: bool, whether this key is in the dict.keys()
        """
        return key in self._dict

    def keys(self) -> KeysView:
        """
        Get all keys in dicts
        :return: Set(keys)
        """
        return self._dict.keys()


class Registers:
    """
    Registers class should not be used to instantiate!
    """
    # Register all used models
    model = Register("model")
    # Register all used models
    module = Register("modules")

    # Raise error when init this class
    def __init__(self):
        raise RuntimeError("Register can not be instantiated!")
