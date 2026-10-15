# ------------------------------------------------------------------------
# BasicSR Registry Implementation
# ------------------------------------------------------------------------

class Registry:
    """Registry for mapping strings to classes or functions."""

    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, obj=None):
        """Register the given object under its name or a specified name."""
        if obj is None:
            # Used as a decorator: @registry.register()
            def deco(func_or_class):
                name = func_or_class.__name__
                self._register_item(name, func_or_class)
                return func_or_class
            return deco

        # Used as a function call: registry.register(obj)
        name = obj.__name__
        self._register_item(name, obj)
        return obj

    def _register_item(self, name: str, item):
        if name in self._obj_map:
            # Allow overwriting or warning
            pass
        self._obj_map[name] = item

    def get(self, name: str):
        """Retrieve an object from the registry."""
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"'{name}' is not registered in '{self._name}'. Available: {list(self._obj_map.keys())}")
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def __repr__(self) -> str:
        table = f"Registry of {self._name}:\n"
        for k, v in self._obj_map.items():
            table += f"  {k}: {v}\n"
        return table


ARCH_REGISTRY = Registry("arch")
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
