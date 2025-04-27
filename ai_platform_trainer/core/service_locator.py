class ServiceLocator:
    _services: dict = {}

    @classmethod
    def register(cls, name: str, instance: object) -> None:
        cls._services[name] = instance

    @classmethod
    def get(cls, name: str) -> object:
        return cls._services[name]
