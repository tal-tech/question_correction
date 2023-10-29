class SingletonIns(object):
    _instance = {}
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self):
        if self._cls not in self._instance.keys():
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]
