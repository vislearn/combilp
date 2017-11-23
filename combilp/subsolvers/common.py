class SubSolver:
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = self.DEFAULT_PARAMETERS.copy()
        if parameters:
            self.parameters.update(parameters)

    def lower_bound(self):
        raise NotImplementedError

    def upper_bound(self):
        raise NotImplementedError

    def labeling(self):
        raise NotImplementedError
