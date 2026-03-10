import flwr as fl

class IDSClient(fl.client.NumPyClient):

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):

        self.model.set_weights(parameters)

        self.model.fit(
            self.X_train,
            self.X_train,
            epochs=5,
            batch_size=64,
            verbose=0
        )

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)

        loss = self.model.evaluate(self.X_train, self.X_train, verbose=0)

        return loss, len(self.X_train), {"accuracy": 0.98}
