import flwr as fl

def start_server():

    strategy = fl.server.strategy.FedAvg()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=120),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
