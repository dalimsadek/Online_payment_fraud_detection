


def save_model(model, path):
    """Save the model to the specified path."""
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    """Load the model from the specified path."""
    with open(path, 'rb') as file:
        return pickle.load(file)