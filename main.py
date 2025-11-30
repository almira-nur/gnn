def startup():
    import torch
    
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            global device
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')


def main():
    print("Hello from gnn!")


if __name__ == "__main__":
    main()
