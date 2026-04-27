import os
import torch
from main import build_model, load_data, generate

torch.set_grad_enabled(False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data + model
    _, _, vocab_size, encode, decode = load_data("input.txt")

    model = build_model(vocab_size, device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    # inputs from ECS
    start = os.getenv("START", "Shubham Education")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "100"))

    # generate
    with torch.no_grad():
        text = generate(
            model=model,
            encode=encode,
            decode=decode,
            device=device,
            start=start,
            max_new_tokens=max_new_tokens
        )

    print(text, flush=True)   # 🔥 important for logs

if __name__ == "__main__":
    main()