import torch


if __name__ == "__main__":
    gpu_flag = torch.cuda.is_available()
    intro = "This is the test script to verify that workspace is operational."
    message = f"CUDA availability: {gpu_flag}"
    print(message)
