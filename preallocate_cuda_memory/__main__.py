from . import occupy_all_devices_memory
from . import occupy_specific_device_memory

from typing import List


def show_menu() -> int:
    print("1. Occupy all available memory")
    print("2. Occupy memory on specific device(s)")
    print("4. Exit")
    choice = input("Enter your choice: ")
    try:
        choice_int = int(choice)
    except ValueError:
        return 0
    return choice_int


def get_cuda_devices() -> List[int]:
    while True:
        cuda_devices = input("Enter the CUDA device(s) to occupy memory: ")
        try:
            cuda_devices_int = [int(x) for x in cuda_devices.split(",")]
        except ValueError:
            print("Invalid input. Please enter a number " "or comma-separated numbers.")
        else:
            break
    return cuda_devices_int


def main() -> None:
    while True:
        choice = show_menu()
        match choice:
            case 1:
                occupy_all_devices_memory()
            case 2:
                cuda_devices = get_cuda_devices()
                for cuda_device in cuda_devices:
                    occupy_specific_device_memory(cuda_device)
            case 4:
                break
            case _:
                print("Invalid input. Please enter 1, 2 or 4.")


if __name__ == "__main__":
    main()
