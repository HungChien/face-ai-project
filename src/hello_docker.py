import platform
import sys


def main() -> None:
    print("Hello from Docker!")
    print("The face-ai-project container is running successfully.")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")


if __name__ == "__main__":
    main()