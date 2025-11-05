
from dag import self_heal

def main():
    print("Self-Healing Classification CLI — type 'quit' to exit\n")
    while True:
        text = input("Enter text: ")
        if text.lower() == "quit":
            print("Goodbye!")
            break
        self_heal(text)

if __name__ == "__main__":
    main()
