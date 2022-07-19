import time

if __name__ == "__main__":
    print("Starting test program.")
    time.sleep(2)
    raise Exception("Purposely fail the program.")
