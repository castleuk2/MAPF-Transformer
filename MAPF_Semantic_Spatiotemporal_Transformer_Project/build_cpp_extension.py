from mapf_sst.cpp import load_extension


if __name__ == "__main__":
    module = load_extension()
    print(f"SST C++ feature extension: OK ({module.__file__})")
