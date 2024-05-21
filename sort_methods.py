import re


def sort_methods(source_file):
    # Read the source file
    with open(source_file, "r") as file:
        source_code = file.read()

    # Split the source code into lines
    lines = source_code.split("\n")

    # Initialize variables to store class definitions and method definitions
    class_defs = []
    method_defs = {}

    # Iterate through each line to extract class definitions and method definitions
    current_class = None
    for line in lines:
        if line.strip().startswith("class"):
            current_class = re.match(
                r"class\s+([a-zA-Z0-9_]+)\s*:", line.strip()
            ).group(1)
            class_defs.append(line)
            method_defs[current_class] = []
        elif line.strip().startswith("def") and current_class:
            method_defs[current_class].append(line)

    # Sort method definitions within each class alphabetically
    for class_name, methods in method_defs.items():
        sorted_methods = sorted(methods)
        method_defs[class_name] = sorted_methods

    # Reconstruct the source code with sorted method definitions
    sorted_source_code = ""
    for line in lines:
        if line.strip().startswith("class"):
            class_name = re.match(
                r"class\s+([a-zA-Z0-9_]+)\s*:", line.strip()
            ).group(1)
            sorted_source_code += line + "\n"
            sorted_source_code += "\n".join(method_defs[class_name]) + "\n"
        elif not line.strip().startswith("def"):
            sorted_source_code += line + "\n"

    # Write the sorted source code back to the file


#    with open(source_file, 'w') as file:
#        file.write(sorted_source_code)

if __name__ == "__main__":
    import sys

    source_file = sys.argv[1]
    sort_methods(source_file)
