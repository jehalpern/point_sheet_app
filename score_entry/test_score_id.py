import os
def save_directory_tree(root_dir, output_file):
    with open(output_file, "w") as f:
        def recurse(path, indent=""):
            for item in sorted(os.listdir(path)):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    f.write(f"{indent}📁 {item}/\n")
                    recurse(full_path, indent + "    ")
                else:
                    f.write(f"{indent}📄 {item}\n")
        recurse(root_dir)

# Example:
save_directory_tree("../../point_sheet_app", "directory_structure.txt")
