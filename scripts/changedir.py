import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, help="specify the file path")
    parser.add_argument("--change_mo", required=True, help="new dir to add to the end")
    parser.add_argument("--output_fd", default=".")
    args = parser.parse_args()

    root = os.path.realpath(args.file_path)
    name = os.path.basename(root)
    ndir = args.change_mo
    src_file = open(root, 'r')
    dst_file = open(os.path.join(args.output_fd, name), 'w')

    for line in src_file:
        parts = line.rstrip().split('/')
        dst_file.write(os.path.join(*parts[:-1], ndir, parts[-1])+'\n')

