r"""
An asyncio-powered folder flattener.

Author: @jerryc05 - https://github.com/jerryc05

Example Usage:
--------------
python flatten_folder.py \
       -i ${PATH_TO_FOLDER_TO_FLATTEN} \
       -o ${PATH_TO_OUTPUT_FOLDER}
"""

import argparse
import asyncio
import os
import shutil


async def main():
    arg_parser = argparse.ArgumentParser(description='An asyncio-powered folder flattener.')
    arg_parser.add_argument('-i', '--input', required=True,
                            help='the path to folder to flatten.')
    arg_parser.add_argument('-o', '--output', required=True,
                            help='the path to output folder.')
    args = arg_parser.parse_args()

    input_path: str = args.input.strip()
    output_path: str = args.output.strip()

    print(f'Processing {input_path}!')

    tasks = []

    async def move(src, dst, name):
        shutil.move(os.path.join(src, name),
                    os.path.join(dst, name))

    for subdir, dirs, files in os.walk(input_path):
        for file in files:
            tasks.append(asyncio.create_task(
                move(subdir, output_path, file)))
    files_count = len(tasks)

    for task in tasks:
        await task

    print('Flattening folders successful!')
    print(f'Processed {files_count} files in total!')


asyncio.run(main())
# flatten_folder.py -i F:\PycharmProjects\Hello-Object-Detection\data\train\data -o F:\PycharmProjects\Hello-Object-Detection\data\train\data