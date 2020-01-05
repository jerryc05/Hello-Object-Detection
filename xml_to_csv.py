r"""
An asyncio-powered xml to vsc parser for labelImg.

Author: @jerryc05 - https://github.com/jerryc05

Example Usage:
--------------
python xml_to_csv.py \
       -i ${PATH_TO_XML_FOLDER} \
       -o ${PATH_TO_CSV_FOLDER}
"""

import argparse
import asyncio
import csv
import glob
import os
import xml.etree.ElementTree as xmlETree


async def main():
    arg_parser = argparse.ArgumentParser(description='An asyncio-powered xml to vsc parser for labelImg.')
    arg_parser.add_argument('-i', '--input', required=True,
                            help='the path to folder containing input xml files.')
    arg_parser.add_argument('-o', '--output', required=True,
                            help='the path to folder containing output csv file.')
    args = arg_parser.parse_args()

    xml_path: str = args.input.strip()
    csv_path: str = args.output.strip()

    print(f'Processing {xml_path}!')
    xml_q = asyncio.Queue()
    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    async def process_xml_file(path):
        for box in xmlETree.parse(path).getroot().findall('object'):
            xml_q.put_nowait((
                xmlETree.parse(path).getroot().find('filename').text.strip(),  # filename
                xmlETree.parse(path).getroot().find('size')[0].text.strip(),  # width
                xmlETree.parse(path).getroot().find('size')[1].text.strip(),  # height
                box[0].text.strip(),  # box.name/class
                box[4][0].text.strip(),  # box.xmin
                box[4][1].text.strip(),  # box.ymin
                box[4][2].text.strip(),  # box.xmax
                box[4][3].text.strip(),  # box.ymax
            ))

    tasks = []
    for xml_file in glob.glob(os.path.join(xml_path, '*.xml')):
        tasks.append(asyncio.create_task(process_xml_file(xml_file)))
    for task in tasks:
        await task
    tasks.clear()

    # split train set and eval set into 1:3
    xml_size = xml_q.qsize()
    if xml_size == 0:
        print('WARNING! No boxes processed!')
        print(f'Please check your xml input path again: [{xml_path}]!')
        exit(1)
    split_index = int(xml_size / 4)

    # save to csv
    async def save_to_csv(writer):
        writer.writerow(xml_q.get_nowait())

    with open(os.path.join(csv_path, 'train.csv'), 'w', newline='') as train_f, \
            open(os.path.join(csv_path, 'eval.csv'), 'w', newline='') as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        train_writer.writerow(columns)
        eval_writer.writerow(columns)

        for i in range(xml_size):
            if i > split_index:
                tasks.append(asyncio.create_task(save_to_csv(train_writer)))
            else:
                tasks.append(asyncio.create_task(save_to_csv(eval_writer)))
        for task in tasks:
            await task

    print('XML -> CSV successful!')
    print(f'Processed {xml_size} boxes in total!')


asyncio.run(main())
