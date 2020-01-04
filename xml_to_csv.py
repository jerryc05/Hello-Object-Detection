# An asyncio-powered xml to vsc parser for labelImg.
# Author: @jerryc05 - https://github.com/jerryc05

import argparse
import asyncio
import csv
import glob
import xml.etree.ElementTree as xmlETree


async def main():
    arg_parser = argparse.ArgumentParser(description='An xml to vsc parser for labelImg.')
    arg_parser.add_argument('-i', '--input', required=True,
                            help='the path to folder containing input xml files.')
    arg_parser.add_argument('-o', '--output', required=True,
                            help='the path to folder containing output csv file.')
    args = arg_parser.parse_args()

    xml_path: str = args.input.strip()
    if xml_path.endswith('/') or xml_path.endswith('\\'):
        xml_path = xml_path[:-1]
    csv_path: str = args.output.strip()
    if csv_path.endswith('/') or csv_path.endswith('\\'):
        csv_path = csv_path[:-1]

    print(f'Processing {xml_path}!')
    xml_q = asyncio.Queue()
    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    async def process_xml_file(path):
        root = xmlETree.parse(path).getroot()
        for box in root.findall('object'):
            value = (
                root.find('filename').text.strip(),  # filename
                root.find('size')[0].text.strip(),  # width
                root.find('size')[1].text.strip(),  # height
                box[0].text.strip(),  # box.name/class
                box[4][0].text.strip(),  # box.xmin
                box[4][1].text.strip(),  # box.ymin
                box[4][2].text.strip(),  # box.xmax
                box[4][3].text.strip(),  # box.ymax
            )
            xml_q.put_nowait(value)

    tasks = []
    for xml_file in glob.glob(f'{xml_path}/*.xml'):
        tasks.append(asyncio.create_task(process_xml_file(xml_file)))
    for task in tasks:
        await task

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

    with open(f'{csv_path}/train.csv', 'w', newline='') as train_f, \
            open(f'{csv_path}/eval.csv', 'w', newline='') as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        train_writer.writerow(columns)
        eval_writer.writerow(columns)

        tasks = []
        for i in range(xml_size):
            if i > split_index:
                tasks.append(asyncio.create_task(save_to_csv(train_writer)))
            else:
                tasks.append(asyncio.create_task(save_to_csv(eval_writer)))
        for task in tasks:
            await task

    print('XML -> CSV successful!')
    print('Processed', xml_size, 'boxes in total!')


asyncio.run(main())
