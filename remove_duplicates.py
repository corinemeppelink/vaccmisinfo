#!/usr/bin/env python3
import csv

allurls = set()


with open('childhood.csv','r') as fi, open('childhood_dedup.csv','w') as fo:
    reader = csv.reader(fi)
    writer = csv.writer(fo)
    for row in reader:
        url = row[4]
        if url in allurls:
            print('skipping duplicate')
        else:
            writer.writerow(row)
            allurls.add(url)

allurls = set()
with open('hpv.csv','r') as fi, open('hpv_dedup.csv','w') as fo:
    reader = csv.reader(fi)
    writer = csv.writer(fo)
    for row in reader:
        url = row[4]
        if url in allurls:
            print('skipping duplicate')
        else:
            writer.writerow(row)
            allurls.add(url)
