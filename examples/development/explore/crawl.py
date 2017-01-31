from os.path import expanduser

from sacred.observers import TinyDbReader

reader = TinyDbReader(expanduser('~/runs'))
report = reader.fetch_report('decompose_images', indices=-1)
metadata = reader.fetch_metadata('decompose_images', indices=-1)
print(metadata[0]['info'])
import sys
sys.stdout.write(report[0])