from impinj_reader import ImpinjReader  # Replace with actual SDK
reader = ImpinjReader('192.168.1.100')
reader.connect()
tags = reader.read_tags(timeout=1)
print(tags)