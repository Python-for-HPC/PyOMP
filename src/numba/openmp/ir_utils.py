def dump_block(label, block):
    print(label, ":")
    for stmt in block.body:
        print("    ", stmt)
