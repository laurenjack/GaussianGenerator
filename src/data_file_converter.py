import struct

"""
Members of this module are responsible for taking a data set
file and accompanying each instance with a unique id
"""

def give_instances_id(filenames, xy_size):
    for filename in filenames:
        give_instances_id_single(filename, xy_size)

def give_instances_id_single(filename, xy_size):
    out_name = _get_outfile_name(filename)
    out_file = open(out_name, "wb")

    ID = 0
    with open(filename, "rb") as in_file:
        inst = in_file.read(xy_size)
        while inst != '':
            raw_id = struct.pack('I', ID)
            if ID == 8000:
                print struct.unpack("4b", raw_id)
            id_and_inst = raw_id + inst
            out_file.write(id_and_inst)
            inst = in_file.read(xy_size)
            ID += 1
    print ID
    out_file.close()


def _get_outfile_name(in_name):
    last_dot = in_name.rfind('.')
    suffix = in_name[last_dot:]
    body = in_name[:last_dot]
    body += '_id'
    return body + suffix

if __name__ == '__main__':
    give_instances_id('data_batch_2.bin', 3073)

