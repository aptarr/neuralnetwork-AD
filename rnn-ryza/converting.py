import binascii
import math
import numpy
import os
import sys
import time
import traceback
sys.path.insert(0, '../aeids-py/')


from PcapReaderThread import PcapReaderThread
from StreamReaderThread import StreamReaderThread

done=False
prt=None
conf = {}
MAX_SEQ_LEN = 3960


def main(argv):
    try:

        if sys.argv[1] != "training" and sys.argv[1] != "testing":
            raise IndexError("Phase {} does not exist.".format(sys.argv[1]))
        else:
            phase = sys.argv[1]
            if sys.argv[2] != "tcp" and sys.argv[2] != "udp":
                raise IndexError("Protocol {} is not supported.".format(sys.argv[3]))
            else:
                protocol = sys.argv[2]

            if not sys.argv[3].isdigit():
                raise IndexError("Port must be numeric.")
            else:
                port = sys.argv[3]

            if not sys.argv[4].isdigit():
                raise IndexError("Sequence length must be numeric.")
            else:
                seq_length = int(sys.argv[4])

            testing_filename = sys.argv[5]
            read_conf()
            count_and_convert(phase, protocol, port, testing_filename, seq_length)
    except IndexError as e:
         print(e)
	 # traceback.print_exc()
         print("Usage: python rnnids.py <tcp|udp> <port> <seq_lenght> <training filename>")
    except KeyboardInterrupt:
        if prt is not None:
            prt.done = True
        done = True

def count_and_convert(phase, protocol, port, testing_filename, seq_length):
    global done
    check_directory("texts")
    fmean = open("texts/{}.txt".format(testing_filename), "w")
    if phase == 'training':
        for elem in byte_seq_generator(phase, testing_filename, protocol, port, seq_length):
            fmean.write("{}\n".format(elem))
    else:
        for id, elem in byte_seq_generator(phase, testing_filename, protocol, port, seq_length):
            fmean.write("{}; {}\n".format(id, elem))
    fmean.close()
    print("Finished")
    done = True

def read_conf():
    global root_directory
    global threshold_method

    fconf = open("rnnids.conf", "r")
    if not fconf:
        print ("File rnnids.conf does not exist.")
        exit(-1)

    conf["root_directory"] = []
    conf["training_filename"] = {"default-80-5": 100000}
    lines = fconf.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        split = line.split("=", 2)
        print (split)
        if split[0] == "root_directory":
            conf["root_directory"].append(split[1].strip())
        elif split[0] == "training_filename":
            tmp = split[1].split(":")
            conf["training_filename"]["{}-{}-{}".format(tmp[0], tmp[1], tmp[2])] = int(tmp[3])

    fconf.close()

def byte_seq_generator(phase, filename, protocol, port, seq_length):
    global prt
    global root_directory
    print(filename)
    stream_counter = 0
    counter = 0
    prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)
    prt.start()
    prt.delete_read_connections = True

    while not prt.done or prt.has_ready_message():
        if not prt.has_ready_message():
            prt.wait_for_data()
            continue
        else:
            buffered_packet = prt.pop_connection()
            if buffered_packet is None:
                prt.wait_for_data()
                continue

            if buffered_packet.get_payload_length("server") > 0:

                payload = buffered_packet.get_payload("server")[:MAX_SEQ_LEN]
                payload_length = len(payload)

                if counter == 0:
                    print(payload)

                stream_counter += 1
                counter += (payload_length - seq_length) + 1
                sys.stdout.write("\r{} streams, {} sequences.".format(stream_counter, counter))
                sys.stdout.flush()

                counter +=1

                if phase == 'training':
                    yield payload
                else:
                    yield buffered_packet.id, payload

                    # if len(texts) >= (batch_size*seq_length):
                    #     fixed_texts = texts[:batch_size*seq_length]
                    #     remainder = texts[batch_size*seq_length:]
                    # else:
                    #     remainder = texts


                    # if ((len(fixed_texts)%(batch_size*seq_length) == 0) & (len(fixed_texts)>0)):
                    #     modified_sentence = ''.join([char + ' ' if (i + 1) % seq_length == 0 else char for i, char in enumerate(fixed_texts)])
                    #     fixed_texts = ''
                    #     yield modified_sentence


def check_directory(filename, root = "/Users/angelaoryza/Documents/TA/noisy-rnnids/rnnids-py"):
    if not os.path.isdir("{}/{}".format(root, filename)):
        os.mkdir("{}/{}".format(root, filename))


def get_pcap_file_fullpath(filename):
    global conf
    for i in range(0, len(conf["root_directory"])):
        if os.path.isfile(conf["root_directory"][i] + filename):
            return conf["root_directory"][i] + filename


if __name__ == '__main__':
	main(sys.argv)