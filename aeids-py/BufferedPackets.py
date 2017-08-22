from impacket import ImpactDecoder, ImpactPacket

class BufferedPackets:
    def __init__(self, first_frame):
        packet = first_frame.child()
        segment = packet.child

        if isinstance(segment, ImpactPacket.TCP):
            self.ready = False
        else:
            self.ready = True

        self.frames = []
        self.frames.append(first_frame)
        self.id = self.generate_id(first_frame)
        self.window_counter = 10
        self.start_time = 0
        self.read = False

    def check_counter(self):
        if self.window_counter <= 0:
            self.ready = True

    def generate_id(self, frame):
        packet = frame.child()
        segment = packet.child()

        src_addr = packet.get_ip_src()
        dst_addr = packet.get_ip_dst()

        if isinstance(segment, ImpactPacket.TCP):
            protocol = "tcp"
            src_port = segment.get_th_sport()
            dst_port = segment.get_th_dport()
            return "{}-{}-{}-{}-{}".format(src_addr, src_port, dst_addr, dst_port, protocol)
        elif isinstance(segment, ImpactPacket.UDP):
            protocol = "udp"
            src_port = segment.get_uh_sport()
            dst_port = segment.get_uh_dport()
            return "{}-{}-{}-{}-{}".format(src_addr, src_port, dst_addr, dst_port, protocol)
        elif isinstance(segment, ImpactPacket.ICMP):
            protocol = "icmp"
            return "{}-{}-{}".format(src_addr, dst_addr, protocol)

    def generate_reverse_id(self, frame):
        packet = frame.child()
        segment = packet.child()

        src_addr = packet.get_ip_src()
        dst_addr = packet.get_ip_dst()

        if isinstance(segment, ImpactPacket.TCP):
            protocol = "tcp"
            src_port = segment.get_th_sport()
            dst_port = segment.get_th_dport()
            return "{}-{}-{}-{}-{}".format(dst_addr, dst_port, src_addr, src_port, protocol)
        elif isinstance(segment, ImpactPacket.UDP):
            protocol = "udp"
            src_port = segment.get_uh_sport()
            dst_port = segment.get_uh_dport()
            return "{}-{}-{}-{}-{}".format(dst_addr, dst_port, src_addr, src_port, protocol)
        elif isinstance(segment, ImpactPacket.ICMP):
            protocol = "icmp"
            return "{}-{}-{}".format(dst_addr, src_addr, protocol)

    def add_frame(self, frame):
        if self.ready:
            return False

        id = self.generate_id(frame)
        rev_id = self.generate_reverse_id(frame)
        last_frame = self.frames[-1]
        new_packet = frame.child()
        new_segment = new_packet.child()
        last_packet = last_frame.child()
        last_segment = last_packet.child()

        #print id + ">>>>" + self.id
        self.window_counter -= 1

        if self.id == id and isinstance(new_segment, ImpactPacket.TCP):
            if new_segment.get_FIN():
                self.ready = True

            if new_segment.get_th_ack() != last_segment.get_th_ack(): #new frame belongs to a different flow
                self.check_counter()
                return False

            for i in range(0, len(self.frames)):
                f = self.frames[i]
                n = f.child()
                s = n.child()

                if new_segment.get_th_seq() == s.get_th_seq(): #retransmitted packet
                    self.window_counter += 1
                    return True
                elif new_segment.get_th_seq() < s.get_th_seq(): # out of order packet
                    self.frames.insert(i, frame)
                    self.window_counter += 1
                    return True

            self.frames.append(frame)
            self.window_counter += 1
            return True
        elif self.id == rev_id and isinstance(new_segment, ImpactPacket.TCP): #frame belongs to the opposite flow
            if new_segment.get_th_seq() == last_segment.get_th_ack():
                self.ready = True
            elif new_segment.get_FIN():
                self.ready = True

            return False
        elif not isinstance(new_segment, ImpactPacket.TCP):
            self.check_counter()
            return True
        else:
            self.check_counter()
            return False

    def get_payload(self):
        payload = ""
        for frame in self.frames:
            packet = frame.child()
            segment = packet.child()
            if segment.get_data_as_string() is not None:
                payload += segment.get_data_as_string()

        return payload

    def get_byte_frequency(self):
        byte_frequency = [0] * 256
        payload = self.get_payload()
        length = float(self.get_payload_length())

        for i in range(0, 256):
            byte_frequency[i] = float(payload.count(chr(i))) / length

        return byte_frequency

    def get_payload_length(self):
        length = 0
        for frame in self.frames:
            packet = frame.child()
            segment = packet.child()
            if isinstance(segment, ImpactPacket.TCP):
                length += len(segment.get_data_as_string())
            elif isinstance(segment, ImpactPacket.UDP):
                length += segment.get_uh_ulen()

        return length


