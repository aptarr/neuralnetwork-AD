import csv
import subprocess
import os

pcap_file_path = "/home/apta/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Port80.pcap"
output_directory = "/home/apta/CIC-IDS-2017/PCAPs/filter-friday"
csv_file_path = (
    "/home/apta/CIC-IDS-2017/CSVs/TrafficLabelling/Friday_port_80_filtered.csv"
)
batch_size = 1000


def read_csv(csv_file_path):
    filters = []
    with open(csv_file_path, mode="r", encoding="ISO-8859-1") as file:
        reader = csv.DictReader(file)

        reader.fieldnames = [field.strip() for field in reader.fieldnames]

        for row in reader:
            filters.append(
                {
                    "src_ip": row["Source_IP"],
                    "src_port": row["Source_Port"],
                    "dst_ip": row["Destination_IP"],
                    "dst_port": row["Destination_Port"],
                }
            )
    return filters


def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def run_tcpdump_batch(pcap_file, batch_filters, output_file):
    filter_parts = [
        f"(src host {f['src_ip']} and src port {f['src_port']} and dst host {f['dst_ip']} and dst port {f['dst_port']})"
        for f in batch_filters
    ]
    filter_string = " or ".join(filter_parts)

    command = ["tcpdump", "-r", pcap_file, "-w", output_file, filter_string]
    subprocess.run(command, check=True)


def process_single_pcap_file():
    os.makedirs(output_directory, exist_ok=True)
    filters = read_csv(csv_file_path)
    batches = list(chunk_list(filters, batch_size))

    for i, batch in enumerate(batches, start=1):
        output_filename = f"batch_{i}.pcap"
        output_path = os.path.join(output_directory, output_filename)
        print(
            f"Processing batch {i}/{len(batches)} → {output_filename} with {len(batch)} filters"
        )
        run_tcpdump_batch(pcap_file_path, batch, output_path)


if __name__ == "__main__":
    process_single_pcap_file()
