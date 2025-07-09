import ast


def clean_data_http(input_file, output_file_with_id, output_file_without_id):
    seen = set()
    dropped_count = 0

    http_methods = [
        "GET",
        "POST",
        "HEAD",
        "PUT",
        "DELETE",
        "OPTIONS",
        "PATCH",
        "TRACE",
        "CONNECT",
        "FAKEVERB",
    ]

    with (
        open(input_file, "r") as infile,
        open(output_file_with_id, "w") as outfile_with_id,
        open(output_file_without_id, "w") as outfile_without_id,
    ):
        for line in infile:
            id_data = line.strip().split(";", 1)
            if len(id_data) != 2:
                continue  # Skip malformed lines

            id_value, data = id_data
            data = data.strip()

            # Skip lines with payload exactly "['\x00']"
            if data in ["['\\x00']", "['\x00']"]:
                continue

            try:
                char_list = ast.literal_eval(data)
                if not isinstance(char_list, list):
                    continue

                full_str = "".join(char_list).lstrip().upper()

                if not any(full_str.startswith(method) for method in http_methods):
                    continue

                ascii_values = [ord(c) for c in char_list]
                encoded_data = str(ascii_values)

                updated_line = f"{id_value}; {encoded_data}\n"
                id_data_tuple = (id_value, encoded_data)

                if id_data_tuple not in seen:
                    seen.add(id_data_tuple)
                    outfile_with_id.write(updated_line)
                    outfile_without_id.write(encoded_data + "\n")
                else:
                    dropped_count += 1

            except (ValueError, SyntaxError):
                continue

    print(f"Total dropped duplicates: {dropped_count}")


input_file = "texts/Friday-WorkingHours_testing_80.txt"
output_file_with_id = "texts_cleaned/Friday-WorkingHours_testing_80.txt"
output_file_without_id = "texts_cleaned/Friday-WorkingHours_training_80.txt"

clean_data_http(input_file, output_file_with_id, output_file_without_id)


# ===============================================================================


def clean_data_ftp(input_file, output_file):
    dropped_count = 0
    kept_lines = []

    with open(input_file, "r") as infile:
        for line in infile:
            try:
                parts = line.split(";")
                if len(parts) < 2:
                    kept_lines.append(line)
                    continue

                raw_list = ast.literal_eval(parts[1].strip())

                first_three = "".join(raw_list[:3])

                print(f"{repr(first_three)} = {first_three.isdigit()}")

                if len(first_three) == 3 and first_three.isdigit():
                    dropped_count += 1
                    continue

                kept_lines.append(line)

            except Exception as e:
                print(f"Error processing line: {line.strip()} -> {e}")
                kept_lines.append(line)
                continue

    with open(output_file, "w") as outfile:
        outfile.writelines(kept_lines)

    print(f"Packets dropped: {dropped_count}")
    print(f"Packets kept and saved to '{output_file}': {len(kept_lines)}")


input_file = "texts/Friday-WorkingHours_testing_21.txt"
output_file = "cleaned_Friday-WorkingHours_testing_21.txt"

clean_data_ftp(input_file, output_file)
