import csv

columns_names = []
columns_map = {}
columns_counters = {}

parsed = open("./word2vec/parsed.csv", "w")
mapper = open("./word2vec/mapper.txt", "w")

# Building mappers and mapping
print("==> Parsing the csv to word2vec/parsed.csv")
with open('200_auto_policies.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    # Considering first line as headers
    headers = next(reader)
    parsed.write(",".join(headers) + "\n")

    i = 0
    for header in headers:
        columns_names.append(header)
        columns_map[header] = {}
        columns_counters[header] = 1
        i += 1

    for row in reader:
        column_i = 0
        parsed_row = []
        for column in row:

            # If it's something parseable
            if column and not column.isdigit():
                header = columns_names[column_i]

                # Add to map, if do not exist yet
                if not column in columns_map[header]:
                    columns_map[header][column] = columns_counters[header]
                    columns_counters[header] += 1

                # save the parsed version
                parsed_row.append(str(columns_map[header][column]))

            # If empty, save 0
            elif column == "":
                parsed_row.append("0")

            # Otherwise, save the original
            else:
                parsed_row.append(column)

            # and let's go to the next column
            column_i += 1

        parsed.write(",".join(parsed_row) + "\n")
print("==> DONE!")

print("==> Saving the mapper file to word2vec/mapper.txt")
for column, list in columns_map.items():
    if not list:
        continue
    mapper.write(column + "\n")
    for k, v in list.items():
        mapper.write("\t" + k + " - " + str(v) + "\n")
print("==> DONE!")

print("==> Check your files on word2vec directory!")
