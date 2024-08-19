# Specify the file path
file_path = "main.py"

# The line number where you want to insert the sentence (1-based index)
line_number = 34

# The sentence you want to insert
sentence_to_insert = "import models.convselfattn \n"

# Read the file and store its contents in a list
with open(file_path, "r") as file:
    lines = file.readlines()

# Close the file

# Insert the sentence at the desired line
lines.insert(line_number - 1, sentence_to_insert + "\n")  # Subtract 1 to convert to 0-based index

# Open the file in write mode and overwrite its contents
with open(file_path, "w") as file:
    file.writelines(lines)