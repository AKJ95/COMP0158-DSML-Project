def read_file_contents(file_path):
    with open(file_path, 'r') as file:
        contents = file.readlines()
    return contents


# Usage
file_path = '/Volumes/Buffalo/UCL/COMP0158/models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
contents = read_file_contents(file_path)
print(len(contents))
